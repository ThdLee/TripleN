import numpy as np
from numpy.lib.stride_tricks import as_strided
import triplen

class _FunctionBase(object):
    @classmethod
    def apply(cls, *inputs):
        ctx = cls._backward_cls()
        _inputs = tuple([x.data if isinstance(x, triplen.Tensor) else x for x in inputs])
        input_vars = tuple([x for x in inputs if isinstance(x, triplen.Tensor)])
        needs_input_grad = tuple([isinstance(x, triplen.Tensor) and x.requires_grad for x in inputs])
        is_tensor_input = tuple([isinstance(x, triplen.Tensor) for x in inputs])
        next_functions = [None] * len(input_vars)
        batch_size = 1
        for i, var in enumerate(input_vars):
            if var.grad_fn is not None and var.requires_grad:
                batch_size = var.shape[0]
                next_functions[i] = var.grad_fn
            elif var.requires_grad:
                next_functions[i] = var.get_grad_accumulator(batch_size)

        ctx.next_functions = tuple(next_functions)
        ctx.needs_input_grad = needs_input_grad
        ctx.is_tensor_input = is_tensor_input

        ctx_tensor_input = (ctx, ) + _inputs
        tensor_output = cls.forward(*ctx_tensor_input)
        if not isinstance(tensor_output, np.ndarray):
            tensor_output = np.array(tensor_output)
        tensor_output = triplen.Tensor(tensor_output)
        ctx.batch_size = tensor_output.size(0)
        if True in [x.requires_grad for x in input_vars]:
            tensor_output.requires_grad = True
            tensor_output.grad_fn = ctx
        return tensor_output


class _ContextMethodMixin(object):
    def save_for_backward(self, *tensors):
        self.to_save = tensors


class BackwardFunction(_FunctionBase, _ContextMethodMixin):
    _is_legacy = False

    def apply(self, *args):
        return self._forward_cls.backward(self, *args)


class AccumulateGrad(_FunctionBase):
    def __init__(self, tensor, batch_size):
        self.tensor = tensor
        self.batch_size = batch_size

    def apply(self, grad):
        self.tensor.grad += grad # / self.batch_size


class FunctionMeta(type):
    def __init__(cls, name, bases, attrs):
        for super_cls in cls.mro():
            forward = super_cls.__dict__.get('forward')
            if forward is not None:
                has_static_forward = isinstance(forward, staticmethod) or isinstance(forward, classmethod)
                break

        setattr(cls, '_is_legacy', not has_static_forward)

        # old-style functions
        if not has_static_forward:
            super(FunctionMeta, cls).__init__(name, bases, attrs)

        backward_fn = type(name + 'Backward', (BackwardFunction,), {'_forward_cls': cls})
        setattr(cls, '_backward_cls', backward_fn)

        super(FunctionMeta, cls).__init__(name, bases, attrs)


class Function(_FunctionBase, _ContextMethodMixin, metaclass=FunctionMeta):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError


class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x + y

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output * 1
        grad_y = grad_output * 1
        return grad_x, grad_y


class Sub(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x - y

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output * 1
        grad_y = grad_output * -1
        return grad_x, grad_y


class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.to_save()
        grad_x = grad_output * y
        grad_y = grad_output * x
        return grad_x, grad_y


class Div(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x / y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.to_save()
        grad_x = grad_output / y
        grad_y = grad_output * (-x / (y ** 2))
        return grad_x, grad_y


class Pow(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x ** y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.to_save()
        grad_x = y * (grad_output ** (y - 1))
        grad_y = (x ** y) * np.log(x)
        return grad_x, grad_y


class Log(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.log(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.to_save()
        next_grad = grad_output * (1 / x)
        return next_grad


class MatMul(Function):
    @staticmethod
    def forward(ctx, x, y):
        assert x.shape[-1] == y.shape[0]
        ctx.save_for_backward(x, y)
        output = np.matmul(x, y)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.to_save
        grad_y = np.matmul(x.T, grad_output)
        grad_x = np.matmul(grad_output, y.T)
        return grad_x, grad_y


class View(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.shape = x.shape
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.shape
        return grad_output.reshape(shape)


class Select(Function):
    @staticmethod
    def forward(ctx, x, *args):
        output = x[args]
        ctx.shape = x.shape
        ctx.args = args
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.shape
        args = ctx.args
        next_grad = np.ones(shape)
        next_grad[args] = grad_output
        return next_grad


class CopySlices(Function):
    @staticmethod
    def forward(ctx, x, y, *args):
        x[args] = y
        ctx.args = args
        return x

    @staticmethod
    def backward(ctx, grad_output):
        args = ctx.args
        next_grad = grad_output[args]
        return next_grad


class Index(Function):
    @staticmethod
    def forward(ctx, x, index_num, *args):
        output = x[args]
        ctx.shape = x.shape
        ctx.args = args
        ctx.index_num = index_num
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.shape
        args = ctx.args
        index_num = ctx.index_num
        next_grad = np.zeros(shape)
        for i in range(index_num):
            grad_args = tuple([arg[i] if isinstance(arg, list) else arg for arg in args])
            next_grad[grad_args] += grad_output[i]
        return next_grad


class IndexPut(Function):
    @staticmethod
    def forward(ctx, x, y, *args):
        x[args] = y
        ctx.args = args
        return x

    @staticmethod
    def backward(ctx, grad_output):
        args = ctx.args
        return grad_output[args]


class MaxPooling(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride):
        output = x.reshape(x.shape[0], x.shape[1],
                           x.shape[2] // stride, stride,
                           x.shape[3] // stride, stride)
        output = output[:, :, :, :kernel_size, :, :kernel_size].max(axis=(3, 5))
        mask = output.repeat(stride, axis=2).repeat(stride, axis=3) != x
        ctx.save_for_backward(mask)
        ctx.stride = stride
        ctx.kernel_size = kernel_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.to_save
        stride = ctx.stride
        next_grad = grad_output.repeat(stride, axis=2).repeat(stride, axis=3)
        next_grad[mask] = 0
        return next_grad


class Conv2D(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding):
        out_channels, in_channels, kernel_size = weight.shape[0], weight.shape[1], weight.shape[-1]
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=0)

        H, W = (x.shape[2] - kernel_size) // stride + 1, (x.shape[3] - kernel_size) // stride + 1
        shape = (x.shape[0], x.shape[1], H, W, kernel_size, kernel_size)
        strides = (x.strides[0], x.strides[1], stride * x.strides[2], stride * x.strides[3], *x.strides[2:])
        output = as_strided(x, shape=shape, strides=strides).transpose([0, 2, 3, 1, 4, 5])

        col_img = output.reshape((output.shape[0], -1, (kernel_size ** 2) * in_channels))
        ctx.padding = padding
        ctx.stride = stride
        ctx.save_for_backward(col_img, weight, bias)
        output = np.matmul(col_img, weight.reshape((out_channels, -1)).swapaxes(0, 1)) + bias
        output = output.swapaxes(1, 2).reshape(-1, out_channels, H, W)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        col_img, weight, bias = ctx.to_save
        padding, stride = ctx.padding, ctx.stride
        out_channels, in_channels, kernel_size = weight.shape[0], weight.shape[1], weight.shape[-1]

        col_grad_output = grad_output.reshape((grad_output.shape[0], out_channels, -1))
        grad_weight = np.matmul(col_grad_output, col_img).sum(axis=0).reshape(out_channels, in_channels, kernel_size, kernel_size) / grad_output.shape[0]
        grad_bias = np.sum(col_grad_output, axis=(0, 2)) / grad_output.shape[0]

        P = kernel_size - 1 - padding
        pad_grad = np.pad(grad_output, ((0, 0), (0, 0), (P, P), (P, P)), 'constant')
        H, W = (pad_grad.shape[2] - kernel_size) // stride + 1, (pad_grad.shape[3] - kernel_size) // stride + 1
        shape = (pad_grad.shape[0], pad_grad.shape[1], H, W, kernel_size, kernel_size)
        strides = (pad_grad.strides[0], pad_grad.strides[1],
                   stride * pad_grad.strides[2], stride * pad_grad.strides[3], *pad_grad.strides[2:])
        grad_strided = as_strided(pad_grad, shape=shape, strides=strides).transpose([0, 2, 3, 1, 4, 5])
        grad_strided = grad_strided.reshape((grad_strided.shape[0], -1, (kernel_size ** 2) * out_channels))

        weights = weight[:, :, ::-1, ::-1].transpose([0, 2, 3, 1]).reshape((-1, in_channels))
        next_grad = np.matmul(grad_strided, weights).swapaxes(1, 2).reshape((-1, in_channels, H, W))
        return next_grad, grad_weight, grad_bias


class Linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        assert input.shape[-1] == weight.shape[0]
        hidden_size = input.shape[-1]
        output_size = weight.shape[-1]
        ctx.input_shape = input.shape
        input = input.reshape(-1, hidden_size)
        ctx.save_for_backward(input, weight)
        output = np.dot(input, weight) + bias
        return output.reshape(input.shape[:-1] + (output_size,))

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.to_save
        input_shape = ctx.input_shape
        batch_size = grad_output.shape[0]

        grad_weight = np.einsum('ji,jk->ik', input, grad_output, optimize=True) / batch_size
        grad_bias = np.einsum('i...->...', grad_output, optimize=True) / batch_size

        next_grad = np.einsum('ij,kj->ik', grad_output, weight, optimize=True)
        next_grad = next_grad.reshape(input_shape)
        return next_grad, grad_weight, grad_bias


class Relu(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.maximum(x, 0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.to_save
        grad_output[x < 0] = 0
        return grad_output


class Dropout(Function):
    @staticmethod
    def forward(ctx, x, prob, training):
        ctx.training = training
        if not training:
            return x
        else:
            drop = np.random.binomial(1, 1 - prob, x.shape)
        fix_value = 1 / (1 - prob)
        ctx.save_for_backward(drop)
        return drop * x * fix_value

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.training:
            return grad_output
        drop, = ctx.to_save
        return drop * grad_output


class Softmax(Function):
    @staticmethod
    def forward(ctx, x, dim):
        exp_x = np.exp(x - x.max(axis=dim, keepdims=True))
        exp_x /= exp_x.sum(axis=dim, keepdims=True)
        ctx.save_for_backward(exp_x)
        ctx.dim = dim
        return exp_x

    @staticmethod
    def backward(ctx, grad_output):
        softmax, = ctx.to_save
        return softmax * (grad_output - grad_output * softmax)


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, x, dim):
        exp_x = np.exp(x - x.max(axis=dim, keepdims=True))
        exp_x /= exp_x.sum(axis=dim, keepdims=True)
        ctx.save_for_backward(exp_x)
        ctx.dim = dim
        return np.log(exp_x)

    @staticmethod
    def backward(ctx, grad_output):
        softmax, = ctx.to_save
        return grad_output - softmax * grad_output


class CrossEntropyLoss(Function):
    @staticmethod
    def forward(ctx, input, target):
        target = np.eye(input.shape[-1])[target.reshape(-1)]
        exp_x = np.exp(input - input.max(axis=-1, keepdims=True))
        exp_x /= exp_x.sum(axis=-1, keepdims=True)
        loss = -1 * np.einsum('ij,ij->', target, np.log(exp_x), optimize=True) / target.shape[0]
        ctx.save_for_backward(exp_x, target)
        return loss

    @staticmethod
    def backward(ctx, grad_output=None):
        input, target = ctx.to_save
        next_grad = np.copy(input)
        next_grad -= target
        return next_grad
