import numpy as np
from numpy.lib.stride_tricks import as_strided
import triplen


class _FunctionBase(object):
    @classmethod
    def apply(cls, *inputs):
        ctx = cls._backward_cls()
        _inputs = tuple(x.data if isinstance(x, triplen.Tensor) else x for x in inputs)
        input_vars = tuple(x for x in inputs if isinstance(x, triplen.Tensor))
        needs_input_grad = tuple(isinstance(x, triplen.Tensor) and x._requires_grad for x in inputs)
        is_tensor_input = tuple(isinstance(x, triplen.Tensor) for x in inputs)
        next_functions = [None] * len(input_vars)
        for i, var in enumerate(input_vars):
            if var._grad_fn is not None and var._requires_grad:
                next_functions[i] = var._grad_fn
            elif var._requires_grad:
                next_functions[i] = var.get_grad_accumulator()
        ctx.next_functions = tuple(next_functions)
        ctx.needs_input_grad = needs_input_grad
        ctx.is_tensor_input = is_tensor_input

        ctx_tensor_input = (ctx, ) + _inputs
        tensor_output = cls.forward(*ctx_tensor_input)
        if not isinstance(tensor_output, np.ndarray):
            tensor_output = np.array(tensor_output)
        tensor_output = triplen.Tensor(tensor_output)
        if True in [x._requires_grad for x in input_vars]:
            tensor_output._grad_fn = ctx
            tensor_output._requires_grad = True
        return tensor_output


class _ContextMethodMixin(object):
    def save_for_backward(self, *tensors):
        self.to_save = tensors


class BackwardFunction(_FunctionBase, _ContextMethodMixin):
    _is_legacy = False

    def apply(self, *args):
        return self._forward_cls.backward(self, *args)


class AccumulateGrad(_FunctionBase):
    def __init__(self, tensor):
        self.tensor = tensor

    def apply(self, grad):
        self.tensor.grad += grad


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
        input_shape = x.shape
        view_shape = (x.shape[0], (x.shape[1] - kernel_size) // stride + 1,
                      (x.shape[2] - kernel_size) // stride + 1, x.shape[3])
        output = as_strided(x, shape=view_shape + (kernel_size, kernel_size),
                            strides=(x.strides[0], stride * x.strides[1],
                                     stride * x.strides[2], x.strides[3]) + x.strides[1:3])
        output = output.reshape((-1, kernel_size * kernel_size))
        index = np.zeros(output.shape)
        index[np.arange(index.shape[0]), output.argmax(axis=1)] = 1
        index = index.reshape(input_shape)
        ctx.save_for_backward(index)
        ctx.stride = stride
        output = output.max(axis=1).reshape(view_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        index, = ctx.to_save
        stride = ctx.stride
        next_grad = np.repeat(np.repeat(grad_output, stride, axis=1), stride, axis=2) * index
        return next_grad


class Conv2D(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding):
        kernel_size, in_channels, out_channels = weight.shape[0], weight.shape[-2], weight.shape[-1]
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=0)
        view_shape = (x.shape[0], (x.shape[1] - kernel_size) // stride + 1,
                      (x.shape[2] - kernel_size) // stride + 1, x.shape[3])
        output = as_strided(x, shape=view_shape + (kernel_size, kernel_size),
                            strides=(x.strides[0], stride * x.strides[1],
                                     stride * x.strides[2], x.strides[3]) + x.strides[1:3])
        col_img = output.reshape((output.shape[0], -1, (kernel_size ** 2) * in_channels))
        ctx.padding = padding
        ctx.stride = stride
        ctx.save_for_backward(col_img, weight, bias)
        output = np.matmul(output.reshape(output.shape[:3] + (-1,)),
                           weight.reshape((-1, out_channels))) + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        col_img, weight, bias = ctx.to_save
        padding, stride = ctx.padding, ctx.stride
        kernel_size, in_channels, out_channels = weight.shape[0], weight.shape[-2], weight.shape[-1]

        col_grad_output = grad_output.reshape((grad_output.shape[0], -1, out_channels))

        grad_weight = np.matmul(col_img.swapaxes(1, 2),
                                col_grad_output).sum(axis=0).reshape(weight.shape)
        grad_bias = np.sum(col_grad_output, axis=(0, 1))

        pad_grad = np.pad(grad_output, ((0, 0),
                                        (kernel_size - 1 - padding, kernel_size - 1 - padding),
                                        (kernel_size - 1 - padding, kernel_size - 1 - padding),
                                        (0, 0)), 'constant', constant_values=0)

        view_shape = (pad_grad.shape[0], (pad_grad.shape[1] - kernel_size) // stride + 1,
                      (pad_grad.shape[2] - kernel_size) // stride + 1, pad_grad.shape[3])
        output = as_strided(pad_grad, shape=view_shape + (kernel_size, kernel_size),
                            strides=(pad_grad.strides[0], stride * pad_grad.strides[1],
                                     stride * pad_grad.strides[2], pad_grad.strides[3]) + pad_grad.strides[1:3])
        weights = np.flipud(np.fliplr(weight)).swapaxes(2, 3).reshape((-1, in_channels))
        next_grad = np.matmul(output.reshape(output.shape[:3] + (-1,)), weights)
        return next_grad, grad_weight, grad_bias


class Linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        assert input.shape[-1] == weight.shape[0]
        batch_size = input.shape[0]
        output_size = weight.shape[-1]
        ctx.input_shape = input.shape
        input = input.reshape(batch_size, -1)
        ctx.save_for_backward(input, weight)
        output = np.matmul(input, weight) + bias
        return output.reshape(input.shape[:-1] + (output_size,))

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.to_save
        input_shape = ctx.input_shape

        grad_weight = np.matmul(input.T, grad_output)
        grad_bias = grad_output.sum(axis=0)

        next_grad = np.matmul(grad_output, weight.T)
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
        if training:
            drop = np.random.binomial(1, 1 - prob, x.shape)
        else:
            drop = np.ones(x.shape)
        ctx.save_for_backward(drop)
        return drop * x

    @staticmethod
    def backward(ctx, grad_output):
        drop, = ctx.to_save
        return drop * grad_output


class Softmax(Function):
    @staticmethod
    def forward(ctx, x, dim):
        exp_x = np.exp(x)
        output = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        softmax, = ctx.to_save
        dim = ctx.dim
        grad = softmax - softmax * np.sum(softmax, axis=dim, keepdims=True)
        return grad * grad_output


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, x, dim):
        exp_x = np.exp(x)
        softmax = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
        # log_softmax = x - np.log(np.sum(exp_x, axis=dim, keepdims=True))
        ctx.save_for_backward(softmax)
        return np.log(softmax)

    @staticmethod
    def backward(ctx, grad_output):
        softmax, = ctx.to_save
        grad = softmax
        return grad + grad_output


class NLLLoss(Function):
    @staticmethod
    def forward(ctx, input, target):
        assert input.ndim == 2
        assert target.ndim == 1
        loss = -input[np.arange(input.shape[0]), target]
        ctx.save_for_backward(target)
        ctx.shape = input.shape
        return np.mean(loss)

    @staticmethod
    def backward(ctx, grad_output=None):
        target, = ctx.to_save
        shape = ctx.shape
        next_grad = np.zeros(shape)
        next_grad[np.arange(shape[0]), target] = -1
        return next_grad


class CrossEntropyLoss(Function):
    @staticmethod
    def forward(ctx, input, target):
        exp_output = np.exp(input)
        ctx.save_for_backward(exp_output, input, target)
        return np.mean(np.log(np.sum(exp_output, axis=1)) - input[np.arange(input.shape[0]), target])

    @staticmethod
    def backward(ctx, grad_output=None):
        exp_output, input, target = ctx.to_save
        next_grad = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        next_grad[np.arange(input.shape[0]), target] -= 1
        return next_grad
