import numpy as np
from numpy.lib.stride_tricks import as_strided
from triplen import Tensor


class _FunctionBase(object):
    @classmethod
    def apply(cls, *inputs):
        ctx = cls._backward_cls()

        _inputs = inputs
        input_vars = tuple(x for x in inputs if isinstance(x, Tensor))
        needs_input_grad = tuple(True if isinstance(x, Tensor) and x.requires_grad and x.grad_fn else False for x in inputs)
        is_tensor_input = tuple(True if isinstance(x, Tensor) else False for x in inputs)
        next_functions = [None] * len(input_vars)
        for i, var in enumerate(input_vars):
            if var.grad_fn is not None:
                next_functions[i] = var.grad_fn
            else:
                next_functions[i] = var.get_grad_accumulator()
        ctx.next_functions = tuple(next_functions)
        ctx.needs_input_grad = needs_input_grad
        ctx.is_tensor_input = is_tensor_input

        ctx_tensor_input = (ctx, ) + _inputs
        tensor_output = cls.forward(*ctx_tensor_input)
        tensor_output.grad_fn = ctx
        return tensor_output


class _ContextMethodMixin(object):
    def save_for_backward(self, *tensors):
        self.to_save = tensors


class BackwardFunction(_FunctionBase, _ContextMethodMixin):
    _is_legacy = False

    def apply(self, *args):
        return self._forward_cls.backward(self, *args)


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


class Subtract(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x - y

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output * 1
        grad_y = grad_output * -1
        return grad_x, grad_y


class Multiply(Function):
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


class Divide(Function):
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


class View(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.shape = shape
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.shape
        return grad_output.reshape(shape)


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
        index = ctx.to_save
        stride = ctx.stride
        return np.repeat(np.repeat(grad_output, stride, axis=1), stride, axis=2) * index


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
                           weight.data.reshape((-1, out_channels))) + bias.data
        return output

    @staticmethod
    def backward(ctx, grad_output):
        col_img, weight, bias = ctx.to_save()
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
        weights = np.flipud(np.fliplr(weight.data)).swapaxes(2, 3).reshape((-1, in_channels))
        next_grad = np.matmul(output.reshape(output.shape[:3] + (-1,)), weights)
        return next_grad, grad_weight, grad_bias


class Linear(Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        assert x.shape[-1] == weight.shape[0]
        batch_size = x.shape[0]
        output_size = weight.shape[-1]
        ctx.save_for_backward(x, weight)
        ctx.shape = x.shape
        output = np.matmul(x.reshape(batch_size, -1), weight.data) + bias.data
        return output.reshape((batch_size, output_size))

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.to_save
        shape = ctx.shape
        col_x = x[:, :, np.newaxis]
        grad = grad_output[:, np.newaxis, :]
        grad_weight = np.matmul(col_x, grad).sum(axis=0)
        grad_bias = grad_output.sum(axis=0)

        next_grad = np.dot(grad_output, weight.data.T)
        next_grad = next_grad.reshape(shape)
        return next_grad, grad_weight, grad_bias


class Relu(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.maximum(x, 0)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.to_save
        grad_output[x < 0] = 0
        return grad_output


class cross_entropy_loss(Function):
    @staticmethod
    def forward(ctx, input, target):
        exp_output = np.exp(input)
        ctx.save_for_backward(exp_output, input, target)
        return np.sum(np.log(np.sum(exp_output, axis=1)) - input[np.arange(input.shape[0]), target])

    @staticmethod
    def backward(ctx):
        exp_output, input, target = ctx.to_save
        grad = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        grad[np.arange(input.shape[0]), target] -= 1
        return grad
