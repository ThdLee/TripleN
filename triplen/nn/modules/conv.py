import math
import triplen
import numpy as np
from triplen.nn.parameter import Parameter
from triplen.nn.modules.module import Module
from .. import init
from numpy.lib.stride_tricks import as_strided


class Conv2D(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = Parameter(triplen.tensor(kernel_size, kernel_size, in_channels, out_channels))
        self.bias = Parameter(triplen.tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = np.pad(x, ((0, 0), (self.padding, self.padding),
                       (self.padding, self.padding), (0, 0)),
                   'constant', constant_values=0)
        view_shape = (x.shape[0], (x.shape[1] - self.kernel_size) // self.stride + 1,
                      (x.shape[2] - self.kernel_size) // self.stride + 1, x.shape[3])
        output = as_strided(x, shape=view_shape + (self.kernel_size, self.kernel_size),
                            strides=(x.strides[0], self.stride * x.strides[1],
                                     self.stride * x.strides[2], x.strides[3]) + x.strides[1:3])
        self.col_img = output.reshape((output.shape[0], -1, (self.kernel_size ** 2) * self.in_channels))
        output = np.matmul(output.reshape(output.shape[:3] + (-1,)),
                           self.weight.data.reshape((-1, self.out_channels))) + self.bias.data
        return output

    def backward(self, grad_output):
        col_grad_output = grad_output.reshape((grad_output.shape[0], -1, self.out_channels))

        self.weight.grad += np.matmul(self.col_img.swapaxes(1, 2),
                                      col_grad_output).sum(axis=0).reshape(self.weight.shape)
        self.bias.grad += np.sum(col_grad_output, axis=(0, 1))

        pad_grad = np.pad(grad_output, ((0, 0),
                                        (self.kernel_size - 1 - self.padding, self.kernel_size - 1 - self.padding),
                                        (self.kernel_size - 1 - self.padding, self.kernel_size - 1 - self.padding),
                                        (0, 0)), 'constant', constant_values=0)

        view_shape = (pad_grad.shape[0], (pad_grad.shape[1] - self.kernel_size) // self.stride + 1,
                      (pad_grad.shape[2] - self.kernel_size) // self.stride + 1, pad_grad.shape[3])
        output = as_strided(pad_grad, shape=view_shape + (self.kernel_size, self.kernel_size),
                            strides=(pad_grad.strides[0], self.stride * pad_grad.strides[1],
                                     self.stride * pad_grad.strides[2], pad_grad.strides[3]) + pad_grad.strides[1:3])
        weights = np.flipud(np.fliplr(self.weight.data)).swapaxes(2, 3).reshape((-1, self.in_channels))
        next_grad = np.matmul(output.reshape(output.shape[:3] + (-1,)), weights)
        return next_grad
