import numpy as np
from triplen.nn.modules.module import Module
from numpy.lib.stride_tricks import as_strided
from .. import functional as F


class AvgPooling2D(Module):
    def __init__(self, kernel_size=2, stride=None):
        super(AvgPooling2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x):
        output_shape = (x.shape[0], (x.shape[1] - self.kernel_size) // self.stride + 1,
                        (x.shape[2] - self.kernel_size) // self.stride + 1, x.shape[3])
        output = as_strided(x, shape=output_shape + (self.kernel_size, self.kernel_size),
                            strides=(x.strides[0], self.stride * x.strides[1],
                                     self.stride * x.strides[2], x.strides[3]) + x.strides[1:3])
        output = output.reshape((-1, self.kernel_size * self.kernel_size))
        output = output.mean(axis=1).reshape(output_shape)
        return output

    def backward(self, grad_output):
        return np.repeat(np.repeat(grad_output, self.stride, axis=1), self.stride, axis=2) / (self.kernel_size ** 2)


class MaxPooling2D(Module):
    def __init__(self, kernel_size=2, stride=None):
        super(MaxPooling2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, input):
        return F.maxpooling(input, self.kernel_size, self.stride)
