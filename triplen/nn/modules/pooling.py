import numpy as np
from triplen.nn.modules.module import Module
from numpy.lib.stride_tricks import as_strided


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

    def forward(self, x):
        input_shape = x.shape
        view_shape = (x.shape[0], (x.shape[1] - self.kernel_size) // self.stride + 1,
                      (x.shape[2] - self.kernel_size) // self.stride + 1, x.shape[3])
        output = as_strided(x, shape=view_shape + (self.kernel_size, self.kernel_size),
                            strides=(x.strides[0], self.stride * x.strides[1],
                                     self.stride * x.strides[2], x.strides[3]) + x.strides[1:3])
        output = output.reshape((-1, self.kernel_size * self.kernel_size))
        self.index = np.zeros(output.shape)
        self.index[np.arange(self.index.shape[0]), output.argmax(axis=1)] = 1
        self.index = self.index.reshape(input_shape)
        output = output.max(axis=1).reshape(view_shape)
        return output

    def backward(self, grad_output):
        return np.repeat(np.repeat(grad_output, self.stride, axis=1), self.stride, axis=2) * self.index
