import numpy as np
from nn.module import Module
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
        output = output.reshape(-1, self.kernel_size * self.kernel_size)
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
        output_shape = (x.shape[0], (x.shape[1] - self.kernel_size) // self.stride + 1,
                        (x.shape[2] - self.kernel_size) // self.stride + 1, x.shape[3])
        output = as_strided(x, shape=output_shape + (self.kernel_size, self.kernel_size),
                            strides=(x.strides[0], self.stride * x.strides[1],
                                     self.stride * x.strides[2], x.strides[3]) + x.strides[1:3])
        output = output.reshape(-1, self.kernel_size * self.kernel_size)
        self.index = np.zeros(output.shape)
        self.index[np.arange(self.index.shape[0]), output.argmax(axis=1)] = 1
        self.index = self.index.reshape(input_shape)
        output = output.max(axis=1).reshape(output_shape)
        return output

    def backward(self, grad_output):
        return np.repeat(np.repeat(grad_output, self.stride, axis=1), self.stride, axis=2) * self.index


if __name__ == "__main__":
    A = np.random.random((16, 28, 28, 3))
    kernel_size = 2
    stride = 2
    output_shape = (16, (A.shape[1] - kernel_size) // stride + 1,
                    (A.shape[2] - kernel_size) // stride + 1, 3)
    kernel_sizes = (kernel_size, kernel_size)
    # print((stride * A.strides[0], stride * A.strides[1]))
    # print(A.strides)
    # print((stride * A.strides[0],
    #                           stride * A.strides[1]) + A.strides)
    A_w = as_strided(A, shape=output_shape + kernel_sizes,
                     strides=(A.strides[0], stride * A.strides[1],
                              stride * A.strides[2]) + A.strides[1:])
    A_w = A_w.reshape(-1, kernel_size * kernel_size)
    print(A_w.argmax(axis=1).reshape(output_shape))
    print(A_w)