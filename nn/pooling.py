import numpy as np
from nn.module import Module
from tensor import Tensor


class AvgPooling(Module):
    def __init__(self, shape, ksize=2, stride=2):
        super(AvgPooling, self).__init__()
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = shape[-1]
        self.integral = Tensor(np.zeros(shape))
        self.index = Tensor(np.zeros(shape))

    def forward(self, x):
        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(x.shape[1]):
                    row_sum = 0
                    for j in range(x.shape[2]):
                        row_sum += x[b, i, j, c]
                        if i == 0:
                            self.integral.data[b, i, j, c] = row_sum
                        else:
                            self.integral.data[b, i, j, c] = self.integral.data[b, i - 1, j, c] + row_sum
        out = np.zeros([x.shape[0], x.shape[1] // self.stride, x.shape[2] // self.stride, self.output_channels],
                       dtype=float)

        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        self.index.data[b, i:i + self.ksize, j:j + self.ksize, c] = 1
                        if i == 0 and j == 0:
                            out[b, i // self.stride, j // self.stride, c] = self.integral.data[
                                b, self.ksize - 1, self.ksize - 1, c]
                        elif i == 0:
                            out[b, i // self.stride, j // self.stride, c] = self.integral.data[b, 1, j + self.ksize - 1, c] - \
                                                                            self.integral.data[b, 1, j - 1, c]
                        elif j == 0:
                            out[b, i // self.stride, j // self.stride, c] = self.integral.data[b, i + self.ksize - 1, 1, c] - \
                                                                            self.integral.data[b, i - 1, 1, c]
                        else:
                            out[b, i // self.stride, j // self.stride, c] = self.integral.data[
                                                                                b, i + self.ksize - 1, j + self.ksize - 1, c] - \
                                                                            self.integral.data[
                                                                                b, i - 1, j + self.ksize - 1, c] - \
                                                                            self.integral.data[
                                                                                b, i + self.ksize - 1, j - 1, c] + \
                                                                            self.integral.data[b, i - 1, j - 1, c]

        out /= (self.ksize * self.ksize)
        return out
    
    def backward(self, grad_output):
        next_grad = np.repeat(grad_output, self.stride, axis=1)
        next_grad = np.repeat(next_grad, self.stride, axis=2)
        next_grad = next_grad * self.index
        return next_grad / (self.ksize ** 2)


class MaxPooling(Module):
    def __init__(self, shape, ksize=2, stride=2):
        super(MaxPooling, self).__init__()
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = shape[-1]
        self.index = Tensor(np.zeros(shape))
        self.output_shape = [shape[0], shape[1] // self.stride, shape[2] // self.stride, self.output_channels]

    def forward(self, x):
        out = np.zeros([x.shape[0], x.shape[1] // self.stride, x.shape[2] // self.stride, self.output_channels])

        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        out[b, i // self.stride, j // self.stride, c] = np.max(
                            x[b, i:i + self.ksize, j:j + self.ksize, c])
                        index = np.argmax(x[b, i:i + self.ksize, j:j + self.ksize, c])
                        self.index.data[b, i + index // self.stride, j + index % self.stride, c] = 1
        return out

    def backward(self, grad_output):
        return np.repeat(np.repeat(grad_output, self.stride, axis=1), self.stride, axis=2) * self.index


if __name__ == "__main__":
    img = np.random.random((2, 28, 28, 3))

    pool = AvgPooling(img.shape, 2, 2)
    img1 = pool.forward(img)
    img2 = pool.gradient(img1)
    # print(img[1, :, :, 1])
    print(img1[1, :, :, 1])
    # print(img2[1, :, :, 1])
