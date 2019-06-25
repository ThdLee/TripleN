import numpy as np
import math
from functools import reduce
from tensor import Parameter
from nn.module import Module


class Conv2D(Module):
    def __init__(self, shape: tuple, output_channels: int, ksize: int = 3, stride: int = 1, padding: int = 0):
        super(Conv2D, self).__init__()
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.batch_size = shape[0]
        self.stride = stride
        self.ksize = ksize
        self.padding = padding

        weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / self.output_channels)
        self.weights = Parameter(np.random.standard_normal(
            (ksize, ksize, self.input_channels, self.output_channels)) / weights_scale)
        self.bias = Parameter(np.random.standard_normal(self.output_channels) / weights_scale)

        self.output_shape = (shape[0], (shape[1] - ksize + 1) // self.stride,
                             (shape[1] - ksize + 1) // self.stride, self.output_channels)

    def forward(self, x):
        col_weights = self.weights.data.reshape([-1, self.output_channels])
        x = np.pad(x, ((0, 0), (self.padding, self.padding),
                       (self.padding, self.padding), (0, 0)),
                   'constant', constant_values=0)

        self.col_image = []
        conv_out = np.zeros(self.output_shape)
        for i in range(self.batch_size):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias.data, self.output_shape[1:])
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out

    def backward(self, grad_output):
        col_grad_output = np.reshape(grad_output, [self.batch_size, -1, self.output_channels])

        for i in range(self.batch_size):
            self.weights._grad += np.dot(self.col_image[i].T, col_grad_output[i]).reshape(self.weights.shape)
        self.bias._grad += np.sum(col_grad_output, axis=(0, 1))

        pad_grad = np.pad(grad_output, ((0, 0), (self.ksize - 1 - self.padding, self.ksize - 1 - self.padding),
                                        (self.ksize - 1 - self.padding, self.ksize - 1 - self.padding), (0, 0)),
                          'constant', constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.weights.data))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_grad = np.array(
            [im2col(pad_grad[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batch_size)])
        next_grad = np.dot(col_pad_grad, col_flip_weights)
        next_grad = np.reshape(next_grad, self.input_shape)
        return next_grad


def im2col(inputs, ksize, stride):
    image_col = []
    for i in range(0, inputs.shape[1] - ksize + 1, stride):
        for j in range(0, inputs.shape[2] - ksize + 1, stride):
            col = inputs[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col


if __name__ == "__main__":
    # img = np.random.standard_normal((2, 32, 32, 3))
    img = np.ones((1, 32, 32, 3))
    img *= 2
    conv = Conv2D(img.shape, 12, 3, 1)
    next = conv.forward(img)
    next1 = next.copy() + 1
    conv.backward(next1 - next)
