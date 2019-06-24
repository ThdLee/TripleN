import numpy as np
import math
from functools import reduce
import torch.nn as nn

nn.Module

class Conv2D(object):
    def __init__(self, shape: tuple, output_channels: int, ksize: int = 3, stride: int = 1, method: str = 'VALID'):
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.batch_size = shape[0]
        self.stride = stride
        self.ksize = ksize
        self.method = method

        weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / self.output_channels)
        self.weights = np.random.standard_normal(
            (ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale

        if method == 'VALID':
            self.eta = np.zeros((shape[0], (shape[1] - ksize + 1) // self.stride,
                                 (shape[1] - ksize + 1) // self.stride, self.output_channels))
        elif method == 'SAME':
            self.eta = np.zeros((shape[0], shape[1] // self.stride,
                                 shape[2] // self.stride, self.output_channels))

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape

    def forward(self, x):
        col_weights = self.weights.reshape([-1, self.output_channels])
        if self.method == 'SAME':
            x = np.pad(x, ((0, 0), (self.ksize // 2, self.ksize // 2), (0, 0)),
                       'constant', constant_values=0)

        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batch_size):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out

    def gradient(self, eta):
        self.eta = eta
        col_eta = np.reshape(eta, [self.batch_size, -1, self.output_channels])

        for i in range(self.batch_size):
            self.w_gradient += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        if self.method == 'VALID':
            pad_eta = np.pad(self.eta, ((0, 0), (self.ksize - 1, self.ksize - 1),
                                        (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)
        elif self.method == 'SAME':
            pad_eta = np.pad(self.eta, ((0, 0), (self.ksize // 2, self.ksize // 2),
                                        (self.ksize // 2, self.ksize // 2), (0, 0)),
                             'constant', constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_eta = np.array([im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batch_size)])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def backward(self, alpha, weight_decay):
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias

        self.w_gradient.fill(0)
        self.b_gradient.fill(0)


def im2col(inputs, ksize, stride):
    image_col = []
    for i in range(0, inputs.shape[1] - ksize + 1, stride):
        for j in range(0, inputs.shape[2] - ksize + 1, stride):
            col = inputs[:, i:i+ksize, j:j+ksize, :].reshape([-1])
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
    conv.gradient(next1-next)
    print(conv.w_gradient)
    print(conv.b_gradient)
    conv.backward(0.001, 0.000001)