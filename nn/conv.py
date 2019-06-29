import numpy as np
import math
from tensor import Parameter
from nn.module import Module
from numpy.lib.stride_tricks import as_strided


class Conv2D(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

        weights_scale = math.sqrt(in_channels * kernel_size * kernel_size / out_channels)
        self.weights = Parameter(np.random.standard_normal(
            (kernel_size, kernel_size, in_channels, out_channels)) / weights_scale)
        self.bias = Parameter(np.random.standard_normal(out_channels) / weights_scale)

    def forward(self, x):
        x = np.pad(x, ((0, 0), (self.padding, self.padding),
                       (self.padding, self.padding), (0, 0)),
                   'constant', constant_values=0)
        view_shape = (x.shape[0], (x.shape[1] - self.kernel_size) // self.stride + 1,
                      (x.shape[2] - self.kernel_size) // self.stride + 1, x.shape[3])
        output = as_strided(x, shape=view_shape + (self.kernel_size, self.kernel_size),
                            strides=(x.strides[0], self.stride * x.strides[1],
                                     self.stride * x.strides[2], x.strides[3]) + x.strides[1:3])
        self.col_img = output.reshape(output.shape[0], -1, self.kernel_size ** 2 * self.in_channels)
        output = np.matmul(output.reshape(output.shape[:3] + (-1,)),
                           self.weights.data.reshape([-1, self.out_channels])) + self.bias.data
        # batch_size = x.shape[0]
        # output_shape = [batch_size, (x.shape[1] - self.kernel_size + 1) // self.stride,
        #                 (x.shape[2] - self.kernel_size + 1) // self.stride, self.out_channels]
        # self.col_image = []
        # conv_out = np.zeros(output_shape)
        # for i in range(batch_size):
        #     img_i = x[i][np.newaxis, :]
        #     self.col_image_i = im2col(img_i, self.kernel_size, self.stride)
        #     conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias.data, output_shape[1:])
        #     self.col_image.append(self.col_image_i)
        # self.col_image = np.array(self.col_image)
        # print(self.col_image.shape)
        return output

    def backward(self, grad_output):
        col_grad_output = grad_output.reshape(grad_output.shape[0], -1, self.out_channels)

        self.weights.grad += np.matmul(self.col_img.swapaxes(1, 2),
                                       col_grad_output).sum(axis=0).reshape(self.weights.shape)
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
        weights = np.flipud(np.fliplr(self.weights.data)).swapaxes(2, 3).reshape([-1, self.in_channels])
        next_grad = np.matmul(output.reshape(output.shape[:3] + (-1,)), weights)
        # flip_weights = np.flipud(np.fliplr(self.weights.data))
        # flip_weights = flip_weights.swapaxes(2, 3)
        # col_flip_weights = flip_weights.reshape([-1, self.in_channels])
        # col_pad_grad = np.array(
        #     [im2col(pad_grad[i][np.newaxis, :], self.kernel_size, self.stride) for i in range(batch_size)])
        # next_grad = np.dot(col_pad_grad, col_flip_weights)
        # next_grad = np.reshape(next_grad, input_shape)
        return next_grad


def im2col(inputs, ksize, stride):
    image_col = []
    for i in range(0, inputs.shape[1] - ksize + 1, stride):
        for j in range(0, inputs.shape[2] - ksize + 1, stride):
            col = inputs[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)
    print(image_col.shape)

    return image_col
