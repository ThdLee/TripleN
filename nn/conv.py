import numpy as np
import math
from tensor import Parameter
from nn.module import Module


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

        # self.output_shape = ((shape[0] - ksize + 1) // self.stride,
        #                      (shape[1] - ksize + 1) // self.stride, self.out_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        col_weights = self.weights.data.reshape([-1, self.out_channels])
        x = np.pad(x, ((0, 0), (self.padding, self.padding),
                       (self.padding, self.padding), (0, 0)),
                   'constant', constant_values=0)
        output_shape = [batch_size, (x.shape[1] - self.kernel_size + 1) // self.stride,
                        (x.shape[2] - self.kernel_size + 1) // self.stride, self.out_channels]
        self.col_image = []
        conv_out = np.zeros(output_shape)
        for i in range(batch_size):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = im2col(img_i, self.kernel_size, self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias.data, output_shape[1:])
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out

    def backward(self, grad_output):
        batch_size = grad_output.shape[0]
        col_grad_output = np.reshape(grad_output, [batch_size, -1, self.out_channels])

        input_shape = [batch_size, (grad_output.shape[1] + self.kernel_size - 1 - self.padding * 2),
                       (grad_output.shape[2] + self.kernel_size - 1 - self.padding * 2), self.in_channels]

        for i in range(batch_size):
            self.weights.grad += np.dot(self.col_image[i].T, col_grad_output[i]).reshape(self.weights.shape)
        self.bias.grad += np.sum(col_grad_output, axis=(0, 1))

        pad_grad = np.pad(grad_output, ((0, 0), (self.kernel_size - 1 - self.padding, self.kernel_size - 1 - self.padding),
                                        (self.kernel_size - 1 - self.padding, self.kernel_size - 1 - self.padding), (0, 0)),
                          'constant', constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.weights.data))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.in_channels])
        col_pad_grad = np.array(
            [im2col(pad_grad[i][np.newaxis, :], self.kernel_size, self.stride) for i in range(batch_size)])
        next_grad = np.dot(col_pad_grad, col_flip_weights)
        next_grad = np.reshape(next_grad, input_shape)
        return next_grad


def im2col(inputs, ksize, stride):
    image_col = []
    for i in range(0, inputs.shape[1] - ksize + 1, stride):
        for j in range(0, inputs.shape[2] - ksize + 1, stride):
            col = inputs[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col
