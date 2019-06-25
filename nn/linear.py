import numpy as np
from functools import reduce
from nn.module import Module
from tensor import Parameter


class Linear(Module):
    def __init__(self, shape, output_num=2):
        self.input_shape = shape
        self.batch_size = shape[0]

        input_len = reduce(lambda x, y: x * y, shape[1:])

        self.weights = Parameter(np.random.standard_normal((input_len, output_num)) / 100)
        self.bias = Parameter(np.random.standard_normal(output_num) / 100)

        self.output_shape = [self.batch_size, output_num]

    def forward(self, x):
        x = x.reshape([self.batch_size, -1])
        output = np.dot(self.x, self.weights.data) + self.bias
        return output

    def gradient(self, grad_output):
        for i in range(grad_output.shape[0]):
            col_x = self.x[i][:, np.newaxis]
            eta_i = grad_output[i][:, np.newaxis].T
            self.weights._grad += np.dot(col_x, eta_i)
            self.bias._grad += eta_i.reshape(self.bias.shape)

        next_grad = np.dot(grad_output, self.weights.data.T)
        next_grad = np.reshape(next_grad, self.input_shape)
        return next_grad


if __name__ == "__main__":
    img = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
    fc = Linear(img.shape, 2)
    out = fc.forward(img)

    fc.gradient(np.array([[1, -2], [3, 4]]))

    print(fc.w_gradient)
    print(fc.b_gradient)

    fc.backward(0.0001, 0.000001)
    print(fc.weights)
