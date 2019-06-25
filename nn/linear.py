import numpy as np
from functools import reduce
from nn.module import Module
from tensor import Parameter


class Linear(Module):
    def __init__(self, input_size: int, output_size: int):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weights = Parameter(np.random.standard_normal((input_size, output_size)) / 100)
        self.bias = Parameter(np.random.standard_normal(output_size) / 100)

    def forward(self, x):
        batch_size = x.shape[0]
        self.input_shape = x.shape
        self.x = x.reshape([batch_size, -1])
        output = np.dot(x, self.weights.data) + self.bias.data
        return np.reshape(output, x.shape[:-1] + [self.output_size])

    def backward(self, grad_output):
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
