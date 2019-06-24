import numpy as np
import math
from functools import reduce


class Linear(object):
    def __init__(self, shape, output_num=2):
        self.input_shape = shape
        self.batch_size = shape[0]

        input_len = reduce(lambda x, y: x * y, shape[1:])

        self.weights = np.random.standard_normal((input_len, output_num)) / 100
        self.bias = np.random.standard_normal(output_num) / 100

        self.output_shape = [self.batch_size, output_num]
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def forward(self, x):
        self.x = x.reshape([self.batch_size, -1])
        output = np.dot(self.x, self.weights) + self.bias
        return output

    def gradient(self, eta):
        for i in range(eta.shape[0]):
            col_x = self.x[i][:, np.newaxis]
            eta_i = eta[i][:, np.newaxis].T
            self.w_gradient += np.dot(col_x, eta_i)
            self.b_gradient += eta_i.reshape(self.bias.shape)

        next_eta = np.dot(eta, self.weights.T)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def backward(self, alpha, weight_decay):
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.b_gradient

        self.w_gradient.fill(0)
        self.b_gradient.fill(0)


if __name__ == "__main__":
    img = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
    fc = Linear(img.shape, 2)
    out = fc.forward(img)

    fc.gradient(np.array([[1, -2], [3, 4]]))

    print(fc.w_gradient)
    print(fc.b_gradient)

    fc.backward(0.0001, 0.000001)
    print(fc.weights)
