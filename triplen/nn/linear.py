import numpy as np
from triplen.nn.module import Module
from triplen.tensor import Parameter


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
        self.x = x.reshape(batch_size, -1)
        output = np.matmul(self.x, self.weights.data) + self.bias.data
        return output.reshape((batch_size, self.output_size))

    def backward(self, grad_output):
        col_x = self.x[:, :, np.newaxis]
        grad = grad_output[:, np.newaxis, :]
        self.weights.grad += np.matmul(col_x, grad).sum(axis=0)
        self.bias.grad += grad_output.sum(axis=0).reshape(self.bias.shape)

        next_grad = np.dot(grad_output, self.weights.data.T)
        next_grad = next_grad.reshape(self.input_shape)
        return next_grad
