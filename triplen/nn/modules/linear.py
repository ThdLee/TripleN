import math
import triplen
import numpy as np
from .module import Module
from triplen.nn.parameter import Parameter
from .. import init


class Linear(Module):
    def __init__(self, input_size: int, output_size: int):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weight = Parameter(triplen.tensor(input_size, output_size))
        self.bias = Parameter(triplen.tensor(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size = x.shape[0]
        self.input_shape = x.shape
        self.x = x.reshape(batch_size, -1)
        output = np.matmul(self.x, self.weight.data) + self.bias.data
        return output.reshape((batch_size, self.output_size))

    def backward(self, grad_output):
        col_x = self.x[:, :, np.newaxis]
        grad = grad_output[:, np.newaxis, :]
        self.weight.grad += np.matmul(col_x, grad).sum(axis=0)
        self.bias.grad += grad_output.sum(axis=0).reshape(self.bias.shape)

        next_grad = np.dot(grad_output, self.weight.data.T)
        next_grad = next_grad.reshape(self.input_shape)
        return next_grad
