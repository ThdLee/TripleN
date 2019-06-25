import numpy as np
from nn.module import Module
from tensor import Tensor


class Relu(Module):
    def __init__(self, shape):
        super(Relu, self).__init__()
        self.output_shape = shape
        self.buffer = Tensor(np.zeros(shape))

    def forward(self, x):
        self.buffer.data = x
        return np.maximum(x, 0)

    def backward(self, grad_output):
        grad_output[self.buffer.data < 0] = 0
        return grad_output
