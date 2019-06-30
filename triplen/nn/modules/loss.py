import numpy as np
from triplen.nn.modules.module import Module


class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        exp_output = np.exp(input)
        loss = np.sum(np.log(np.sum(exp_output, axis=1)) - input[np.arange(input.shape[0]), target])
        grad = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        grad[np.arange(input.shape[0]), target] -= 1

        return loss, grad

    def backward(self):
        raise NotImplementedError

