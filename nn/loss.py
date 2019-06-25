import numpy as np
from nn.module import Module


class CrossEntropyLoss(Module):
    def __init__(self, shape):
        super(CrossEntropyLoss, self).__init__()
        self.softmax = np.zeros(shape)
        self.eta = np.zeros(shape)
        self.batch_size = shape[0]

    def forward(self, input, target):
        batch_size = input.shape[0]

        exp_prediction = np.zeros(input.shape)
        grad = np.zeros(input.shape)
        for i in range(batch_size):
            input[i, :] -= np.max(input[i, :])
            exp_prediction[i] = np.exp(input[i])
            grad[i] = exp_prediction[i] / np.sum(exp_prediction[i])

        loss = 0
        for i in range(batch_size):
            loss += np.log(np.sum(np.exp(input[i]))) - input[i, target[i]]

        for i in range(batch_size):
            grad[i, target[i]] -= 1
        return loss, grad

    def backward(self):
        raise NotImplementedError

