import numpy as np
from nn.module import Module


class CrossEntropyLoss(Module):
    def __init__(self, shape):
        super(CrossEntropyLoss, self).__init__()
        self.softmax = np.zeros(shape)
        self.eta = np.zeros(shape)
        self.batch_size = shape[0]

    def cal_loss(self, input, target):
        self.input = input
        self.target = target
        self.cal_prob(input)
        self.loss = 0
        for i in range(self.batch_size):
            self.loss = np.log(np.sum(np.exp(input[i]))) - input[i, target[i]]
        return self.loss

    def cal_prob(self, input):
        exp_prediction = np.zeros(input.shape)
        self.softmax = np.zeros(input.shape)
        for i in range(self.batch_size):
            input[i, :] -= np.max(input[i, :])
            exp_prediction[i] = np.exp(input[i])
            self.softmax[i] = exp_prediction[i] / np.sum(exp_prediction[i])
        return self.softmax

    def gradient(self):
        self.eta = self.softmax.copy()
        for i in range(self.batch_size):
            self.eta[i, self.target[i]] -= 1
        return self.eta