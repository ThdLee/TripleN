from optim.optimizer import Optimizer
import numpy as np


class SGD(Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0, weight_decay: float = 0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.mtmp = None

        super(SGD, self).__init__(params)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad.data += self.weight_decay * p.data
                if self.momentum != 0:
                    if self.mtmp is None:
                        self.mtmp = np.zeros(p.grad.shape)
                    p.grad = self.mtmp * self.momentum + p.grad
                p.data -= self.lr * p.grad
