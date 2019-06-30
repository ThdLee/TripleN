from triplen.optim.optimizer import Optimizer
import numpy as np
import math


class Adam(Optimizer):
    def __init__(self, params, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(beta2))

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        super(Adam, self).__init__(params)

    def step(self, batch_size):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p.grad = p.grad / batch_size
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = np.zeros(p.data.shape)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = np.zeros(p.data.shape)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1

                if self.weight_decay != 0:
                    p.grad += self.weight_decay * p.data

                # Decay the first and second moment running average coefficient
                exp_avg = self.beta1 * exp_avg + (1 - self.beta1) * p.grad
                exp_avg_sq = self.beta2 * exp_avg_sq + (1 - self.beta2) * p.grad * p.grad
                denom = np.sqrt(exp_avg_sq) + self.eps

                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq

                bias_correction1 = 1 - self.beta1 ** state['step']
                bias_correction2 = 1 - self.beta2 ** state['step']
                step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1

                p.data -= step_size * exp_avg / denom
