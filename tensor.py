import numpy as np


class Tensor(object):
    def __init__(self, data, dtype=None, requires_grad=False):
        self.data = data

        if dtype is not None:
            self.data.astype(dtype)

        self.requires_grad = requires_grad
        if self.requires_grad:
            self._grad = np.zeros(data.shape, data.dtype)

        self.shape = self.data.shape

    def size(self):
        return self.shape


class Parameter(Tensor):
    def __init__(self, data, dtype=None, requires_grad=True):
        super(Parameter, self).__init__(data, dtype, requires_grad)

    def grad_fn(self):
        pass
