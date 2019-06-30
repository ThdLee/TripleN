import numpy as np
from functools import reduce


def tensor(*args):
    return Tensor(np.random.randn(*args))


class Tensor(object):
    def __init__(self, data, dtype: str = None, requires_grad: bool = False):
        if isinstance(data, Tensor):
            self.data = data.data
        else:
            self.data = data

        if dtype is not None:
            self.dtype = dtype
            self.data.astype(dtype)
        else:
            self.dtype = data.dtype

        self._requires_grad = requires_grad
        if self.requires_grad:
            self.grad = np.zeros(data.shape, data.dtype)

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        if requires_grad:
            self.grad = np.zeros(self.shape, self.dtype)
        else:
            self.grad = None

    @property
    def shape(self):
        return self.data.shape

    def size(self, dim: int = None):
        assert dim < len(self.shape)
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def ndim(self):
        return self.data.ndim

    def numel(self):
        return reduce(lambda x, y: x * y, self.shape)

    def uniform_(self, a=0.0, b=1.0):
        self.data = np.random.uniform(a, b, self.shape)

    def normal_(self, mean=0.0, std=1.0):
        self.data = np.random.normal(mean, std, self.shape)


def numel(shape):
    return reduce(lambda x, y: x * y, shape)
