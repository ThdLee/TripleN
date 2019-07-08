import numpy as np
from .tensor import Tensor
import triplen.nn
import triplen.optim
import triplen.autograd

__all__ = ['tensor', 'Tensor']


def tensor(*size, dtype=None, requires_grad=False):
    return Tensor(np.random.randn(*size), dtype=dtype, requires_grad=requires_grad)


def ones(*size, dtype=None, requires_grad=False):
    return Tensor(np.ones(size, dtype=dtype), dtype=dtype, requires_grad=requires_grad)


def zeros(*size, dtype=None, requires_grad=False):
    return Tensor(np.zeros(size, dtype=dtype), dtype=dtype, requires_grad=requires_grad)


def randn(*size, dtype=None, requires_grad=False):
    return Tensor(np.random.rand(*size), dtype=dtype, requires_grad=requires_grad)


def randn(*size, dtype=None, requires_grad=False):
    return Tensor(np.random.randn(*size), dtype=dtype, requires_grad=requires_grad)


def arange(start, stop, step, dtype=None, requires_grad=False):
    return Tensor(np.arange(start, stop, step, dtype), dtype=dtype, requires_grad=requires_grad)

