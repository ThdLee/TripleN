import numpy as np
from .tensor import Tensor
import triplen.nn
import triplen.optim
import triplen.autograd
from functools import reduce

__all__ = ['tensor', 'Tensor']


def tensor(*args):
    return Tensor(np.random.randn(*args))

def numel(shape):
    return reduce(lambda x, y: x * y, shape)
