import numpy as np
from functools import reduce
from triplen.autograd.function import *


def _tensor_wrapper(data):
    if isinstance(data, float) or isinstance(data, int):
        return Tensor([data])
    elif isinstance(data, np.ndarray):
        return Tensor(data)
    elif isinstance(data, Tensor):
        return data
    else:
        raise TypeError('float, int, np.ndarray, tensor')


class Tensor(object):
    def __init__(self, data, dtype: str = None, requires_grad: bool = False):
        if isinstance(data, Tensor):
            self.data = data.data
        elif isinstance(data, list) or isinstance(data, tuple):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise TypeError('list, tuple, np.ndarray, Tensor')

        if requires_grad and dtype not in [np.float, np.float16, np.float32, np.float64]:
            raise RuntimeError('Only Tensor of floating point dtype can require gradients')

        if dtype is not None:
            self.data.astype(dtype)
        else:
            self.data.astype(np.float)

        self._requires_grad = requires_grad
        self._grad = None
        self.grad_fn = None

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.getter
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        if requires_grad and self.dtype not in [np.float, np.float16, np.float32, np.float64]:
            raise RuntimeError('only Tensors of floating point dtype can require gradients')
        self._requires_grad = requires_grad

    def backward(self, gradient=None):
        if not self._requires_grad:
            return
        triplen.autograd.backward(self, gradient)

    def get_grad_accumulator(self):
        return triplen.autograd.AccumulateGrad(self)

    @property
    def grad(self):
        return self._grad

    @grad.getter
    def grad(self):
        if self._grad is None:
            self._grad = np.zeros(self.shape)
        return self._grad

    @grad.setter
    def grad(self, grad):
        self._grad = grad

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def item(self):
        return self.data.item()

    def is_leaf(self):
        return self.grad_fn is None

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

    def numpy(self):
        return self.data

    def uniform_(self, a=0.0, b=1.0):
        self.data = np.random.uniform(a, b, self.shape)

    def normal_(self, mean=0.0, std=1.0):
        self.data = np.random.normal(mean, std, self.shape)

    def add(self, other):
        return Add.apply(self, _tensor_wrapper(other))

    def sub(self, other):
        return Sub.apply(self, _tensor_wrapper(other))

    def mul(self, other):
        return Mul.apply(self, _tensor_wrapper(other))

    def div(self, other):
        return Div.apply(self, _tensor_wrapper(other))

    def pow(self, other):
        return Pow.apply(self, _tensor_wrapper(other))

    def view(self, *args):
        return View.apply(self, args)

    def __neg__(self):
        return self.mul(-1)

    def __pos__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return _tensor_wrapper(other).sub(self)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        return _tensor_wrapper(other).div(self)

    def __pow__(self, power, modulo=None):
        if modulo is not None:
            raise ValueError
        return self.pow(power)

    def __rpow__(self, other):
        return _tensor_wrapper(other).pow(self)

    def __getitem__(self, *args):
        if len(args) > self.ndim():
            raise IndexError("too many indices for array")
        index_num = -1
        for arg in args:
            if isinstance(arg, list):
                if index_num >= 0 and index_num != len(arg):
                    raise IndexError("index mismatch")
                index_num = len(arg)
            elif isinstance(arg, slice) or isinstance(arg, int):
                pass
            else:
                raise ValueError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`)"
                                 " and integer or boolean arrays are valid indices")
        if index_num >= 0:
            return Index.apply(index_num, *args)
        else:
            return Select.apply(*args)

