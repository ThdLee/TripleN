import numpy as np
from functools import reduce


# def tensor(*args):
#     return Tensor(np.random.randn(*args))


class AccumulateGrad(object):
    def __init__(self, tensor):
        self.tensor = tensor

    def apply(self, grad):
        self.tensor.grad += grad


def _variable_check(data):
    if isinstance(data, float) or isinstance(data, int) or isinstance(data, Tensor):
        return
    else:
        raise TypeError('float, int, tensor')


class Tensor(object):
    def __init__(self, data, dtype: str = np.float, requires_grad: bool = False):
        if isinstance(data, Tensor):
            self.data = data.data
        elif isinstance(data, list) or isinstance(data, tuple):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise TypeError('list, tuple, np.ndarray, Tensor')

        if dtype is not None:
            self.data.astype(dtype)

        self.requires_grad = requires_grad
        self._grad = None
        self.grad_fn = None

    def backward(self, gradient=None):
        if not self.requires_grad:
            return
        backward_stack = [(self.grad_fn, gradient)]
        while len(backward_stack) != 0:
            grad_fn, gradient = backward_stack.pop()
            if grad_fn is None:
                continue
            outputs = grad_fn.apply(gradient)
            if isinstance(outputs, tuple):
                assert len(outputs) == len(grad_fn.next_functions)
                for func, grad in zip(self.grad_fn.next_functions, outputs):
                    backward_stack.append((func, grad))
            else:
                assert len(grad_fn.next_functions) == 1
                backward_stack.append((grad_fn.next_functions[0], outputs))

    def get_grad_accumulator(self):
        return AccumulateGrad(self)

    @property
    def grad(self):
        return self._grad

    @grad.getter
    def grad(self):
        if self._grad is None:
            self._grad = np.zeros(self.shape)
        return self._grad

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

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

    def uniform_(self, a=0.0, b=1.0):
        self.data = np.random.uniform(a, b, self.shape)

    def normal_(self, mean=0.0, std=1.0):
        self.data = np.random.normal(mean, std, self.shape)

    def add(self, other):
        from .nn.functional import add
        _variable_check(other)
        return add(self, other)

    def mul(self, other):
        from .nn.functional import mul
        _variable_check(other)
        return mul(self, other)

    def div(self, other):
        from .nn.functional import div
        _variable_check(other)
        return div(self, other)

    def pow(self, other):
        from .nn.functional import pow
        _variable_check(other)
        return pow(self, other)

    def view(self, *args):
        from .nn.functional import view
        return view(self, args)

    def __add__(self, other):
        _variable_check(other)
        if isinstance(other, Tensor):
            return self.data + other.data
        else:
            return self.data + other


def numel(shape):
    return reduce(lambda x, y: x * y, shape)

