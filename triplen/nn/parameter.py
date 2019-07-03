from triplen import Tensor


class Parameter(Tensor):
    def __init__(self, data, dtype: str = None, requires_grad: bool = True):
        super(Parameter, self).__init__(data, dtype, requires_grad)

    def grad_fn(self):
        pass
