import math
import triplen
from .module import Module
from triplen.nn.parameter import Parameter
from .. import init
from .. import functional as F


class Linear(Module):
    def __init__(self, input_size: int, output_size: int):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weight = Parameter(triplen.tensor(input_size, output_size))
        self.bias = Parameter(triplen.tensor(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
