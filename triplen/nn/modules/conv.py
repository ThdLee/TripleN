import math
import triplen
from triplen.nn.parameter import Parameter
from triplen.nn.modules.module import Module
from .. import init
from .. import functional as F


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = Parameter(triplen.tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = Parameter(triplen.tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)

