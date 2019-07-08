import math
from functools import reduce
from triplen import Tensor


def uniform_(tensor, a=0.0, b=1.0):
    return tensor.uniform_(a, b)


def normal_(tensor, mean=0.0, std=1.0):
    return tensor.normal_(mean, std)


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return tensor.uniform_(-bound, bound)


def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv2d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _calculate_fan_in_and_fan_out(tensor: Tensor):
    dimensions = tensor.ndim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(-1)
        fan_out = tensor.size(-2)
    else:
        num_input_fmaps = tensor.size(-1)
        num_output_fmaps = tensor.size(-2)
        receptive_field_size = 1
        if tensor.ndim() > 2:
            receptive_field_size = reduce(lambda x, y: x * y, tensor.shape[:-2])
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out