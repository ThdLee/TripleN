import numpy as np
import math
from functools import reduce


def initializer(shape, method):
    if method == 'const':
        return np.random.standard_normal(shape) / 100
    elif method == 'None':
        return np.zeros(shape)
    elif method == 'MSRA':
        weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / shape[-1])
        return np.random.standard_normal(shape) / weights_scale
