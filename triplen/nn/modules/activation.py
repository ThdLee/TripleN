# import numpy as np
# from triplen.nn.modules.module import Module
# from triplen import Tensor
#
#
# class Relu(Module):
#     def __init__(self):
#         super(Relu, self).__init__()
#         self.buffer = Tensor(np.zeros(1))
#
#     def forward(self, x):
#         self.buffer.data = x
#         return np.maximum(x, 0)
#
#     def backward(self, grad_output):
#         grad_output[self.buffer.data < 0] = 0
#         return grad_output
