import numpy as np
from nn.Variable import Variable, GLOBAL_VARIABLE_SCOPE
from nn.Operator import Operator


class Relu(Operator):
    def __init__(self, input_variable: Variable, name: str):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope=name)
        Operator.__init__(self, name, [self.input_variables], [self.output_variables])

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.output_variables.data = np.maximum(self.input_variables.data, 0)
            self.wait_forward = False

    def backward(self):
        if not self.wait_forward:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables.diff = self.output_variables.diff
            self.output_variables.diff[self.input_variables.data < 0] = 0
            self.wait_forward = True


class Sigmoid(Operator):
    def __init__(self, input_variable: Variable, name:str):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope=name)
        Operator.__init__(self, name, [self.input_variables], [self.output_variables])

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.output_variables.data = 1.0 / (1.0 + np.exp(-self.input_variables.data))
            self.wait_forward = False

    def backward(self):
        if not self.wait_forward:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables.diff = self.output_variables.data * (1 - self.output_variables.data) * \
                                        self.output_variables.diff
            self.wait_forward = True


class Tanh(Operator):
    def __init__(self, input_variable: Variable, name: str):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope='name')
        Operator.__init__(self, name, [self.input_variables], [self.output_variables])

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.output_variables.data = 1 - 2.0 / (np.exp(self.input_variables.data * 2))
            self.wait_forward = False

    def backward(self):
        if not self.wait_forward:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables.diff = self.output_variables.diff * (1 - self.output_variables.data ** 2)
            self.wait_forward = True
