from triplen.autograd.function import *


def add(x, y):
    output = Add.apply(x, y)
    return output


def mul(x, y):
    output = Mul.apply(x, y)
    return output


def div(x, y):
    output = Div.apply(x, y)
    return output


def pow(x, y):
    output = Pow.apply(x, y)
    return output


def view(x, shape):
    output = View.apply(x, shape)
    return output


def max_pool2d(x, kernel_size, stride=None):
    return MaxPooling.apply(x, kernel_size, stride or kernel_size)


def conv2d(x, weight, bias, stride, padding):
    return Conv2D.apply(x, weight, bias, stride, padding)


def dropout(x, prob, training):
    return Dropout.apply(x, prob, training)


def softmax(x, dim):
    return Softmax.apply(x, dim)


def log_softmax(x, dim):
    return LogSoftmax.apply(x, dim)


def relu(x):
    return Relu.apply(x)


def linear(x, weight, bias):
    return Linear.apply(x, weight, bias)


def cross_entropy_loss(input, target):
    # return NLLLoss.apply(LogSoftmax.apply(input, -1), target)
    return CrossEntropyLoss.apply(input, target)
