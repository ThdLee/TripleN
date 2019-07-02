from triplen.autograd.function import *


def add(x, y):
    output = Add.apply(x, y)
    return output


def subtract(x, y):
    output = Subtract.apply(x, y)
    return output


def multiply(x, y):
    output = Multiply.apply(x, y)
    return output


def divide(x, y):
    output = Divide.apply(x, y)
    return output


def pow(x, y):
    output = Pow.apply(x, y)
    return output


def view(x, shape):
    output = View.apply(x, shape)
    return output


def maxpooling(x, kernel_size, stride):
    return MaxPooling.apply(x, kernel_size, stride)


def conv2d(x, weight, bias, stride, padding):
    return Conv2D.apply(x, weight, bias, stride, padding)


def relu(x):
    return Relu.apply(x)


def linear(x, weight, bias):
    return Linear.apply(x, weight, bias)
