import math


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(x * -1))


def tanh(x: float) -> float:
    return 2 * sigmoid(2 * x) - 1


def reLU(x: float) -> float:
    return max(x, 0)


def identity(x: float) -> float:
    return x


def leakyReLU(x: float) -> float:
    return 0.01 * x if x < 0 else x

