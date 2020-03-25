import math
import numpy as np
from typing import Callable


class ActivationFunction:
    """Class represeting activation function used in neural networks"""
    def __init__(self, activation_function: Callable[[np.array], np.array],
                 gradient: Callable[[np.array], np.array]):
        self.function = activation_function
        self.gradient = gradient

class ErrorFunction:
    """Class represeting error function used in neural networks"""
    def __init__(self, error_function: Callable[[np.array, np.array], np.array],
                 gradient: Callable[[np.array, np.array], np.array]):
        self.function = error_function
        self.gradient = gradient

def mean_squared_error(result: np.array, expected: np.array) -> np.array:
    res = np.power(expected - result, 2)
    return res

def mean_squared_error_derivative(result: np.array, expected: np.array) -> np.array:
    res = 2*(result - expected)
    return res


mean_squared_error_function = ErrorFunction(mean_squared_error, mean_squared_error_derivative)

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x: float) -> float:
    return sigmoid(x) * sigmoid(1-x)

def sigmoid_vec(x: np.array) -> float:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative_vec(x: np.array) -> np.array:
    return np.multiply(sigmoid_vec(x), (1-sigmoid_vec(x)))


sigmoid_activation_function = ActivationFunction(sigmoid_vec, sigmoid_derivative_vec)

def tanh(x: float) -> float:
    return 2 * sigmoid(2 * x) - 1


def reLU(x: float) -> float:
    return max(x, 0)


def identity(x: float) -> float:
    return x


def leakyReLU(x: float) -> float:
    return 0.01 * x if x < 0 else x

