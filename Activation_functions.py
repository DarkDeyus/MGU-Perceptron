import math
import numpy as np
from typing import Callable


class ActivationFunction:
    """Class representing activation function used in neural networks"""
    def __init__(self, activation_function: Callable[[np.array], np.array],
                 gradient: Callable[[np.array], np.array]):
        self.function = activation_function
        self.gradient = gradient


class ErrorFunction:
    """Class representing error function used in neural networks"""
    def __init__(self, error_function: Callable[[np.array, np.array], np.array],
                 gradient: Callable[[np.array, np.array], np.array]):
        self.function = error_function
        self.gradient = gradient


def mean_squared_error(result: np.array, expected: np.array) -> np.array:
    res = np.power(expected - result, 2)
    return res


def mean_squared_error_derivative(result: np.array, expected: np.array) -> np.array:
    res = 2 * (result - expected)
    return res


mean_squared_error_function = ErrorFunction(mean_squared_error, mean_squared_error_derivative)


def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.array) -> np.array:
    return np.multiply(sigmoid(x), (1 - sigmoid(x)))


def tanh(x: np.array) -> np.array:
    return np.tanh(x)


def tanh_derivative(x: np.array) -> np.array:
    return 1 - np.tanh(x) ** 2


def reLU(x: np.array) -> np.array:
    return np.maximum(0, x)


def reLU_derivative(x: np.array) -> np.array:
    return np.greater(x, 0).astype(float)


def identity(x: np.array) -> np.array:
    return x


def identity_derivative(x: np.array) -> np.array:
    return np.ones_like(x)


def leakyReLU(x: np.array) -> np.array:
    return np.where(x > 0, x, x * 0.01)


def leakyReLU_derivative(x: np.array) -> np.array:
    dx = np.ones_like(x)
    dx[x < 0] = 0.01
    return dx

sigmoid_activation_function = ActivationFunction(sigmoid, sigmoid_derivative)

