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


def mean_squared_error(result: np.array, expected: np.array) -> float:
    sub = np.subtract(result, expected)
    squared = np.square(sub)
    res = squared.mean()
    return res


def mean_squared_error_derivative(result: np.array, expected: np.array) -> np.array:
    res = 2*(result - expected)/result.shape[0]
    return res

def softmax(x: np.array) -> np.array:
    ex = np.exp(x)
    ex /= ex.sum(0)
    return ex

def sotfmax_cross_entropy(result: np.array, expected: np.array) -> np.array:
    softmaxed = softmax(result)
    logs = np.log(softmaxed)
    result = -np.multiply(logs, expected).sum(0).mean()
    return result

def softmax_cross_entropy_derivative(result: np.array, expected: np.array) -> np.array:
    softmaxed = softmax(result)
    return softmaxed - expected 

#def mean_squared_softmax_error(result: np.array, expected: np.array) -> float:
#    return mean_squared_error(softmax(result), expected)

#def mean_squared_softmax_error_derivative(result: np.array, expected: np.array) -> float:
#    soft_result = softmax(result)
#    exps = np.exp(result)
#    sums = exps.sum(0)
#    n = result.shape[0]
#    x = (2/n)*np.multiply(result, 1/np.power(sums, 2))
#    x = np.multiply(x, soft_result - expected)
#    diff = soft_result - expected
#    diffT = np.reshape(diff, (diff.shape[0], 1))    
#    np.matmul( ,diffT)



def mean_squared_softmax(result: np.array, expected: np.array) -> float:
    sub = np.subtract(result, expected)
    squared = np.square(sub)
    res = squared.mean()
    return res


def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))

def sigmoid_inverse(x: np.array) -> np.array:
    return -np.log(1/x - 1)

def sigmoid_derivative(x: np.array) -> np.array:
    y = sigmoid(x)
    return np.multiply(y, (1 - y))

def tanh(x: np.array) -> np.array:
    return np.tanh(x)


def tanh_derivative(x: np.array) -> np.array:
    return 1 - np.power(np.tanh(x), 2)


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
    return (x > 0).astype(float) * x + (x < 0).astype(float) * 0.01 * x


mean_squared_error_function = ErrorFunction(mean_squared_error, mean_squared_error_derivative)
softmax_cross_entropy_error_function = ErrorFunction(sotfmax_cross_entropy, softmax_cross_entropy_derivative)

sigmoid_activation_function = ActivationFunction(sigmoid, sigmoid_derivative)
tanh_activation_function = ActivationFunction(tanh, tanh_derivative)
ReLU_activation_function = ActivationFunction(reLU, reLU_derivative)
identity_activation_function = ActivationFunction(identity, identity_derivative)
leakyReLU_activation_function = ActivationFunction(leakyReLU, leakyReLU_derivative)

error_functions_dict = {
    "mean_squared": mean_squared_error_function,
    "softmax_cross_entropy": softmax_cross_entropy_error_function
}

activation_functions_dict = {
    "sigmoid": sigmoid_activation_function,
    "tanh": tanh_activation_function,
    "relu": ReLU_activation_function,
    "identity": identity_activation_function,
    "leakyrelu": leakyReLU_activation_function
}