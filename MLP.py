from typing import List, Callable
import numpy as np


class MLP:
    """
       Multi-Layer Perceptron (MLP)
    """

    def __init__(self, hidden_layer_sizes: List[int],
                 activation_function: Callable[[float], float],
                 batch_size: int, epochs: int,
                 learning_rate: float, momentum: float,
                 bias: bool, rng: int, classification: bool):
        """
        :param hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
        :param activation_function: Function used for activating neurons
        :param batch_size: Size of minibatches
        :param epochs: Number of iterations
        :param learning_rate: Initial learning rate
        :param momentum: Momentum for gradient descend
        :param bias: Constant value of bias node
        :param rng: Seed for random number generator used for initial weights
        :param classification: Choice between classification and regression problem
        """
        pass
