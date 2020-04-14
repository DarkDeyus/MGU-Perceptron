from __future__ import annotations
from typing import List, Callable
import Neural_network as nn
import Activation_functions as af
import Visualizer as v
import numpy as np
import pandas as pd

class MLP:
    """
      Multi-Layer Perceptron (MLP)
    """

    def __init__(self, hidden_layer_sizes: List[int],
                 activation_function: af.ActivationFunction,
                 batch_size: int, epochs: int,
                 learning_rate: float, momentum: float,
                 bias: bool, rng: int, classification: bool,
                 iter_callback: Callable[[MLP, float, int, int], bool]):
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
        np.random.seed(rng)
        self.earning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.classification = classification
        self.momentum = momentum
        self.bias = bias
        self.activation_function = activation_function
        self.fit_params = None
        self.sizes = hidden_layer_sizes
        self.iter_callback = lambda net, avg_error, epoch, iter: iter_callback(self, avg_error, epoch, iter)
        self.net = nn.NeuralNetwork("softmax_cross_entropy", hidden_layer_sizes)

    def fit_df(self, train_data: pd.DataFrame, x_columns: List[str], y_columns: List[str]) -> None:
        self.fit_params = nn.FitParams(self.earning_rate, self.batch_size, self.epochs,
            self.classification, self.momentum, x_columns, y_columns, self.bias,
            self.activation_function, self.iter_callback)
        self.net.fit_df(train_data, self.fit_params)

    def fit(self, train_data_X: pd.DataFrame, train_data_Y: pd.DataFrame) -> None:
        self.fit_params = nn.FitParams(self.earning_rate, self.batch_size, self.epochs,
            self.classification, self.momentum, None, None, self.bias,
            self.activation_function, self.iter_callback)
        self.net.fit(train_data_X, train_data_Y, self.fit_params)

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        return self.net.predict(test_data)

def print_iter(perceptron: MLP, avg_error: float, epoch: int, iter: int,
               X_train: pd.DataFrame, Y_train: pd.DataFrame,
               X_test: pd.DataFrame, Y_test: pd.DataFrame) -> bool:
    if epoch % 25 == 0 and iter == 0:
        print("--------------------------------")
        print(f"Epoch {epoch}, iter {iter}: mean_squared_network_error {avg_error}")
        Y_train_predicted = perceptron.net.predict(X_train)
        Y_test_predicted = perceptron.net.predict(X_test)
        if perceptron.classification:
            acc_t = np.array(Y_train_predicted == Y_train).astype(int)
            acc = np.array(Y_test_predicted == Y_test).astype(int)
            acc_t = sum(acc_t)/len(acc_t)
            acc = sum(acc)/len(acc)
            print("Train acc", acc_t)
            print("Test acc", acc)
        else:
            sigma_t = np.sqrt((np.array(Y_train_predicted - Y_train)**2).mean())
            sigma = np.sqrt((np.array(Y_test_predicted - Y_test)**2).mean())
            print("Train standard deviation", sigma_t)
            print("Test standard deviation", sigma)
        #v.show_edges_weight(mlp)
        pass
#        for (i, l) in enumerate(net.layers):
#            print(f"Layer {i}")
#            print(l.weights)
#            print("\n")
#        print("\n")
    return True
