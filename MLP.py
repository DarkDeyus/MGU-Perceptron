from typing import List
import Neural_network as nn
import Activation_functions as af
import Visualizer as v
import numpy as np
import pandas as pd

def print_iter(net: nn.NeuralNetwork, avg_error: float, i: int) -> bool:
    if i % 25 == 0:
        print("--------------------------------")
        print(f"Iter {i}: error {avg_error}")
#        for (i, l) in enumerate(net.layers):
#            print(f"Layer {i}")
#            print(l.weights)
#            print("\n")
#        print("\n")
    return True


class MLP:
    """
      Multi-Layer Perceptron (MLP)
    """

    def __init__(self, hidden_layer_sizes: List[int],
                 activation_function: af.ActivationFunction,
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
        self.net = nn.NeuralNetwork(af.mean_squared_error_function, hidden_layer_sizes)

    def fit_df(self, train_data: pd.DataFrame, x_columns: List[str], y_columns: List[str]) -> None:
        self.fit_params = nn.FitParams(self.earning_rate, self.batch_size, self.epochs,
            self.classification, self.momentum, x_columns, y_columns, self.bias,
            self.activation_function, print_iter)
        self.net.fit_df(train_data, self.fit_params)

    def fit(self, train_data_X: pd.DataFrame, train_data_Y: pd.DataFrame) -> None:
        self.fit_params = nn.FitParams(self.earning_rate, self.batch_size, self.epochs,
            self.classification, self.momentum, None, None, self.bias,
            self.activation_function, print_iter)
        self.net.fit(train_data_X, train_data_Y, self.fit_params)

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        return self.net.predict(test_data)

def main():
    np.random.seed(5)
    s = 10
    m = np.random.randint(0, 2, size=(s, 5))
    m2 = np.hstack((m, m[:, [0]]))
    #m = np.outer(np.array(list(range(s))), np.array([1/s, 1/s, 1/s, 1/s, 1/s]))
    #m2 = np.hstack((m, af.sigmoid_vec(m[:, [0]])))
    df = pd.DataFrame(m2, columns=list("abcdef"))
    batch_size = s//2
    epochs = 100
    learning_rate = 0.001
    momentum = 0.001
    bias = False
    rng = 1337
    classification = True
    hidden_layers_sizes = [2, 2]
    x_columns = list("abcde")
    y_columns = ["f"]
    activation_function = af.sigmoid_activation_function
    mlp = MLP(hidden_layers_sizes, activation_function, batch_size, epochs,
              learning_rate, momentum, bias, rng, classification)
    mlp.fit_df(df, x_columns, y_columns)
    res = mlp.predict(df)
    print(df)
    print(res)

    v.visualize_classification(mlp, df, res)
    v.show_edges_weight(mlp)


if __name__ == "__main__":
    main()
