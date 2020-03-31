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
        self.net = nn.NeuralNetwork(af.mean_squared_error_function, hidden_layer_sizes)

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
        print(f"Epoch {epoch}, iter {iter}: error {avg_error}")
        #Y_train_predicted = perceptron.net.predict(X_train)
        #Y_test_predicted = perceptron.net.predict(X_test)
        #print(Y_train_predicted)
        #print(Y_test_predicted)
        #v.show_edges_weight(mlp)
        pass
#        for (i, l) in enumerate(net.layers):
#            print(f"Layer {i}")
#            print(l.weights)
#            print("\n")
#        print("\n")
    return True

def main():
    np.random.seed(5)
    
    s = 10
    m = np.random.randint(0, 2, size=(s, 5))

    lm = np.array([np.linspace(0, 1, s)]).T
    lmy = np.power(lm, 1)
    Xdf = pd.DataFrame(lm, columns=["X"])
    Ydf = pd.DataFrame(lmy < 0.5, columns=["Y"])
    lm_test = np.array([np.linspace(0, 1, 5*s)]).T
    lmy_test = np.power(lm_test, 1)
    Xt_df = pd.DataFrame(lm_test, columns=["X"])
    Yt_df = pd.DataFrame(lmy_test < 0.5, columns=["Y"])


    m = np.hstack((lm, lm, lm, lm, lm))
    #m = np.array([np.array([0,0,0,0,0]), np.array([1,1,1,1,1])])
    #m2 = np.hstack((m, np.array([np.array([0, 0, 0, 0, 0, 1, 1, 1, 1 ,1, 1])]).T ))
    m2 = np.hstack((m, np.array([np.array([0]*(s - 10) + [1]*10)]).T))
    #m = np.outer(np.array(list(range(s))), np.array([1/s, 1/s, 1/s, 1/s, 1/s]))
    #m2 = np.hstack((m, af.sigmoid_vec(m[:, [0]])))
    df = pd.DataFrame(m2, columns=list("abcdef"))
    batch_size = s
    epochs = 5000
    learning_rate = 0.1
    momentum = 0
    bias = False
    rng = 12369666
    classification = True
    hidden_layers_sizes = [5]
    x_columns = list("a")
    y_columns = ["f"]
    X = pd.DataFrame(df.loc[:, x_columns])
    Y = pd.DataFrame(df.loc[:, y_columns])
    activation_function = af.sigmoid_activation_function

    def it_cb(perceptron, avg_error, epoch, iter):
        return print_iter(perceptron, avg_error, epoch, iter, X, Y, X, Y)

    global mlp
    mlp = MLP(hidden_layers_sizes, activation_function, batch_size, epochs,
              learning_rate, momentum, bias, rng, classification, it_cb)
    #mlp.fit_df(df, x_columns, y_columns)
    mlp.fit(Xdf, Ydf)
    Yp_df = mlp.predict(Xt_df)
    v.visualize_regression(Xdf, Ydf, Xt_df, Yt_df, Yp_df)
    #res = mlp.predict(df)
    #mlp.predict(pd.DataFrame(df.iloc[[s - 2, s - 1], ]).reset_index())
    #print(df)
    #print(res)
    v.show_edges_weight(mlp)


if __name__ == "__main__":
    main()