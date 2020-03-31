from __future__ import annotations
import numpy as np
import pandas as pd
import Activation_functions as af
from typing import Tuple, List, Callable

class NeuralNetworkLayer:
    """Class representing single layer of neural network"""
    def __init__(self, weight_matrix: np.matrix,
                 activation_function: af.ActivationFunction,
                 has_bias: bool):
        self.weights = weight_matrix
        self.activation = activation_function
        self.bias = has_bias

    @classmethod
    def from_matrix(cls, matrix: np.matrix, add_bias: bool, has_bias: bool,
                    activation_function: af.ActivationFunction) -> NeuralNetworkLayer:
        if add_bias:
            matrix = np.hstack((matrix, np.random(matrix.shape[0])))
        return NeuralNetworkLayer(matrix, activation_function, has_bias)

    @classmethod
    def create_random(cls, input_n: int, output_n: int, add_bias: bool, has_bias: bool,
                      activation_function: af.ActivationFunction) -> NeuralNetworkLayer:
        if add_bias:
            input_n += 1
        return NeuralNetworkLayer(np.random.rand(output_n, input_n), activation_function, has_bias)

    def process_forward(self, input: np.array, add_bias: bool) -> Tuple[np.array, np.array]:
        if add_bias:
            input = np.append(input, 1.0)
        product = np.matmul(self.weights, input)
        res = self.activation.function(product)
        return (product, res)

    def process_backwards(self, product: np.array, next_delta: np.array, next_weights: np.matrix) -> np.array:
        gradient = self.activation.gradient(product)
        delta = np.multiply(gradient, np.matmul(next_weights.T, next_delta))
        return delta

    def process_backwards_last(self, product: np.array,
                               errors_grad: np.array) -> np.array:
        gradient = self.activation.gradient(product)
        delta = np.multiply(errors_grad, gradient)
        return delta

    def calc_change(self, input: np.array, deltas: np.array) -> np.matrix:
        change = np.outer(deltas, input)
        return change

    def update(self, change: np.matrix, prev_change: np.matrix,
               learning_rate: float, momentum: float) -> np.matrix:
        if prev_change is None:
            prev_change = np.zeros_like(self.weights)
        total_change = momentum * prev_change + learning_rate * change
        self.weights -= total_change
        return -total_change


class FitParams:
    """Class configuring fitting options"""
    def __init__(self, learning_rate: float, batch_size: int, epochs: int, classification: bool,
                 momentum: float, x_column_names: List[str], y_column_names: List[str], bias: bool,
                 activation_function: af.ActivationFunction,
                 iter_callback: Callable[[NeuralNetwork, float, int], bool] = None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.momentum = momentum
        self.x_column_names = x_column_names
        self.y_column_names = y_column_names
        self.classification = classification
        self.iter_callback = iter_callback
        self.bias = bias
        self.activation_function = activation_function


class NeuralNetwork:
    """Class of model of neural network"""
    def __init__(self,
                 error_function: af.ErrorFunction,
                 hidden_layers_sizes: List[int]):
        self.error = error_function
        self.hidden_layers_sizes = hidden_layers_sizes
        self.model_created = False
        self.fit_params = None

    def _choose_train_data(self, X: pd.DataFrame, Y: pd.DataFrame,
                           fit_params: FitParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
        choice = np.random.choice(np.arange(X.shape[0]),
                                  fit_params.batch_size)
        return (pd.DataFrame(X.iloc[choice, :]), pd.DataFrame(Y.iloc[choice, :]))

    def _prepare_layers(self, hidden_layers_sizes: List[int], input_size: int,
                        output_size: int, fit_params: FitParams) -> None:
        self.layers = []
        sizes = [input_size] + hidden_layers_sizes + [output_size]
        for i in range(len(hidden_layers_sizes)):
            self.layers.append(NeuralNetworkLayer.create_random(
              sizes[i], sizes[i+1], fit_params.bias, fit_params.bias, fit_params.activation_function))
        self.layers.append(NeuralNetworkLayer.create_random(
          sizes[-2], sizes[-1], fit_params.bias, fit_params.bias, fit_params.activation_function))

    def _iterate_fit(self, X: pd.DataFrame, Y: pd.DataFrame,
                     fit_params: FitParams) -> Tuple[List[np.matrix, float]]:
        inputs_all = []
        products_all = []
        deltas = []
        total_error = 0.0
        for layer in self.layers:
            deltas.append(np.zeros_like(layer.weights))
        for i in range(len(X)):
            expected_output = np.array(Y.iloc[i, :].values[0])
            first_input = np.array(X.iloc[i, :])
            if self.layers[0].bias:
                first_input = np.append(first_input, 1.0)
            inputs = [first_input]
            products = []
            for j in range(len(self.layers)):
                layer = self.layers[j]
                (product, output) = layer.process_forward(inputs[-1], False)
                products.append(product)
                if j + 1 < len(self.layers) and self.layers[j+1].bias:
                    next_input = np.append(output, 1.0)
                else:
                    next_input = output
                inputs.append(next_input)
            products_all.append(products)
            inputs_all.append(inputs)
            error = self.error.function(inputs[-1], expected_output)
            total_error += error
            error_grad = self.error.gradient(inputs[-1], expected_output)
            for i in reversed(range(len(self.layers))):
                layer = self.layers[i]
                product = products[i]
                input = inputs[i]
                if i == len(self.layers) - 1:
                    delta = layer.process_backwards_last(product, error_grad)
                else:
                    next_layer = self.layers[i+1]
                    no_bias_weights = next_layer.weights[:, :-1] if next_layer.bias else next_layer.weights
                    delta = layer.process_backwards(product, delta, no_bias_weights)
                deltas[i] += layer.calc_change(input, delta)
        avg_error = total_error/len(X)
        for delta in deltas:
            delta /= len(X)
        return (deltas, avg_error)

    def fit(self, Xdf: pd.DataFrame, Ydf: pd.DataFrame, fit_params: FitParams) -> None:
        fit_params.x_column_names = list(Xdf.columns)
        fit_params.y_column_names = list(Ydf.columns)
        if self.model_created:
            raise RuntimeError("Model was already trained")
        X = Xdf
        Y_org = Ydf
        input_size = len(fit_params.x_column_names)
        if fit_params.classification:
            self.Y = Y_org.drop_duplicates()
            self.clf_dict = {}
            self.clf_reverse_dict = {}
            output_size = len(self.Y)
            for i in range(len(self.Y)):
                self.clf_dict[i] = tuple(self.Y.iloc[[i], :].values[0])
                self.clf_reverse_dict[self.clf_dict[i]] = i
            Y_org = \
              pd.DataFrame(Y_org.apply(
                lambda x:
                  np.eye(output_size)[:, self.clf_reverse_dict[tuple(x)]],
                                 axis=1))
        else:
            self.Y = Y_org
            output_size = len(self.Y.columns)
            Y_org = pd.DataFrame(Y_org.apply(
                lambda x: 
                    np.array(x),
                axis = 1
            ))
        self._prepare_layers(self.hidden_layers_sizes, input_size, output_size, fit_params)
        prev_changes = [None] * len(self.layers)
        self.model_created = True
        for k in range(fit_params.epochs):
            (X_it, Y_it) = self._choose_train_data(X, Y_org, fit_params)
            (deltas, avg_error) = self._iterate_fit(X_it, Y_it, fit_params)
            if fit_params.iter_callback is not None:
                res = fit_params.iter_callback(self, avg_error, k)
                if not res:
                    break
            for i in range(len(self.layers)):
                prev_changes[i] = self.layers[i].update(deltas[i], prev_changes[i],
                                                        fit_params.learning_rate, fit_params.momentum)
        self.fit_params = fit_params

    def fit_df(self, df: pd.DataFrame, fit_params: FitParams) -> None:
        self.fit(pd.DataFrame(df[fit_params.x_column_names]),
                 pd.DataFrame(df.loc[:, fit_params.y_column_names], 
                    columns=fit_params.y_column_names), 
                 fit_params)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model_created:
            raise RuntimeError("Model was not trained")
        results = []
        for i in range(len(df)):
            result = self._predict_single_raw(np.array(df.loc[[i], self.fit_params.x_column_names].values[0]))
            results.append(result)
        if self.fit_params.classification:
            cls_no = np.argmax(results, axis=1)
            return pd.DataFrame(self.Y.iloc[cls_no, :])
        else:
            return pd.DataFrame(results, columns=self.Y.columns)

    def predict_single(self, single_X: np.array) -> np.array:
        result = self._predict_single_raw(single_X)
        if self.fit_params.classification:
            cls_no = np.argmax(result)
            return np.array(self.Y[cls_no, :])
        else:
            return result

    def _predict_single_raw(self, single_X: np.array) -> np.array:
        if not self.model_created:
            raise RuntimeError("Model was not trained")
        input = single_X
        for l in self.layers:
            (_, input) = l.process_forward(input, self.fit_params.bias)
        return input
