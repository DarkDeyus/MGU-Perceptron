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

    def process_backwards(self, product: np.array, errors: np.array) -> Tuple[np.array, np.array]:
        prev_errors = np.dot(self.weights.T, errors)
        if self.bias:
            prev_errors = prev_errors[:-1]
        gradient = self.activation.gradient(product)
        delta = np.multiply(errors, gradient)
        return (prev_errors, delta)

    def process_backwards_last(self, product: np.array,
                               errors_grad: np.array) -> Tuple[np.array, np.array]:
        prev_errors = np.dot(self.weights.T, errors_grad)
        if self.bias:
            prev_errors = prev_errors[:-1]
        gradient = self.activation.gradient(product)
        delta = np.multiply(errors_grad, gradient)
        return (prev_errors, delta)

    def calc_change(self, input: np.array, deltas: np.array) -> np.matrix:
        change = np.outer(deltas, input)
        return change

    def update(self, change: np.matrix, prev_change: np.matrix,
               learning_rate: float, momentum: float) -> np.matrix:
        if prev_change is None:
            prev_change = np.zeros_like(self.weights)
        total_change = momentum * prev_change + learning_rate * change
        self.weights -= total_change
        return total_change


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
        #errors_all = []
        #error_grads_all = []
        deltas = []
        total_error = 0.0
        for layer in self.layers:
            deltas.append(np.zeros_like(layer.weights))
        for i in range(len(X)):
            expected_output = np.array(Y.iloc[i, :].values[0])
            if self.layers[0].bias:
                first_input = np.append(np.array(X.iloc[i, :]), 1.0)
            else:
                first_input = np.append(np.array(X.iloc[i, :]))
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
            sum_error = sum(error)
            total_error += sum_error
            error_grad = self.error.gradient(inputs[-1], expected_output)
            #errors_all.append(error)
            #error_grads_all.append(error_grads_all)
            error_it = error_grad
            for i in reversed(range(len(self.layers))):
                layer = self.layers[i]
                product = products[i]
                input = inputs[i]
                if i == len(self.layers) - 1:
                    (error_it, delta) = layer.process_backwards_last(product, error_grad)
                else:
                    (error_it, delta) = layer.process_backwards(product, error_it)
                deltas[i] += layer.calc_change(input, delta)
        avg_error = (total_error/self.layers[-1].weights.shape[0])/len(X)
        return (deltas, avg_error)

    def fit(self, df: pd.DataFrame, fit_params: FitParams):
        if self.model_created:
            raise RuntimeError("Model was already trained")
        X = df[fit_params.x_column_names]
        Y_org = pd.DataFrame(df.loc[:, fit_params.y_column_names], columns=fit_params.y_column_names)
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
        self._prepare_layers(self.hidden_layers_sizes, input_size, output_size, fit_params)
        prev_changes = [None] * len(self.layers)
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
        self.model_created = True
        self.fit_params = fit_params

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model_created:
            raise RuntimeError("Model was not trained")
        results = []
        for i in range(len(df)):
            result = self._predict_single_raw(np.array(df.loc[[i], self.fit_params.x_column_names].values[0]))
            results.append(result)
        if self.fit_params.classification:
            cls_no = np.argmax(results, axis=1)
            return pd.DataFrame(self.Y.iloc[cls_no, :]).reset_index()
        else:
            return pd.DataFrame(results, columns=self.Y.columns).reset_index()

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
            (_, input) = l.process_forward(input, True)
        return input
