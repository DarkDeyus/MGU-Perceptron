from __future__ import annotations
import numpy as np
import pandas as pd
import Activation_functions as af
from typing import Tuple, List, Callable
import os
import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
jsonpickle_pd.register_handlers()

def to_json_file(obj: object, path: str) -> None:
    with open(path, 'w') as f:
        f.write(jsonpickle.dumps(obj))

def from_json_file(path: str) -> obj:
    with open(path, 'r') as f:
        return jsonpickle.loads(f.read())
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
        return NeuralNetworkLayer((np.random.rand(output_n, input_n)*2 - 1)*0.05, activation_function, has_bias)

    def process_forward(self, input: np.matrix, add_bias: bool) -> Tuple[np.matrix, np.matrix]:
        if add_bias:
            ones = np.ones((1, input.shape[1]))
            input = np.vstack((input, ones))
        product = np.matmul(self.weights, input)
        res = self.activation.function(product)
        return (product, res)

    def process_backwards(self, product: np.array, next_delta: np.array, next_weights: np.matrix) -> np.array:
        gradient = self.activation.gradient(product)
        output_grad = np.matmul(next_weights.T, next_delta)
        delta = np.multiply(gradient, output_grad)
        return delta

    def process_backwards_last(self, product: np.array,
                               errors_grad: np.array) -> np.array:
        gradient = self.activation.gradient(product)
        delta = np.multiply(errors_grad, gradient)
        return delta

    def calc_change(self, input: np.array, deltas: np.array) -> np.matrix:
        change = np.matmul(deltas.T[:,:,np.newaxis], input.T[:,np.newaxis,:]).mean(0)
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
                 activation_function_name: str,
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
        self.activation_function_name = activation_function_name
        self.activation_function = af.activation_functions_dict[activation_function_name]

class MinMaxScaler:
    """Class scaling values to [0,1] using MinMax scaling"""
    def __init__(self, vals: np.matrix):
        self.shift_factor = np.array(vals.min(1)).astype(float)
        self.scale_factor = np.array(vals.max(1) - self.shift_factor).astype(float)
        self.scale_factor = self.scale_factor + (self.scale_factor == 0).astype(float)
        self.shift_factor = self.shift_factor.reshape(self.shift_factor.shape[0], 1)
        self.scale_factor = self.scale_factor.reshape(self.scale_factor.shape[0], 1)
        self.scale_factor = 1.0

    def scale(self, vals: np.matrix) -> np.matrix:
        return (vals - self.shift_factor)/self.scale_factor

    def unscale(self, vals: np.matrix) -> np.matrix:
        return self.scale_factor*vals + self.shift_factor


class ClassificationPreparer:
    """Class translating multivariate classification to one variable classification"""
    def __init__(self, df: pd.DataFrame):
        self.df = pd.DataFrame(df).drop_duplicates().reset_index(drop=True)
        self.clf_dict = {}
        self.clf_reverse_dict = {}
        self.output_size = len(self.df)
        for i in range(len(self.df)):
            self.clf_dict[i] = tuple(self.df.iloc[[i], :].values[0])
            self.clf_reverse_dict[self.clf_dict[i]] = i

    def classification_translate_to(self, classes_values: pd.DataFrame) -> np.array:
        return classes_values.apply(lambda x:
                            self.clf_reverse_dict[tuple(x)],
                            axis=1).values

    def classification_translate_from(self, classes_nos: np.array) -> pd.DataFrame:
        return pd.DataFrame(self.df.loc[classes_nos, :]).reset_index(drop=True)

    def one_hot_encode(self, ar: np.array) -> np.array:
        return np.eye(self.output_size)[:, ar]

    def one_hot_decode(self, ar: np.array) -> np.array:
        return np.argmax(ar, axis=0)

    def save_to(self, filename: str) -> None:
        self.df.to_csv(filename)

    @classmethod
    def load_from(self, filename: str) -> ClassificationPreparer:
        return ClassificationPreparer(pd.read_csv(filename))

class NeuralNetwork:
    """Class of model of neural network"""
    def __init__(self,
                 error_function: str,
                 hidden_layers_sizes: List[int]):
        self.error_name = error_function
        self.error = af.error_functions_dict[error_function]
        self.hidden_layers_sizes = hidden_layers_sizes
        self.model_created = False
        self.fit_params = None
        self.layers_prepared = False

    def _split_train_data(self, X: np.matrix, Y: np.matrix,
                          fit_params: FitParams) -> List[Tuple[np.matrix, np.matrix]]:
        if fit_params.batch_size == 1.0:
            return [(X, Y)]
        indices = np.arange(0, X.shape[1])
        np.random.shuffle(indices)
        single_batch_size = fit_params.batch_size * X.shape[1]
        split_no = X.shape[1]//single_batch_size + (1 if X.shape[1] % single_batch_size != 0 else 0)
        indices_split = np.array_split(indices, split_no)
        res = []
        for split in indices_split:
            res.append((X[:, split], Y[:, split]))
        return res

    def _prepare_layers(self, hidden_layers_sizes: List[int], input_size: int,
                        output_size: int, fit_params: FitParams) -> None:
        if self.layers_prepared:
            return
        self.layers = []
        sizes = [input_size] + hidden_layers_sizes + [output_size]
        for i in range(len(hidden_layers_sizes)):
            self.layers.append(NeuralNetworkLayer.create_random(
              sizes[i], sizes[i+1], fit_params.bias, fit_params.bias, fit_params.activation_function))
        self.layers.append(NeuralNetworkLayer.create_random(
          sizes[-2], sizes[-1], fit_params.bias, fit_params.bias,
          af.identity_activation_function))
        self.layers_prepared = True

    def _iterate_fit(self, X: np.matrix, Y: np.matrix,
                     fit_params: FitParams) -> Tuple[List[np.matrix], float]:
        changes = []
        for layer in self.layers:
            changes.append(np.zeros_like(layer.weights))
        first_input = X
        if self.layers[0].bias:
            first_input = np.vstack((first_input,
                                    np.repeat(1.0, first_input.shape[1]).reshape(1, first_input.shape[1])))
        inputs = [first_input]
        products = []
        for j in range(len(self.layers)):
            layer = self.layers[j]
            (product, output) = layer.process_forward(inputs[-1], False)
            products.append(product)
            if j + 1 < len(self.layers) and self.layers[j+1].bias:
                next_input = np.vstack((output,
                                        np.ones((1, output.shape[1]))))
            else:
                next_input = output
            inputs.append(next_input)
        avg_error = self.error.function(inputs[-1], Y)
        error_grad = self.error.gradient(inputs[-1], Y)
        local_deltas = [None] * len(self.layers)
        for k in reversed(range(len(self.layers))):
            layer = self.layers[k]
            product = products[k]
            input = inputs[k]
            if k == len(self.layers) - 1:
                delta = layer.process_backwards_last(product, error_grad)
            else:
                next_layer = self.layers[k+1]
                no_bias_weights = next_layer.weights[:, :-1] if next_layer.bias else next_layer.weights
                delta = layer.process_backwards(product, delta, no_bias_weights)
            local_deltas[k] = delta
            changes[k] = layer.calc_change(input, delta)
        return (changes, avg_error)

    def fit(self, Xdf: pd.DataFrame, Ydf: pd.DataFrame, fit_params: FitParams) -> None:
        fit_params.x_column_names = list(Xdf.columns)
        fit_params.y_column_names = list(Ydf.columns)
        self.fit_params = fit_params
        self.X = Xdf.values.T
        if not self.layers_prepared:
            self.Xscaler = MinMaxScaler(self.X)
        X = self.Xscaler.scale(self.X)
        Y = Ydf
        input_size = len(fit_params.x_column_names)
        if fit_params.classification:
            if not self.layers_prepared:
                self.classification_preparer = ClassificationPreparer(Y)
            Y = self.classification_preparer.classification_translate_to(Y)
            Y = self.classification_preparer.one_hot_encode(Y)
            output_size = self.classification_preparer.output_size
        else:
            Y = Y.values.T
            if not self.layers_prepared:
                self.Yscaler = MinMaxScaler(Y)
            Y = self.Yscaler.scale(Y)
            output_size = Y.shape[0]

        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)
        if len(Y.shape) == 1:
            Y = Y.reshape(Y.shape[0], 1)
        self._prepare_layers(self.hidden_layers_sizes, input_size, output_size, fit_params)
        prev_changes = [None] * len(self.layers)
        self.model_created = True
        for k in range(fit_params.epochs):
            XY_arr = self._split_train_data(X, Y, fit_params)
            flag = True
            for (iter_no, XY) in enumerate(XY_arr):
                (X_it, Y_it) = XY
                (changes, avg_error) = self._iterate_fit(X_it, Y_it, fit_params)
                if fit_params.iter_callback is not None:
                    res = fit_params.iter_callback(self, avg_error, k, iter_no)
                    if not res:
                        flag = False
                        break
                for i in range(len(self.layers)):
                    prev_changes[i] = self.layers[i].update(changes[i], prev_changes[i],
                                                            fit_params.learning_rate, fit_params.momentum)
            if not flag:
                break

    def fit_df(self, df: pd.DataFrame, fit_params: FitParams) -> None:
        self.fit(pd.DataFrame(df[fit_params.x_column_names]),
                 pd.DataFrame(df.loc[:, fit_params.y_column_names], 
                    columns=fit_params.y_column_names), 
                 fit_params)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model_created:
            raise RuntimeError("Model was not created")
        X = self.Xscaler.scale(df.values.T)
        input = X
        for l in self.layers:
            (_, input) = l.process_forward(input, l.bias)
        results = input
        if self.fit_params.classification:
            cls_no = self.classification_preparer.one_hot_decode(results)
            res_df = self.classification_preparer.classification_translate_from(cls_no)
            res_df.reset_index(drop=True, inplace=True)
            return res_df
        else:
            res_df = pd.DataFrame(self.Yscaler.unscale(results).T,
                                  columns=self.fit_params.y_column_names)
            return res_df

    #def predict_single(self, single_X: np.array) -> np.array:
        #single_X = self.Xscaler.scale(single_X)
        #result = self._predict_single_raw(single_X)
        #if self.fit_params.classification:
            #cls_no = np.argmax(result)
            #return np.array(self.classification_preparer.classification_translate_to(cls_no))
        #else:
            #result = self.Yscaler.unscale(result)
            #return result

    # def _predict_single_raw(self, single_X: np.array) -> np.array:
    #     if not self.model_created:
    #         raise RuntimeError("Model was not trained")
    #     input = single_X
    #     for l in self.layers:
    #         (_, input) = l.process_forward(input, l.bias)
    #     return input

    def score(self, X_test: pd.DataFrame, Y_test: pd.DataFrame) -> Tuple[float, float]:
        if not self.model_created:
            raise RuntimeError("Model was not created")
        X = self.Xscaler.scale(X_test.values.T)
        input = X
        for l in self.layers:
            (_, input) = l.process_forward(input, l.bias)
        results = input
        if self.fit_params.classification:
            cls_no = self.classification_preparer.one_hot_decode(results)
            Y_test_translated = self.classification_preparer.classification_translate_to(Y_test)
            Y_test_translated_encoded = self.classification_preparer.one_hot_encode(Y_test_translated)
            accuracies = (cls_no == Y_test_translated).astype(int)
            accuracy = float(accuracies.mean())
            mean_squared = float(af.mean_squared_error(results, Y_test_translated_encoded))
            return (mean_squared, accuracy)
        else:
            res_unscaled = self.Yscaler.unscale(results)
            Y_test_array = Y_test.values.T
            Y_test_scaled = self.Yscaler.scale(Y_test_array)
            mean_squared = float(((results - Y_test_scaled)**2).mean())
            avg_error = float(np.abs(res_unscaled - Y_test_array).mean())
            return (mean_squared, avg_error)

    def save_to_files(self, path: str) -> None:
        if not self.model_created:
            raise ValueError("Model not created")
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        params_path = os.path.join(path, "params")
        to_json_file(self.fit_params, params_path)
        Xscaler_path = os.path.join(path, "Xscaler")
        to_json_file(self.Xscaler, Xscaler_path)
        hidden_layers_sizes_path = os.path.join(path, "hidden_layers_sizes")
        to_json_file(self.hidden_layers_sizes, hidden_layers_sizes_path)
        if self.fit_params.classification:
            preparer_path = os.path.join(path, "preparer")
            self.classification_preparer.save_to(preparer_path)
        else:
            Yscaler_path = os.path.join(path, "Yscaler")
            to_json_file(self.Yscaler, Yscaler_path)
        for (i, l) in enumerate(self.layers):
            layer_path = os.path.join(path, f"layer{i}")
            np.save(layer_path, l.weights)

    @classmethod
    def load_from_files(cls, path: str) -> NeuralNetwork:
        nn = NeuralNetwork("mean_squared", [])
        params_path = os.path.join(path, "params")
        nn.fit_params = from_json_file(params_path)
        nn.fit_params.activation_function = af.activation_functions_dict[nn.fit_params.activation_function_name]
        Xscaler_path = os.path.join(path, "Xscaler")
        nn.Xscaler = from_json_file(Xscaler_path)
        nn.layers = []
        hidden_layers_sizes_path = os.path.join(path, "hidden_layers_sizes")
        nn.hidden_layers_sizes = from_json_file(hidden_layers_sizes_path)
        if nn.fit_params.classification:
            preparer_path = os.path.join(path, "preparer")
            nn.classification_preparer = ClassificationPreparer.load_from(preparer_path)
        else:
            Yscaler_path = os.path.join(path, "Yscaler")
            nn.Yscaler = from_json_file(Yscaler_path)
        for (i, l_size) in enumerate(nn.hidden_layers_sizes):
            layer_path = os.path.join(path, f"layer{i}.npy")
            weights = np.load(layer_path)
            layer = NeuralNetworkLayer(weights, nn.fit_params.activation_function, nn.fit_params.bias)
            nn.layers.append(layer)
        layer_path = os.path.join(path, f"layer{len(nn.hidden_layers_sizes)}.npy")
        weights = np.load(layer_path)
        layer = NeuralNetworkLayer(weights,
                                   nn.fit_params.activation_function,
                                   nn.fit_params.bias)
        nn.layers.append(layer)
        nn.model_created = True
        nn.layers_prepared = True
        return nn