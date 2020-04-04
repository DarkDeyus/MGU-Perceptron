import itertools
import Main as m
from MLP import MLP
import Visualizer as v
import Activation_functions as af
import os
from math import ceil


def run_perceptron(batch_size, bias, epoch, function, layers, learning_rate, momentum, list_of_paths_to_data, rng,
                   suffix, classification):
    (learning_set, learning_answers, testing_set, testing_answers) = m.get_data_for_learning(list_of_paths_to_data[0],
                                                                                             list_of_paths_to_data[1])
    mean_squared_errors_test = []
    mean_squared_errors_train = []
    avg_acc_errors_test = []
    avg_acc_errors_train = []
    epoch_points_size = 100
    epoch_separation = epoch // epoch_points_size if epoch >= epoch_points_size else 1
    epoch_measure_points = []
    iters_in_epoch = ceil(1.0 / batch_size)

    def iter_cb(mlp, avg_error, epoch, iter):
        if (epoch + 1) % epoch_separation == 0 and iter == iters_in_epoch - 1:
            m.print_iter(mlp, avg_error, epoch + 1, iter)
            (train_errors, test_errors) = m.score_perceptron(mlp,
                                                             learning_set,
                                                             learning_answers,
                                                             testing_set,
                                                             testing_answers)
            (mean_squared_error_train, avg_acc_error_train) = train_errors
            (mean_squared_error_test, avg_acc_error_test) = test_errors
            mean_squared_errors_train.append(mean_squared_error_train)
            mean_squared_errors_test.append(mean_squared_error_test)
            avg_acc_errors_train.append(avg_acc_error_train)
            avg_acc_errors_test.append(avg_acc_error_test)
            epoch_measure_points.append(epoch + 1)
        return True

    perceptron = MLP(layers, function, batch_size, epoch, learning_rate,
                     momentum, bias, rng, classification, iter_cb)
    perceptron.fit(learning_set, learning_answers)
    result = perceptron.predict(testing_set)
    base_path = os.path.basename(list_of_paths_to_data[0])[:-4]

    if classification:
        v.visualize_classification(perceptron, testing_set, result, True,
                                   f"{base_path}_{suffix}_classification.png")
        v.confusion_matrix(testing_answers, result, True,
                           f"{base_path}_{suffix}_confusion_matrix.png")
        v.visualize_accuracy(avg_acc_errors_train, avg_acc_errors_test, epoch_measure_points, True,
                             f"{base_path}_{suffix}_accuracy.png")
    else:
        v.visualize_regression(learning_set, learning_answers, testing_set, testing_answers,
                               result, True, f"{base_path}_{suffix}_regression.png")
        v.visualize_avg_errors(avg_acc_errors_train, avg_acc_errors_test, epoch_measure_points, True,
                               f"{base_path}_{suffix}_avg_errors.png")

    v.visualize_mean_sqrt_errors(mean_squared_errors_train, mean_squared_errors_test, epoch_measure_points,
                       True, f"{base_path}_{suffix}_mean_square_errors.png")
    v.show_edges_weight(perceptron, True, f"{base_path}_{suffix}_weights.png")


def run_optimum():
    classification = [(r'.\data.XOR.train.500.csv', r'.\data.XOR.test.500.csv')]
    regression = [(r'.\data.square.train.100.csv', r'.\data.square.test.100.csv')]
    learning_rate = 0.25
    momentum = 0.1
    bias = True
    batch_size = 0.25
    layers = [5, 5, 5]
    rng = 1337
    epoch = 10
    function = af.ActivationFunction(af.sigmoid, af.sigmoid_derivative)

    for example in regression:
        run_perceptron(batch_size, bias, epoch, function, layers, learning_rate, momentum, example, rng,
                       "optimum", False)

    for example in classification:
        run_perceptron(batch_size, bias, epoch, function, layers, learning_rate, momentum, example, rng,
                       "optimum", True)


def run_perceptrons():
    classification = [(r'.\data.XOR.train.500.csv', r'.\data.XOR.test.500.csv')]
    regression = [(r'.\data.square.train.100.csv', r'.\data.square.test.100.csv')]
    learning_rates = [0.1, 0.5, 1]
    momentums = [0.0, 0.1]
    biases = [True, False]
    batch_sizes = [0.25, 1]
    layers = [[], [1], [5], [3, 3], [5, 5, 5], [4, 4, 4, 4]]
    functions = [af.ActivationFunction(af.sigmoid, af.sigmoid_derivative),
                 af.ActivationFunction(af.tanh, af.tanh_derivative)]
    function_description = {functions[0]: "Sigmoid", functions[1]: "Tanh"}
    rngs = [123, 1337]
    epoch = 10000

    for example in regression:
        for parameters in itertools.product(learning_rates, momentums, biases, batch_sizes, layers, functions, rngs):
            (rate, momentum, bias, batch_size, layer, function, rng) = parameters
            suffix = f"rate={rate}_momentum={momentum}_bias={bias}_batch={batch_size}_layer={layer}_" \
                     f"function={function_description[function]}_rng={rng}"

            run_perceptron(batch_size, bias, epoch, function, layer, rate, momentum, example, rng, suffix, False)

    for example in classification:
        for parameters in itertools.product(learning_rates, momentums, biases, batch_sizes, layers, functions, rngs):
            (rate, momentum, bias, batch_size, layer, function, rng) = parameters
            suffix = f"rate={rate}_momentum={momentum}_bias={bias}_batch={batch_size}_layer={layer}_" \
                     f"function={function_description[function]}_rng={rng}"

            run_perceptron(batch_size, bias, epoch, function, layer, rate, momentum, example, rng, suffix, True)


if __name__ == "__main__":
    #run_perceptrons()
    run_optimum()