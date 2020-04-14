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
            print("MSE:", mean_squared_error_test)
        return True

    perceptron = MLP(layers, function, batch_size, epoch, learning_rate,
                     momentum, bias, rng, classification, iter_cb)
    perceptron.fit(learning_set, learning_answers)
    result = perceptron.predict(testing_set)
    base_path = os.path.basename(list_of_paths_to_data[0])[:-4]
    perceptron.net.save_to_files(f"{base_path}_{suffix}_net")

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

    with open(f"{base_path}_{suffix}_results.txt", 'w') as file:
        file.write("mean_squared_errors_test: ")
        file.writelines(f"{error} " for error in mean_squared_errors_test)
        file.write('\n')
        file.write("mean_squared_errors_train: ")
        file.writelines(f"{error} " for error in mean_squared_errors_train)
        file.write('\n')
        file.write("avg_acc_errors_test: ")
        file.writelines(f"{error} " for error in avg_acc_errors_test)
        file.write('\n')
        file.write("avg_acc_errors_train: ")
        file.writelines(f"{error} " for error in avg_acc_errors_train)


def run_optimum():
    regression = []
    classification = []
    reg_path = "data/projekt1/regression"
    reg_raw_files = os.listdir(reg_path)
    cls_path = "data/projekt1/classification"
    cls_raw_files = os.listdir(cls_path)
    for file in reg_raw_files:
        f1 = reg_path + "/" + file
        f2 = f1.replace("projekt1", "projekt1_test").replace("train", "test")
        regression.append((f1, f2))
    for file in cls_raw_files:
        f1 = cls_path + "/" + file
        f2 = f1.replace("projekt1", "projekt1_test").replace("train", "test")
        classification.append((f1, f2))
    #classification = [(r'.\data.XOR.train.500.csv', r'.\data.XOR.test.500.csv')]
    #regression = [(r'.\data.square.train.100.csv', r'.\data.square.test.100.csv')]

    learning_rate = 1.0
    momentum = 0.1
    bias = True
    batch_size = 0.25
    layers = [4, 4, 4]
    rng = 6235
    epoch = 100000
    function = "sigmoid"

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
    functions = ["sigmoid", "tanh"]
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