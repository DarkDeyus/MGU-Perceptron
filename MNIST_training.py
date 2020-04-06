from math import ceil
import pandas as pd
import numpy as np
import Main as m
from MLP import MLP
import os
import Visualizer as v


def run_perceptron(batch_size, bias, epoch, function, layers, learning_rate, momentum, list_of_paths_to_data, rng,
                   suffix, classification):
    (learning_set, learning_answers, testing_set, testing_answers) = get_data_for_learning(list_of_paths_to_data[0],
                                                                                           list_of_paths_to_data[1],
                                                                                           list_of_paths_to_data[2])
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

    # save to csv
    res = result.rename(columns={result.columns[0]: "Label"})
    res.insert(loc=0, column='ImageId', value=np.arange(1, len(res) + 1))
    res.to_csv("./result.csv", index=False)


def get_data_for_learning(learning_set_path, testing_set_path, solution_csv):
    learning = pd.read_csv(learning_set_path)
    testing = pd.read_csv(testing_set_path)
    solution = pd.read_csv(solution_csv)
    learning_set = pd.DataFrame(learning.iloc[:, 1:])
    learning_answers = pd.DataFrame(learning.iloc[:, 0])
    testing_set = pd.DataFrame(testing)
    testing_answers = pd.DataFrame(solution.iloc[:, -1])

    return learning_set, learning_answers, testing_set, testing_answers


def run_optimum():
    path_to_learning_set = "./train.csv"
    path_to_testing_set = "./test.csv"
    path_to_solution = "./Do_not_submit.csv"
    learning_rate = 0.25
    momentum = 0.1
    bias = True
    batch_size = 0.25
    layers = [5, 5, 5]
    rng = 1337
    epoch = 10
    function = "sigmoid"

    run_perceptron(batch_size, bias, epoch, function, layers, learning_rate, momentum,
                   [path_to_learning_set, path_to_testing_set, path_to_solution], rng, "optimum", True)


if __name__ == "__main__":
    run_optimum()
