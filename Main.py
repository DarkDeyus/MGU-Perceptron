import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Activation_functions as af
import Neural_network as nn
import configparser
import sys
import os
import Visualizer as v
from MLP import MLP
from typing import Tuple
from math import ceil

def print_iter(perceptron: MLP, avg_error: float, epoch: int, iter: int) -> None:
    print("--------------------------------")
    print(f"Epoch {epoch}, iter {iter}: mean_squared_network_error {avg_error}")
    #if perceptron.classification:
        #acc_t = np.array(Y_train_predicted == Y_train).astype(int)
        #acc = np.array(Y_test_predicted == Y_test).astype(int)
        #acc_t = sum(acc_t)/len(acc_t)
        #acc = sum(acc)/len(acc)
        #print("Train acc", acc_t)
        #print("Test acc", acc)
    #else:
        #sigma_t = np.sqrt((np.array(Y_train_predicted - Y_train)**2).mean())
        #sigma = np.sqrt((np.array(Y_test_predicted - Y_test)**2).mean())
        #print("Train standard deviation", sigma_t)
        #print("Test standard deviation", sigma)
    #v.show_edges_weight(mlp)
    #pass
#   for (i, l) in enumerate(net.layers):
#       print(f"Layer {i}")
#       print(l.weights)
#       print("\n")
#       print("\n")

def score_perceptron(perceptron: MLP,
                     X_train: pd.DataFrame,
                     Y_train: pd.DataFrame,
                     X_test: pd.DataFrame,
                     Y_test: pd.DataFrame) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    Y_train_scored = perceptron.net.score(X_train, Y_train)
    Y_test_scored = perceptron.net.score(X_test, Y_test)
    return (Y_train_scored, Y_test_scored)
        

def error_in_paths(learning_set_path, testing_set_path):
    error = False
    if not os.path.isfile(learning_set_path) or not learning_set_path.endswith('.csv'):
        print('Incorrect path to learning set, or learning set is not a .csv file')
        error = True

    if not os.path.isfile(testing_set_path) or not testing_set_path.endswith('.csv'):
        print('Incorrect path to testing set, or testing set is not a .csv file')
        error = True
    return error


def get_data_for_learning(learning_set_path, testing_set_path):
    learning = pd.read_csv(learning_set_path)
    testing = pd.read_csv(testing_set_path)

    learning_set = pd.DataFrame(learning.iloc[:, :-1])
    learning_answers = pd.DataFrame(learning.iloc[:, -1])
    testing_set = pd.DataFrame(testing.iloc[:, :-1])
    testing_answers = pd.DataFrame(testing.iloc[:, -1])

    return learning_set, learning_answers, testing_set, testing_answers


def ask_to_see_visualisation(name_of_visualisation: str) -> bool:

    choice = input(f"Write 'y' or 'Y' if you want to see {name_of_visualisation}, 'n' or 'N' otherwise\n")
    while choice.lower() not in {'y', 'n'}:
        print("Incorrect option. Try again.\n")
        choice = input(f"Write 'y' or 'Y' if you want to see {name_of_visualisation}, 'n' or 'N' otherwise\n")
    return choice.lower() == 'y'

#todo -> nie działa, sa zle wyniki
def minMaxScale(X: pd.DataFrame) -> pd.DataFrame:
    def minMaxSeries(s: pd.Series):
        min = s.min()
        max = s.max()
        return (s - min) / (max - min)

    return X.apply(minMaxSeries)


def prepare_and_run_perceptron(learning_set_path, testing_set_path):

    if error_in_paths(learning_set_path, testing_set_path):
        return

    print("Loading data...")
    (learning_set, learning_answers, testing_set, testing_answers) = get_data_for_learning(learning_set_path, testing_set_path)

    a = minMaxScale(learning_set)
    print("Loading perceptron parameters...")
    config = configparser.ConfigParser()
    config.read('./parameters.ini')

    true_false_converter = {'yes': True, 'no': False}

    classification = true_false_converter.get(config['Parameters']['classification'], None)
    if classification is None:
        raise ValueError("Incorrect classification or regression setting")
        return

    activation_function = config['Parameters']['activation function']
    if activation_function is None:
        raise ValueError("Incorrect activation function")
        return

    bias = true_false_converter.get(config['Parameters']['bias'], None)
    if bias is None:
        raise ValueError("Incorrect bias setting")
        return

    layers = config['Parameters']['number of neurons in each layer']
    layers = [int(x.strip()) for x in layers.split(',')]

    epochs = int(config['Parameters']['epochs'])
    batch_size = float(config['Parameters']['batch size'])
    learning_rate = float(config['Parameters']['learning rate'])
    momentum = float(config['Parameters']['momentum'])
    rng_seed = int(config['Parameters']['rng seed'])

    print("Creating perceptron...")

    mean_squared_errors_test = []
    mean_squared_errors_train = []

    avg_acc_errors_test = []
    avg_acc_errors_train = []

    epoch_points_size = 10
    epoch_separation = epochs//epoch_points_size if epochs >= epoch_points_size else 1

    epoch_measure_points = []
    iters_in_epoch = ceil(1.0/batch_size)

    def iter_cb(mlp, avg_error, epoch, iter):
        if (epoch + 1) % epoch_separation == 0 and iter == iters_in_epoch - 1:
            print_iter(mlp, avg_error, epoch + 1, iter)
            (train_errors, test_errors) = score_perceptron(mlp,
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

    perceptron = MLP(layers, activation_function, batch_size, epochs,
                     learning_rate, momentum, bias, rng_seed, classification,
                     iter_cb)

    print("Learning in progress...")
    perceptron.fit(learning_set, learning_answers)

    result = perceptron.predict(testing_set)
    print("Completed!")
    print()
    oldnet = perceptron.net
    perceptron.net.save_to_files("save_test")
    perceptron.net = nn.NeuralNetwork.load_from_files("save_test")
    perceptron.net.fit_params.iter_callback = iter_cb
    perceptron.fit(learning_set, learning_answers)
    if classification:
        if ask_to_see_visualisation("confusion matrix"):
            v.confusion_matrix(testing_answers, result)
        if ask_to_see_visualisation("result of classification"):
            v.visualize_classification(perceptron, testing_set, result)
        if ask_to_see_visualisation("accuracy plot"):
            v.visualize_accuracy(avg_acc_errors_train, avg_acc_errors_test, epoch_measure_points)
    else:
        if ask_to_see_visualisation("result of regression"):
            v.visualize_regression(learning_set, learning_answers, testing_set, testing_answers, result)
        if ask_to_see_visualisation("average error values plot"):
            v.visualize_avg_errors(avg_acc_errors_train, avg_acc_errors_test, epoch_measure_points)

    if ask_to_see_visualisation("mean square errors plot"):
        v.visualize_mean_sqrt_errors(mean_squared_errors_train, mean_squared_errors_test, epoch_measure_points)
    if ask_to_see_visualisation("result model with weights"):
        v.show_edges_weight(perceptron)

if __name__ == '__main__':
    learning_classification = r'.\data.XOR.train.500.csv'
    testing_classification = r'.\data.XOR.test.500.csv'

    learn_reg = r'.\data.square.train.100.csv'
    test_reg = r'.\data.square.test.100.csv'

    learning_regression = r'.\data.activation.train.500.csv'
    testing_regression = r'.\data.activation.test.500.csv'
    #prepare_and_run_perceptron(learning_classification, testing_classification)
    prepare_and_run_perceptron(learning_regression, testing_regression)
    #prepare_and_run_perceptron(learn_reg, test_reg)
    sys.exit(0)

    if len(sys.argv) != 3:
        print("usage: Main.py path_to_learning_set.csv path_to_testing_set.csv")
    else:
        prepare_and_run_perceptron(sys.argv[1], sys.argv[2])


