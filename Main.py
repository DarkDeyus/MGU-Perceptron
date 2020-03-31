import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Activation_functions as af
import configparser
import sys
import os
import Visualizer as v
from MLP import MLP


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

    choice = input(f"Write 'y' or 'Y' if you want to see {name_of_visualisation}, 'n' or 'N' otherwise")
    while choice.lower() not in {'y', 'n'}:
        print("Incorrect option. Try again.")
        choice = input("Write 'y' or 'Y' if you want to see {name_of_visualisation}, 'n' or 'N' otherwise")
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

    functions = {'sigmoid': af.ActivationFunction(af.sigmoid, af.sigmoid_derivative),
                 'tanh': af.ActivationFunction(af.tanh, af.tanh_derivative),
                 'reLU': af.ActivationFunction(af.reLU, af.reLU_derivative),
                 'identity': af.ActivationFunction(af.identity, af.identity_derivative),
                 'leakyReLU': af.ActivationFunction(af.leakyReLU, af.leakyReLU_derivative)}

    classification = true_false_converter.get(config['Parameters']['classification'], None)
    if classification is None:
        raise ValueError("Incorrect classification or regression setting")
        return

    activation_function = functions.get(config['Parameters']['activation function'], None)
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
    batch_size = int(config['Parameters']['batch size'])
    learning_rate = float(config['Parameters']['learning rate'])
    momentum = float(config['Parameters']['momentum'])
    rng_seed = int(config['Parameters']['rng seed'])

    print("Creating perceptron...")

    perceptron = MLP(layers, activation_function, batch_size, epochs,
                     learning_rate, momentum, bias, rng_seed, classification)

    print("Learning in progress...")
    perceptron.fit(learning_set, learning_answers)

    result = perceptron.predict(testing_set)
    print("Completed!")
    print()
    if classification:
        if ask_to_see_visualisation("confusion matrix"):
            v.confusion_matrix(testing_answers, result)
        if ask_to_see_visualisation("result of classification"):
            v.visualize_classification(perceptron, testing_set, result)
    else:
        if ask_to_see_visualisation("result of regression"):
            v.visualize_regression(learning_set, learning_answers, testing_set, testing_answers, result)
    if ask_to_see_visualisation("graph of errors over iterations"):
        v.visualize_errors([], []) #todo, brak funkcji wyciagajacej bledy po kazdej iteracji dla obu zbiorow
    if ask_to_see_visualisation("result model with weights"):
        v.show_edges_weight(perceptron)

if __name__ == '__main__':
    learning_classification = r'.\data.XOR.train.500.csv'
    testing_classification = r'.\data.XOR.test.500.csv'

    learn_reg = r'.\data.square.train.100.csv'
    test_reg = r'.\data.square.test.100.csv'

    learning_regression = r'.\data.activation.train.500.csv'
    testing_regression = r'.\data.activation.test.500.csv'
    prepare_and_run_perceptron(learning_classification, testing_classification)
    #prepare_and_run_perceptron(learning_regression, testing_regression)
    #prepare_and_run_perceptron(learn_reg, test_reg)
    sys.exit(0)

    if len(sys.argv) != 3:
        print("usage: Main.py path_to_learning_set.csv path_to_testing_set.csv")
    else:
        prepare_and_run_perceptron(sys.argv[1], sys.argv[2])


