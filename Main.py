import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Activation_functions as af
import configparser
import sys
import os
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

    learning_set = learning.iloc[:, :-1]
    learning_answers = learning.iloc[:, -1]
    testing_set = testing.iloc[:, :-1]
    testing_answers = learning.iloc[:, -1]

    return learning_set, learning_answers, testing_set, testing_answers


def prepare_and_run_perceptron(learning_set_path, testing_set_path):

    if error_in_paths(learning_set_path, testing_set_path):
        return

    config = configparser.ConfigParser()
    config.read('./parameters.ini')

    true_false_converter = {'yes': True, 'no': False}

    classification = config['Parameters']['classification']
    classificator = true_false_converter[classification]

    layers = config['Parameters']['number of neurons in each layer']
    layers = [int(x.strip()) for x in layers.split(',')]

    functions = {'sigmoid': af.ActivationFunction(af.sigmoid_vec, af.sigmoid_derivative_vec),
                 'tanh': af.tanh,
                 'reLU': af.reLU,
                 'identity': af.identity,
                 'leakyReLU': af.leakyReLU}

    activation_function = config['Parameters']['activation function']
    activation_function = functions[activation_function]

    epochs = int(config['Parameters']['epochs'])
    batch_size = int(config['Parameters']['batch size'])
    learning_rate = float(config['Parameters']['learning rate'])
    momentum = float(config['Parameters']['momentum'])
    bias = true_false_converter[config['Parameters']['bias']]
    rng_seed = int(config['Parameters']['rng seed'])

    # print(f'{layers}, {activation_function}, {batch_size}, {epochs}, {learning_rate}, {momentum}, {bias}, {rng_seed}, {classificator}')
    perceptron = MLP(layers, activation_function, batch_size, epochs, learning_rate, momentum, bias, rng_seed, classificator)

    (learning_set, learning_answers, testing_set, testing_answers) = get_data_for_learning(learning_set_path, testing_set_path)

    perceptron.learn(learning_set, learning_answers, testing_set, testing_answers)

    pass


if __name__ == '__main__':
    learning = r'C:\Users\Dark\Downloads\MGU_projekt1\projekt1\classification\data.three_gauss.train.100.csv'
    testing = r'C:\Users\Dark\Downloads\MGU_projekt1\projekt1\classification\data.three_gauss.test.100.csv'

    if len(sys.argv) != 3:
        print("usage: Main.py path_to_learning_set.csv path_to_testing_set.csv")
    else:
        prepare_and_run_perceptron(sys.argv[1], sys.argv[2])



