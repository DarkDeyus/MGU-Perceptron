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

    learning_set = learning.iloc[:, :-1]
    learning_answers = learning.iloc[:, -1]
    testing_set = testing.iloc[:, :-1]
    testing_answers = testing.iloc[:, -1]

    return learning_set, learning_answers, testing_set, testing_answers


def prepare_and_run_perceptron(learning_set_path, testing_set_path):

    if error_in_paths(learning_set_path, testing_set_path):
        return

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

    # print(f'{layers}, {activation_function}, {batch_size}, {epochs}, {learning_rate}, {momentum}, {bias}, {rng_seed}, {classificator}')
    perceptron = MLP(layers, activation_function, batch_size, epochs,
                     learning_rate, momentum, bias, rng_seed, classification)

    (learning_set, learning_answers, testing_set, testing_answers) = get_data_for_learning(learning_set_path, testing_set_path)

    perceptron.learn(learning_set, learning_answers, testing_set, testing_answers)


if __name__ == '__main__':
    learning = r'C:\Users\Dark\Downloads\MGU_projekt1\projekt1\classification\data.three_gauss.train.100.csv'
    testing = r'C:\Users\Dark\Downloads\MGU_projekt1\projekt1\classification\data.three_gauss.test.100.csv'

    if len(sys.argv) != 3:
        print("usage: Main.py path_to_learning_set.csv path_to_testing_set.csv")
    else:
        prepare_and_run_perceptron(sys.argv[1], sys.argv[2])


