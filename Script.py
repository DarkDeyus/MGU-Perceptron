import itertools
import Main as m
import MLP as mlp
import Visualizer as v
import Activation_functions as af
import os
def optimum_run():

    classification = [(r'.\data.XOR.train.500.csv', r'.\data.XOR.test.500.csv')]
    regression = [(r'.\data.square.train.100.csv', r'.\data.square.test.100.csv')]
    learning_rate = 0.25
    momentum = 0.1
    bias = True
    batch_size = 0.25
    layers = [5, 5, 5]
    rng = 1337
    epoch = 100
    function = af.ActivationFunction(af.sigmoid, af.sigmoid_derivative)



    for regression_ex in regression:
        (learning_set, learning_answers, testing_set, testing_answers) = m.get_data_for_learning(regression_ex[0], regression_ex[1])

        def iter_cb(mlp, avg_error, epoch, iter):
            return m.print_iter(mlp, avg_error, epoch, iter,
                                learning_set, learning_answers,
                                testing_set, testing_answers)

        perceptron = mlp.MLP(layers, function, batch_size, epoch, learning_rate,
                             momentum, bias, rng, False, iter_cb)

        perceptron.fit(learning_set, learning_answers)
        result = perceptron.predict(testing_set)
        base_path = os.path.basename(regression_ex[0])[:-4]
        suffix = "optimum"
        v.visualize_regression(learning_set, learning_answers, testing_set, testing_answers,
                               result, True, f"{base_path}_{suffix}_regression.png")
        v.visualize_errors([], [], True, f"{base_path}_{suffix}_errors.png")
        v.show_edges_weight(perceptron, True, f"{base_path}_{suffix}_weights.png")


    for classification_ex in classification:
        (learning_set, learning_answers, testing_set, testing_answers) = m.get_data_for_learning(classification_ex[0],
                                                                                                 classification_ex[1])

        def iter_cb(mlp, avg_error, epoch, iter):
            return m.print_iter(mlp, avg_error, epoch, iter,
                                learning_set, learning_answers,
                                testing_set, testing_answers)

        perceptron = mlp.MLP(layers, function, batch_size, epoch, learning_rate,
                             momentum, bias, rng, True, iter_cb)

        perceptron.fit(learning_set, learning_answers)
        result = perceptron.predict(testing_set)
        base_path = os.path.basename(classification_ex[0])[:-4]
        suffix = "optimum"
        v.visualize_classification(perceptron, testing_set, result, True,
                              f"{base_path}_{suffix}_classification.png")
        v.confusion_matrix(testing_answers, result, True,
                               f"{base_path}_{suffix}_confusion_matrix.png")
        v.visualize_errors([], [], True, f"{base_path}_{suffix}_errors.png")
        v.show_edges_weight(perceptron, True, f"{base_path}_{suffix}_weights.png")





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



    for regression_ex in regression:
        for parameters in itertools.product(learning_rates, momentums, biases, batch_sizes, layers, functions, rngs):

            (learning_set, learning_answers, testing_set, testing_answers) = m.get_data_for_learning(regression_ex[0],
                                                                                                     regression_ex[1])

            def iter_cb(mlp, avg_error, epoch, iter):
                return m.print_iter(mlp, avg_error, epoch, iter,
                                    learning_set, learning_answers,
                                    testing_set, testing_answers)
            rate = parameters[0]
            momentum = parameters[1]
            bias = parameters[2]
            batch = parameters[3]
            layer = parameters[4]
            function = parameters[5]
            rng = parameters[6]

            base_path = os.path.basename(regression_ex[0])[:-4]
            suffix = f"rate={rate}_momentum={momentum}_bias={bias}_batch={batch}_layer={layer}_" \
                     f"function={function_description[function]}_rng={rng}"

            perceptron = mlp.MLP(layer, function, batch, epoch, rate,
                                 momentum, bias, rng, False, iter_cb)

            perceptron.fit(learning_set, learning_answers)
            result = perceptron.predict(testing_set)

            v.visualize_regression(learning_set, learning_answers, testing_set, testing_answers,
                                   result, True, f"{base_path}_{suffix}_regression.png")
            v.visualize_errors([], [], True, f"{base_path}_{suffix}_errors.png")
            v.show_edges_weight(perceptron, True, f"{base_path}_{suffix}_weights.png")

    for classification_ex in classification:
        for parameters in itertools.product(learning_rates, momentums, biases, batch_sizes, layers, functions, rngs):
            (learning_set, learning_answers, testing_set, testing_answers) = m.get_data_for_learning(classification_ex[0],
                                                                                                     classification_ex[1])

            def iter_cb(mlp, avg_error, epoch, iter):
                return m.print_iter(mlp, avg_error, epoch, iter,
                                    learning_set, learning_answers,
                                    testing_set, testing_answers)
            rate = parameters[0]
            momentum = parameters[1]
            bias = parameters[2]
            batch = parameters[3]
            layer = parameters[4]
            function = parameters[5]
            rng = parameters[6]

            perceptron = mlp.MLP(layer, function, batch, epoch, rate, momentum, bias, rng, True, iter_cb)

            perceptron.fit(learning_set, learning_answers)
            result = perceptron.predict(testing_set)
            base_path = os.path.basename(classification_ex[0])[:-4]
            suffix = f"rate={rate}_momentum={momentum}_bias={bias}_batch={batch}_layer={layer}_" \
                     f"function={function_description[function]}_rng={rng}"
            v.visualize_classification(perceptron, testing_set, result, True,
                                       f"{base_path}_{suffix}_classification.png")
            v.confusion_matrix(testing_answers, result, True,
                               f"{base_path}_{suffix}_confusion_matrix.png")
            v.visualize_errors([], [], True, f"{base_path}_{suffix}_errors.png")
            v.show_edges_weight(perceptron, True, f"{base_path}_{suffix}_weights.png")


if __name__ == "__main__":
    run_perceptrons()