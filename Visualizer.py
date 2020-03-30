import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import networkx as nx
import MLP


def visualize_errors(learning_set_error, testing_set_error):
    max_length = max(len(learning_set_error), len(testing_set_error))
    x = np.arange(1, max_length + 1)
    plt.plot(x, learning_set_error, color='blue', label="Learning set error")
    plt.plot(x, testing_set_error, color='red', label='Testing set error')
    plt.xlabel('Number of weight changes')
    plt.ylabel('Error value')
    plt.legend()
    plt.show()
    pass


def visualize_regression(x_learning, y_learning, x_testing, y_testing, y_predicted):
    # show learning data
    plt.scatter(x_learning, y_learning, color='black', label='Learning set')
    # show testing data
    plt.plot(x_testing, y_testing, color='red', label='Testing set')
    # show predicted data
    plt.plot(x_testing, y_predicted, color='blue', label='Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    pass


def visualize_classification(perceptron, x_testing, y_testing, class_predicted):
    x_min, x_max = np.min(x_testing) - 1, np.max(x_testing) + 1
    y_min, y_max = np.min(y_testing) - 1, np.max(y_testing) + 1

    # get the points separated by 0.01
    step_size = 0.01
    x_points = np.arange(x_min, x_max, step_size)
    y_points = np.arange(y_min, y_max, step_size)
    xx, yy = np.meshgrid(x_points, y_points)

    # ravel flattens the array, c_ connects two arrays together, getting all points coords
    z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])  # fix for perceptron predict call
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z)  # , alpha=0.4)
    plt.scatter(x_testing, y_testing, c=class_predicted, edgecolor='black')  # , s=20
    plt.show()


def confusion_matrix(class_actual, class_predicted):
    data = {'Actual class': class_actual, 'Predicted class': class_predicted}
    df = pd.DataFrame(data)
    matrix = pd.crosstab(df['Actual class'], df['Predicted class'],
                         rownames=['Actual class'], colnames=['Predicted class'])
    cmap = sn.cubehelix_palette(light=1, as_cmap=True)
    sn.heatmap(matrix, annot=True, cmap=cmap, linecolor='black',
               linewidths=0.5, rasterized=False)
    plt.title('Confusion Matrix')
    plt.show()


def show_edges_weight(perceptron: MLP.MLP):
    dense = nx.Graph()
    layers = perceptron.net.layers
    previous_layer = {}
    current_row = 0
    all_neurons = {}

    for layer_number in range(len(layers)):
        curr_neuron_layer_count = layers[layer_number].weights.shape[0]
        prev_neuron_layer_count = layers[layer_number].weights.shape[1]
        if prev_neuron_layer_count != len(previous_layer):
            part = {i + current_row: (layer_number, i) for i in
                               range(len(previous_layer), prev_neuron_layer_count)}
            previous_layer = {**previous_layer, **part}
        #merge dictionaries, adding previous layer
        all_neurons = {**all_neurons, **previous_layer}

        if current_row == 0:
            current_row = 1000
        else:
            current_row *= 1000

        current_layer = {i + current_row: (layer_number + 1, i) for i in
                         range(0, curr_neuron_layer_count)}

        #add edges with weights
        i = 0
        j = 0
        for curr_neuron in current_layer:
            for prev_neuron in previous_layer:
                dense.add_edge(curr_neuron, prev_neuron, weight=round(layers[layer_number].weights[i, j], 2))
                j += 1
            i += 1
            j = 0

        previous_layer = current_layer

    all_neurons = {**all_neurons, **previous_layer}

    nx.draw_networkx_nodes(dense, all_neurons,
                           nodelist=all_neurons.keys(), node_color='black')
    nx.draw_networkx_edges(dense, all_neurons, edge_color='green')
    labels = nx.get_edge_attributes(dense, 'weight')
    nx.draw_networkx_edge_labels(dense, all_neurons, edge_labels=labels,
                                 alpha=0.7, label_pos=0.78, font_size=7,
                                 bbox=dict(color='white', alpha=0.7, edgecolor=None))
    axes = plt.axis('off')

    plt.show()
