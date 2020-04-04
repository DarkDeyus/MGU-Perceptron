import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import networkx as nx
import MLP


def visualize_errors(learning_set_error, testing_set_error, save=False, path=""):
    max_length = max(len(learning_set_error), len(testing_set_error))
    x = np.arange(1, max_length + 1)
    plt.plot(x, learning_set_error, color='blue', label="Learning set error")
    plt.plot(x, testing_set_error, color='red', label='Testing set error')
    plt.xlabel('Number of weight changes')
    plt.ylabel('Error value')
    plt.legend()
    if save:
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    pass


def visualize_regression(x_learning, y_learning, x_testing, y_testing, y_predicted, save=False, path=""):
    # show learning data
    plt.scatter(x_learning, y_learning, color='black', label='Learning set')
    # show testing data
    plt.plot(x_testing, y_testing, color='red', label='Testing set')
    # show predicted data
    plt.plot(x_testing, y_predicted, color='blue', label='Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    if save:
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    pass


def visualize_classification(perceptron, x_testing, class_predicted, save=False, path=""):
    # only for points
    class_prediction = class_predicted.squeeze()
    x_values = x_testing.iloc[:, 0].to_numpy()
    y_values = x_testing.iloc[:, 1].to_numpy()
    x_min, x_max = np.min(x_values) - 1, np.max(x_values) + 1
    y_min, y_max = np.min(y_values) - 1, np.max(y_values) + 1

    # get the points
    number_of_points = 100
    x_points = np.linspace(x_min, x_max, number_of_points)
    y_points = np.linspace(y_min, y_max, number_of_points)
    xx, yy = np.meshgrid(x_points, y_points)

    # ravel flattens the array, c_ connects two arrays together, getting all points coords
    point_coords = np.c_[xx.ravel(), yy.ravel()]
    coords = pd.DataFrame(data=point_coords, columns=perceptron.net.fit_params.x_column_names)
    z = perceptron.predict(coords)
    z = z.squeeze().values.reshape(xx.shape)
    plt.contourf(xx, yy, z)  # , alpha=0.4)
    plt.scatter(x_values, y_values, c=class_prediction, edgecolor='black')  # , s=20
    if save:
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def confusion_matrix(class_actual, class_predicted, save=False, path=""):
    data = {'Actual class': class_actual.squeeze().tolist(), 'Predicted class': class_predicted.squeeze().tolist()}
    df = pd.DataFrame(data)
    matrix = pd.crosstab(df['Actual class'], df['Predicted class'],
                         rownames=['Actual class'], colnames=['Predicted class'])
    cmap = sn.cubehelix_palette(light=1, as_cmap=True)
    sn.heatmap(matrix, annot=True, cmap=cmap, linecolor='black',
               linewidths=0.5, rasterized=False, fmt='g')
    plt.title('Confusion Matrix')
    if save:
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def show_edges_weight(perceptron: MLP.MLP, save=False, path=""):
    dense = nx.Graph()
    layers = perceptron.net.layers
    previous_layer = {}
    current_row = 0
    all_neurons = {}

    for layer_number in range(len(layers)):
        curr_neuron_layer_count = layers[layer_number].weights.shape[0]
        prev_neuron_layer_count = layers[layer_number].weights.shape[1]
        # that means that bias exists
        if prev_neuron_layer_count != len(previous_layer):
            part = {i + current_row: (layer_number, i) for i in
                    range(len(previous_layer), prev_neuron_layer_count)}
            previous_layer = {**previous_layer, **part}
        # merge dictionaries, adding previous layer
        all_neurons = {**all_neurons, **previous_layer}

        if current_row == 0:
            current_row = 1000
        else:
            current_row *= 1000

        current_layer = {i + current_row: (layer_number + 1, i) for i in
                         range(0, curr_neuron_layer_count)}

        # add edges with weights
        i = 0
        j = 0
        for curr_neuron in current_layer:
            for prev_neuron in previous_layer:
                weight = round(layers[layer_number].weights[i, j], 2)
                if weight < 0:
                    weight = f"—{abs(weight)}"
                else:
                    weight = str(weight)
                dense.add_edge(curr_neuron, prev_neuron, weight=weight)
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
                                 alpha=0.9, label_pos=0.78, font_size=7,
                                 bbox=dict(color='white', alpha=0.9, edgecolor=None))
    plt.axis('off')

    if save:
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
