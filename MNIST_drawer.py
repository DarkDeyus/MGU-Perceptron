from MNIST_training import visualize_perceptron, get_data_for_learning
import pandas as pd
import os
import Neural_network as nn

def main():
    name = "train_optimum"
    path_to_learning_set = "./train.csv"
    path_to_testing_set = "./test.csv"
    path_to_solution = "./Do_not_submit.csv"
    (learning_set, learning_answers, testing_set, testing_answers) = get_data_for_learning(path_to_learning_set,
                                                                                           path_to_testing_set,
                                                                                           path_to_solution)
    prefix = "run16"
    suffix = "optimum"
    for i in range(1):
        base_path = f"{prefix}{i}_{os.path.basename(path_to_learning_set)[:-4]}"
        fname = f"{prefix}_{name}"
        res_name = fname + "_results.txt"
        net = nn.NeuralNetwork.load_from_files(f"{prefix}_train")
        #result.to_csv("f./{base_path}_{suffix}_result.csv", index=False)
        result = pd.read_csv(fname + "_result.csv").reset_index(drop=True).iloc[:, -1]
        with open(res_name, 'r') as f:
            content = f.readlines()
        split = []
        for line in content:
            split.append(line.split(" ")[1:])
        fsplit = []
        for line in split:
            fline = []
            for num in line:
                if num == "\n" or num == "":
                    continue
                fline.append(float(num)) 
            fsplit.append(fline)
        mse_test = fsplit[0]
        mse_train = fsplit[1]
        avg_acc_test = fsplit[2]
        avg_acc_train = fsplit[3]
        epoch_measure_points = fsplit[4] if len(fsplit) > 4 else list(range(len(mse_test)))
        visualize_perceptron(base_path, suffix, mse_train, mse_test, avg_acc_train, avg_acc_test,
            learning_set, learning_answers, epoch_measure_points, testing_set, testing_answers, result, net)


if __name__ == "__main__":
    main()