from threading import Thread

from MNIST_training import run_perceptron

def runp(af, layers, rate, mom, i):
    path_to_learning_set = "./train.csv"
    path_to_testing_set = "./test.csv"
    path_to_solution = "./Do_not_submit.csv"
    list_to_data = [path_to_learning_set, path_to_testing_set, path_to_solution]
    rng = 986
    run_perceptron(0.025, True, 15, af, layers, rate, mom, list_to_data, rng + i, "optimum", True, f"run6test15iter{i}")

def main():
    i = 0
    Thread(target=runp, args=("sigmoid", [256, 256], 1.0, 0.2, i)).start()
    i = i + 1
    Thread(target=runp, args=("sigmoid", [256, 256], 0.1, 0.0, i)).start()
    i = i + 1
    Thread(target=runp, args=("sigmoid", [256, 256], 0.01, 0.0, i)).start()
    i = i + 1
    Thread(target=runp, args=("sigmoid", [256, 256], 0.01, 0.0, i)).start()
    i = i + 1
    Thread(target=runp, args=("tanh", [256, 256], 1.0, 0.2, i)).start()
    i = i + 1
    Thread(target=runp, args=("tanh", [256, 256], 0.1, 0.0, i)).start()
    i = i + 1
    Thread(target=runp, args=("tanh", [256, 256], 0.01, 0.0, i)).start()
    i = i + 1
    Thread(target=runp, args=("tanh", [256, 256], 0.01, 0.0, i)).start()
    i = i + 1
    Thread(target=runp, args=("relu", [256, 256], 1.0, 0.2, i)).start()
    i = i + 1
    Thread(target=runp, args=("relu", [256, 256], 0.1, 0.0, i)).start()
    i = i + 1
    Thread(target=runp, args=("relu", [256, 256], 0.01, 0.0, i)).start()
    i = i + 1
    Thread(target=runp, args=("relu", [256, 256], 0.01, 0.0, i)).start()
    i = i + 1


if __name__ == "__main__":
    main()