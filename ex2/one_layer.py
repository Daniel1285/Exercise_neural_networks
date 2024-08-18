import csv
import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters:
    z (numpy array): Input data

    Returns:
    numpy array: Sigmoid of z
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_der(z):
    """
    Compute the derivative of the sigmoid function

    Parameters:
    z (numpy array): Input data

    Returns:
    numpy array: Derivative of sigmoid of z
    """
    A = sigmoid(z)
    return A * (1 - A)


def tanh(z):
    """
    Compute the hyperbolic tangent of z

    Parameters:
    z (numpy array): Input data

    Returns:
    numpy array: Hyperbolic tangent of z
    """
    return np.tanh(z)


def tanh_der(z):
    """
    Compute the derivative of the hyperbolic tangent function

    Parameters:
    z (numpy array): Input data

    Returns:
    numpy array: Derivative of hyperbolic tangent of z
    """
    return 1 - np.tanh(z) ** 2


def relu(z):
    """
    Compute the Rectified Linear Unit (ReLU) of z

    Parameters:
    z (numpy array): Input data

    Returns:
    numpy array: ReLU of z
    """
    return np.maximum(0, z)


def relu_der(z):
    """
    Compute the derivative of the Rectified Linear Unit (ReLU) function

    Parameters:
    z (numpy array): Input data

    Returns:
    numpy array: Derivative of ReLU of z
    """
    return np.where(z > 0, 1, 0)


def initialize_parameters(n_x, n_h, n_y):
    """
    Initialize parameters for a 1-layer neural network

    Parameters:
    n_x (int): Number of input features
    n_h (int): Number of neurons in the hidden layer
    n_y (int): Number of output neurons

    Returns:
    dict: Initialized parameters W1, b1, W2, b2
    """
    return {
        "W1": np.random.randn(n_h, n_x) * 0.01,
        "b1": np.zeros([n_h, 1]),
        "W2": np.random.randn(n_y, n_h) * 0.01,
        "b2": np.zeros([n_y, 1]),
    }


print(initialize_parameters(4, 3, 1))


def forward_propagation(X, parameters, activation):
    """
    Perform forward propagation through the neural network

    Parameters:
    X (numpy array): Input data
    parameters (dict): Neural network parameters
    activation (str): Activation function to use in the hidden layer

    Returns:
    A2 (numpy array): Output of the neural network
    cache (dict): Cached values needed for backward propagation
    """
    # Hidden Layer
    Z1 = parameters["W1"].dot(X) + parameters["b1"]
    if activation == "tanh":
        A1 = tanh(Z1)
    elif activation == "relu":
        A1 = relu(Z1)
    elif activation == "sigmoid":
        A1 = sigmoid(Z1)

    # Output Layer
    Z2 = parameters["W2"].dot(A1) + parameters["b2"]
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache


def backward_propagation(parameters, cache, X, Y, activation):
    """
    Perform backward propagation to compute gradients

    Parameters:
    parameters (dict): Neural network parameters
    cache (dict): Cached values from forward propagation
    X (numpy array): Input data
    Y (numpy array): True labels
    activation (str): Activation function used in the hidden layer

    Returns:
    dict: Gradients dW1, dW2, db1, db2
    """
    m = X.shape[1]  # Number of samples
    dZ2 = cache["A2"] - Y  # for the sigmoid layer
    dW2 = (1 / m) * dZ2.dot(cache["A1"].T)
    db2 = (1 / m) * np.sum(dZ2)

    # Hidden Layer
    dA1 = np.dot(parameters["W2"].T, dZ2)
    if activation == "tanh":
        dZ1 = dA1 * tanh_der(cache["Z2"])
    elif activation == "relu":
        dZ1 = dA1 * relu_der(cache["Z2"])
    elif activation == "sigmoid":
        dZ1 = dA1 * sigmoid_der(cache["Z2"])
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1)

    return {"dW1": dW1, "dW2": dW2, "db1": db1, "db2": db2}


def LogLoss_calculation(A, Y):
    """
    Compute the logistic loss

    Parameters:
    A (numpy array): Predictions
    Y (numpy array): True labels

    Returns:
    float: Logistic loss
    """
    cost = np.mean(-(Y * np.log(A) + (1 - Y) * np.log(1 - (A))))
    return cost


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Parameters:
    parameters (dict): Neural network parameters
    grads (dict): Gradients
    learning_rate (float): Learning rate

    Returns:
    dict: Updated parameters
    """
    return {
        "W1": parameters["W1"] - learning_rate * grads["dW1"],
        "W2": parameters["W2"] - learning_rate * grads["dW2"],
        "b1": parameters["b1"] - learning_rate * grads["db1"],
        "b2": parameters["b2"] - learning_rate * grads["db2"],
    }


def nn_model(X, Y, iterations, lr, num_of_n, activations):
    """
    Train a neural network model

    Parameters:
    X (numpy array): Input data
    Y (numpy array): True labels
    iterations (int): Number of training iterations
    lr (float): Learning rate
    num_of_n (int): Number of neurons in the hidden layer
    activations (str): Activation function to use in the hidden layer

    Returns:
    dict: Trained parameters
    list: Cost over iterations
    """
    n_x = X.shape[0]
    n_h = num_of_n
    n_y = 1
    parameters = initialize_parameters(n_x, n_h, n_y)
    print("Network shape ", X.shape[0], n_h, n_y)
    for i in range(iterations):
        A2, cache = forward_propagation(X, parameters, activations)
        cost = LogLoss_calculation(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y, activations)
        parameters = update_parameters(parameters, grads, lr)
        costs.append(cost)
    return parameters, costs


def predict(X, parameters, activations):
    """
    Make predictions using the trained model

    Parameters:
    X (numpy array): Input data
    parameters (dict): Trained parameters
    activations (str): Activation function to use in the hidden layer

    Returns:
    numpy array: Predictions
    """
    A2, cache = forward_propagation(X, parameters, activations)
    return np.rint(A2)


def prediction_accuracy(y_pred, y_true):
    """
    Compute prediction accuracy

    Parameters:
    y_pred (numpy array): Predicted labels
    y_true (numpy array): True labels

    Returns:
    float: Prediction accuracy
    """
    return np.mean(y_pred == y_true)


# Load data
url = 'https://github.com/rosenfa/nn/blob/master/pima-indians-diabetes.csv?raw=true'
df = pd.read_csv(url, header=0)
features = df.drop(['Outcome'], axis=1)
features = ((features - features.mean()) / features.std())
X = np.array(features)
Y = np.array(df['Outcome'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

# Train logistic regression model for comparison
sk_model = LogisticRegression()
sk_model.fit(X_train, Y_train)
accuracy = sk_model.score(X_test, Y_test)
print("accuracy = ", accuracy * 100, "%")

# Prepare data for neural network
X_train, X_test = X_train.T, X_test.T

# Training neural network
learning_rate = 1
nodes_list = [1, 2, 3, 4, 5, 6]
activations = ["tanh", "relu", "sigmoid"]
train_results = []
test_results = []

for item in activations:
    print(f"{Fore.YELLOW}{Style.BRIGHT}activation: {item}{Style.RESET_ALL}")
    for iterations in range(500, 2500, 500):
        print(f"{Fore.GREEN}{Style.BRIGHT}num iterations : {iterations}{Style.RESET_ALL}")
        for node in range(1, 7):
            print(f"{Fore.RED}{Style.BRIGHT}num of node: {node}{Style.RESET_ALL}")
            alpha = 1
            costs = []
            parameters, costs = nn_model(X_train, Y_train, iterations, alpha, num_of_n=node, activations=item)
            Y_train_predict = predict(X_train, parameters, activations=item)
            train_acc = prediction_accuracy(Y_train_predict, Y_train)
            Y_test_predict = predict(X_test, parameters, activations=item)
            test_acc = prediction_accuracy(Y_test_predict, Y_test)
            parameters["train_accuracy"] = train_acc
            parameters["test_accuracy"] = test_acc

            print("Training acc : ", str(train_acc))
            print("Testing acc : ", str(test_acc))
            train_results.append({"activation": item, "nodes": node, "iterations": iterations, "accuracy": train_acc})
            test_results.append({"activation": item, "nodes": node, "iterations": iterations, "accuracy": test_acc})


def write_results_to_csv(train_results, test_results, train_filename, test_filename):
    """
    Write training and testing results to CSV files

    Parameters:
    train_results (list): Training results
    test_results (list): Testing results
    train_filename (str): Training results CSV filename
    test_filename (str): Testing results CSV filename
    """
    header = ["Activation", "Nodes", "accuracy 1 lay-500 iter", "accuracy 1 lay-1000 iter", "accuracy 1 lay-1500 iter",
              "accuracy 1 lay-2000 iter"]

    # Write training results
    with open(train_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for activation in activations:
            writer.writerow([activation])
            for nodes in range(1, 7):
                row = [activation, f"{nodes} nodes"]
                for iterations in range(500, 2500, 500):
                    accuracy = next((item['accuracy'] for item in train_results if
                                     item["activation"] == activation and item["nodes"] == nodes and item[
                                         "iterations"] == iterations), None)
                    row.append(round(accuracy, 5))
                writer.writerow(row)

    # Write testing results
    with open(test_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for activation in activations:
            writer.writerow([activation])
            for nodes in range(1, 7):
                row = [activation, f"{nodes} nodes"]
                for iterations in range(500, 2500, 500):
                    accuracy = next((item['accuracy'] for item in test_results if
                                     item["activation"] == activation and item["nodes"] == nodes and item[
                                         "iterations"] == iterations), None)
                    row.append(round(accuracy, 5))
                writer.writerow(row)


write_results_to_csv(train_results, test_results, "train_results.csv", "test_results.csv")
print("Training results have been written to train_results.csv")
print("Testing results have been written to test_results.csv")
