import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from colorama import Fore, Style


# Activation functions and their derivatives
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


# Initialize parameters for a 2-layer network
def initialize_parameters(n_x, n_h1, n_h2, n_y):
    """
    Initialize parameters for a three-layer neural network

    Parameters:
    n_x (int): Number of input features
    n_h1 (int): Number of neurons in the first hidden layer
    n_h2 (int): Number of neurons in the second hidden layer
    n_y (int): Number of output neurons

    Returns:
    dict: Initialized parameters W1, b1, W2, b2, W3, b3
    """
    np.random.seed(1)
    parameters = {
        "W1": np.random.randn(n_h1, n_x) * 0.01,
        "b1": np.zeros((n_h1, 1)),
        "W2": np.random.randn(n_h2, n_h1) * 0.01,
        "b2": np.zeros((n_h2, 1)),
        "W3": np.random.randn(n_y, n_h2) * 0.01,
        "b3": np.zeros((n_y, 1)),
    }
    return parameters


# Forward propagation
def forward_propagation(X, parameters):
    """
    Perform forward propagation through the neural network

    Parameters:
    X (numpy array): Input data
    parameters (dict): Neural network parameters

    Returns:
    numpy array: Output of the neural network
    dict: Cached values needed for backward propagation
    """
    Z1 = np.dot(parameters["W1"], X) + parameters["b1"]
    A1 = tanh(Z1)
    Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]
    A2 = tanh(Z2)
    Z3 = np.dot(parameters["W3"], A2) + parameters["b3"]
    A3 = sigmoid(Z3)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, cache


# Backward propagation
def backward_propagation(parameters, cache, X, Y):
    """
    Perform backward propagation to compute gradients

    Parameters:
    parameters (dict): Neural network parameters
    cache (dict): Cached values from forward propagation
    X (numpy array): Input data
    Y (numpy array): True labels

    Returns:
    dict: Gradients dW1, dW2, dW3, db1, db2, db3
    """
    m = X.shape[1]
    dZ3 = cache["A3"] - Y
    dW3 = (1 / m) * np.dot(dZ3, cache["A2"].T)
    db3 = (1 / m) * np.sum(dZ3)
    dA2 = np.dot(parameters["W3"].T, dZ3)
    dZ2 = dA2 * tanh_der(cache["Z2"])
    dW2 = (1 / m) * np.dot(dZ2, cache["A1"].T)
    db2 = (1 / m) * np.sum(dZ2)
    dA1 = np.dot(parameters["W2"].T, dZ2)
    dZ1 = dA1 * tanh_der(cache["Z1"])
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1)
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return grads


# Compute the cost
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


# Update parameters
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
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    parameters["W3"] -= learning_rate * grads["dW3"]
    parameters["b3"] -= learning_rate * grads["db3"]
    return parameters


# Neural Network model
def nn_model(X, Y, n_h1, n_h2, learning_rate, iterations):
    """
    Train a neural network model

    Parameters:
    X (numpy array): Input data
    Y (numpy array): True labels
    n_h1 (int): Number of neurons in the first hidden layer
    n_h2 (int): Number of neurons in the second hidden layer
    learning_rate (float): Learning rate
    iterations (int): Number of training iterations

    Returns:
    dict: Trained parameters
    list: Cost over iterations
    """
    n_x = X.shape[0]
    n_y = 1
    num_iterations = iterations
    parameters = initialize_parameters(n_x, n_h1, n_h2, n_y)
    costs = []
    for i in range(num_iterations):
        A3, cache = forward_propagation(X, parameters)
        cost = LogLoss_calculation(A3, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)
        costs.append(cost)
    return parameters, costs


# Predict function
def predict(X, parameters):
    """
    Make predictions using the trained model

    Parameters:
    X (numpy array): Input data
    parameters (dict): Trained parameters

    Returns:
    numpy array: Predictions
    """
    A3, _ = forward_propagation(X, parameters)
    predictions = (A3 > 0.5).astype(int)
    return predictions


# Accuracy function
def compute_accuracy(predictions, Y):
    """
    Compute prediction accuracy

    Parameters:
    predictions (numpy array): Predicted labels
    Y (numpy array): True labels

    Returns:
    float: Prediction accuracy
    """
    return np.mean(predictions == Y)


# Load and preprocess data
url = 'https://github.com/rosenfa/nn/blob/master/pima-indians-diabetes.csv?raw=true'
df = pd.read_csv(url, header=0)
X = df.drop(['Outcome'], axis=1).values.T
Y = np.array(df['Outcome'])
X = (X - np.mean(X)) / np.std(X)
X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.2, random_state=42)
X_train, X_test = X_train.T, X_test.T

# Logistic Regression Baseline
logistic_model = LogisticRegression()
logistic_model.fit(X_train.T, Y_train.T.ravel())
logistic_train_acc = logistic_model.score(X_train.T, Y_train.T.ravel())
logistic_test_acc = logistic_model.score(X_test.T, Y_test.T.ravel())

# Parameters for neural network
learning_rate = 1
node = 3
results = []

for iterations in range(500, 2500, 500):
    print(f"{Fore.GREEN}{Style.BRIGHT}num iterations : {iterations}{Style.RESET_ALL}")
    for n in range(1, 7):
        print(f"{Fore.RED}{Style.BRIGHT}num of nodes: {n}{Style.RESET_ALL}")
        parameters, costs = nn_model(X_train, Y_train, 3, n_h2=n, learning_rate=learning_rate, iterations=iterations)
        train_predictions = predict(X_train, parameters)
        test_predictions = predict(X_test, parameters)
        train_accuracy = compute_accuracy(train_predictions, Y_train)
        test_accuracy = compute_accuracy(test_predictions, Y_test)
        print(
            f"tanh with 3 Nodes: Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        results.append(
            {"iterations": iterations, "nodes": n, "train_accuracy": train_accuracy, "test_accuracy": test_accuracy})

results_df = pd.DataFrame(results)
pivot_train = results_df.pivot(index='nodes', columns='iterations', values='train_accuracy')
pivot_test = results_df.pivot(index='nodes', columns='iterations', values='test_accuracy')

# Combine train and test into a single DataFrame
combined_df = pivot_train.add_suffix('_train').join(pivot_test.add_suffix('_test'))

# Save to CSV
combined_df.to_csv("combined_results.csv")


plt.figure(figsize=(10, 6))

nodes = np.array(pivot_train.index)
y = np.max(pivot_train.values, axis=1)  # Takes the best iteration for each node

#   Location of the best iteration of each vertex to take the corresponding test
max_indices = np.argmax(pivot_train.values, axis=1)
z = pivot_test.values[np.arange(len(max_indices)), max_indices]

plt.plot(nodes, y, marker='o')
plt.plot(nodes, z, marker='x')
plt.xlabel('Nodes')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy for 2 layers')
plt.grid(True)
plt.show()
