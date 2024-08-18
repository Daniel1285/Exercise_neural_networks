import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from colorama import Fore, Style


# Load dataset
boston = fetch_california_housing()
print("Shape of Dataset : ", boston.data.shape)
print("Feature names : ", boston.feature_names)
print("Target Values : ", boston.target[:20])

X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = boston.target
# X = X[:1000]
# Y = Y[:1000]

MSE_points = []
num_iterations = 300
alpha = 0.05

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=5)
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

def error_function(y_actual, y_predicted):
    error = np.mean((y_actual - y_predicted) ** 2)
    return error

def y_predicted(theta, x):
    return np.dot(x, theta)

def gradient_descent(y_actual, y_pred, x):
    error = y_pred - y_actual
    grad = np.dot(x.T, error) / len(y_actual)
    return grad

def weights(x_train, y_train, num_iterations, alpha):
    no_of_rows, no_of_columns = x_train.shape
    new_x_train = np.ones((no_of_rows, no_of_columns + 1))
    new_x_train[:, :-1] = x_train
    theta = np.zeros(no_of_columns + 1)
    for i in range(num_iterations):
        y_pred = y_predicted(theta, new_x_train)
        error = error_function(y_train, y_pred)
        MSE_points.append(error)
        grad = gradient_descent(y_train, y_pred, new_x_train)
        theta = theta - alpha * grad
    return theta

# Train model from scratch
thetas = weights(X_train_standardized, Y_train, num_iterations, alpha)
new_X_test_standardized = np.ones((X_test_standardized.shape[0], X_test_standardized.shape[1] + 1))
new_X_test_standardized[:, :-1] = X_test_standardized
Y_pred = y_predicted(thetas, new_X_test_standardized)
custom_mse = mean_squared_error(Y_test, Y_pred)

# Plot cost function
plt.title('Cost Function J', size=30)
plt.xlabel('No. of iterations', size=20)
plt.ylabel('Cost', size=20)
plt.plot(MSE_points)
plt.show()

# Predictions and comparison
pred_df = pd.DataFrame({'Actual Value': Y_test, 'Predicted Values': Y_pred})
print(pred_df.head(10))

print(thetas)

# Using sklearn's Linear Regression for comparison
lm = LinearRegression()
lm.fit(X_train, Y_train)
y_pred_from_sklearn = lm.predict(X_test)
sklearn_mse = mean_squared_error(Y_test, y_pred_from_sklearn)

print(f'{Fore.GREEN}{Style.BRIGHT}Linear Regression (SKLEARN)', str(sklearn_mse))
print('Linear Regression (From Scratch)', str(custom_mse))
