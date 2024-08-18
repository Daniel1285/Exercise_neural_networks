import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from colorama import Fore, Style


boston_data = fetch_california_housing()
df = pd.DataFrame(data=boston_data['data'])
df.columns = boston_data['feature_names']
df['Price'] = boston_data['target']

corr = df.corr()
corr['Price'].sort_values(ascending=False)

X = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
Y = boston_data.target

scaler = preprocessing.StandardScaler()
X = (X - X.mean()) / X.std()
X = np.c_[np.ones(X.shape[0]), X]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

alpha = 0.05
m = y_train.size
theta = np.random.rand(X.shape[1])

def gradient_descent(x, y, m, theta, alpha):
    cost_list = []  # to record all cost values to this list
    theta_list = []  # to record all theta_0 and theta_1 values to this list
    prediction_list = []
    run = True
    cost_list.append(1e10)  # we append some large value to the cost list
    i = 0
    while run:
        prediction = np.dot(x, theta)  # predicted y values theta_1*x1+theta_2*x2
        prediction_list.append(prediction)
        error = prediction - y  # compare to y_pred[j] - y_actual[j] in other file
        cost = 1 / (2 * m) * np.dot(error.T, error)  # (1/2m)*sum[(error)^2]
        cost_list.append(cost)
        theta = theta - (alpha * (2 / m) * np.dot(x.T, error))
        # compare to theta = theta - learning_rate*grad in other file
        theta_list.append(theta)
        if np.abs(
                cost_list[i] - cost_list[i + 1]) < 1e-7:
            run = False
        i += 1

    cost_list.pop(0)  # Remove the large number we added in the beginning
    return prediction_list, cost_list, theta_list
prediction_list, cost_list, theta_list = gradient_descent(x_train, y_train, m, theta, alpha)
def y_predicted(theta, x):
    return np.dot(x, theta)
def error_function(y_actual, y_predicted):
    error = 0
    for i in range(0, len(y_actual)):  # This is the value of n
        error = error + pow((y_actual[i] - y_predicted[i]), 2)  # This is Y-YP
    # return error/(2*len(y_actual))
    return error / (len(y_actual))


def regression_test(x_test, theta):
    row = x_test.shape[0]
    column = x_test.shape[1]
    new_x_test = np.ones((row, column + 1))
    new_x_test[:, 0:column] = x_test
    y_pred = y_predicted(theta, new_x_test)
    return (y_pred)


theta = theta_list[-1]
pred_df = pd.DataFrame(
    {
        'Actual Value': Y,
        'Predicted Values': np.dot(X, theta)
    }
)
print(pred_df.head(10))

y_pred = y_predicted(theta, x_test)
MSE_custom_LR_Model = error_function(y_test, y_pred)

lm = LinearRegression()
lm.fit(x_train, y_train)
y_pred_from_sklearn = lm.predict(x_test)
MSE_sklearn_LR_Model = mean_squared_error(y_test, y_pred_from_sklearn)

print(f'\n{Fore.GREEN}{Style.BRIGHT}Linear Regression (From Scratch)', str(MSE_custom_LR_Model))
print('Linear Regression (SKLEARN)', str(MSE_sklearn_LR_Model))
