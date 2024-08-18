import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

url = "https://github.com/rosenfa/nn/blob/master/pima-indians-diabetes.csv?raw=true"
df_pima = pd.read_csv(url, header=0)
print(df_pima.head(10))

X = df_pima.iloc[:, :-1].values  # everything except the target
y = df_pima.iloc[:, -1].values  # the target

# Standardize the predictor variables to have mean 0 and variance 1
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # normalizing the data
X = np.append(np.ones([len(X), 1]), X, 1)  # For the bias

theta = np.zeros(X.shape[1])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(h, y):
    """
    Note: when y = 0 the first half of the equation is 0,
    and when y = 1, the second half of the equation is equal to 0.
    """
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def predict_probs(X, theta):
    return sigmoid(np.dot(X, theta))


def predict(X, theta, threshold=0.5):
    if predict_probs(X, theta) >= threshold:
        return 1
    return 0


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

cost_array = []  # keeping a list of the cost at each iteration to make sure it is constantly decreasing
iterations = 2000  # like the red arrow in slide 37
alpha = 0.01  # also called lr or learning rate
m, n = X_train.shape
for i in range(iterations):
    A = np.zeros(m)  # Initialize the hypothesis vector
    for j in range(m):
        z = 0
        for k in range(n):
            z += X_train[j][k] * theta[k]
        A[j] = sigmoid(z)

    gradient = np.zeros(n)
    for k in range(n):
        grad_sum = 0
        for j in range(m):
            grad_sum += (A[j] - y_train[j]) * X_train[j][k]
        gradient[k] = grad_sum / m

    for k in range(n):
        theta[k] -= alpha * gradient[k]
    cost = loss(A, y_train)
    cost_array.append(cost)


plt.plot(cost_array)
plt.show()

print(theta)
print(y_train[0])
print(X_train[0])

#  test our model on our test data.
correct = 0
for x, y in zip(X_test, y_test):
    p = predict(x, theta)
    if p == y:
        correct += 1

m = len(y_test)
accuracy = correct / m * 100
print("accuracy: {}".format(accuracy), "%")

sk_model = LogisticRegression()
sk_model.fit(X_train, y_train)
accuracy = sk_model.score(X_test, y_test)

print("accuracy = ", accuracy * 100, "%")
print('Coefficients: \n', sk_model.coef_)
print(theta)