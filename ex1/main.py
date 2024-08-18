from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

boston = fetch_california_housing()

print("Shape of Dataset : ", boston.data.shape)
print("Feature names : ", boston.feature_names)
print("Target Values : ", boston.target[:20])

X = pd.DataFrame(boston.data, columns = boston.feature_names)
Y = boston.target


MSE_points = []
num_iterations = 300
alpha = 0.05


print(X.head())
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print(Y_train)


scaler = preprocessing.StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.fit_transform(X_test)



def error_function(y_actual,y_predicted):
    error = 0
    for i in range(0,len(y_actual)): #This is the value of n
        error =  error + pow((y_actual[i] - y_predicted[i]),2)#This is Y-YP
    #return error/(2*len(y_actual))
    return error/(len(y_actual))


def gradient_descent(y_actual,y_pred,x):
    grad = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        for j in range(0,len(y_actual)):
            grad[i] = grad[i] + (y_pred[j] - y_actual[j])*x[j][i]
            #grad[i] = grad[i] - (y_actual[j]-y_pred[j])*x[j][i] #  other way to write this

    return grad/len(y_actual)


def y_predicted(theta,x):
    y_pred = np.zeros(len(x))
    for i in range(0,len(x)):
        for j in range(0,len(theta)-1):
            y_pred[i] = y_pred[i]+(theta[j]*x[i][j])#for the weights
        y_pred[i]+=theta[-1] #for the bias
    return y_pred


def regression_test(x_test,theta):
    row = x_test.shape[0]
    column = x_test.shape[1]
    new_x_test = np.ones((row,column+1))
    new_x_test[:,0:column] = x_test
    y_pred = y_predicted(theta,new_x_test)
    return(y_pred)


def weights(x_train,y_train,num_iterations,alpha):
    no_of_rows = x_train.shape[0]
    no_of_columns = x_train.shape[1]
    new_x_train = np.ones((no_of_rows,no_of_columns+1))
    new_x_train[:,0:no_of_columns] = x_train
    theta = np.zeros(no_of_columns)
    theta = np.append(theta,1)
    for i in range(0,num_iterations):
        y_pred = y_predicted(theta,new_x_train)
        error = error_function(y_train,y_pred)
        #print("mean square error: ",error,"after",i,"th iteration")
        MSE_points.append(error)
        grad = gradient_descent(y_train,y_pred,new_x_train)
        theta = theta - alpha*grad
    return theta


thetas = weights(X_train_standardized,Y_train,num_iterations,alpha)
Y_pred = regression_test(X_test_standardized,thetas)
error_function(Y_test,Y_pred)



plt.title('Cost Function J', size = 30)
plt.xlabel('No. of iterations', size=20)
plt.ylabel('Cost', size=20)
plt.plot(MSE_points)
plt.show()


pred_df = pd.DataFrame(
    {
        'Actual Value' : Y_test,
     'Predicted Values' : Y_pred,
    }
)
print(pred_df.head(10))
print(thetas)

MSE_custom_LR_Model = mean_squared_error(Y_test, Y_pred)

lm = LinearRegression()
lm.fit(X_train, Y_train)
y_pred_from_sklearn = lm.predict(X_test)

MSE_sklearn_LR_Model = mean_squared_error(Y_test, y_pred_from_sklearn)

print('Linear Regression (SKLEARN)', str(MSE_sklearn_LR_Model))
print('Linear Regression (From Scratch)', str(MSE_custom_LR_Model))






