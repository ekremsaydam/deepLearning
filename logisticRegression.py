import warnings

# from subprocess import check_output
# python -m pip install matplotlib
import matplotlib.pyplot as plt
# python -m pip install numpy
import numpy as np  # linear algebra
# python -m pip install pandas
# import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
# python -m pip install scikit-learn
from sklearn.model_selection import train_test_split

# filter warnings
warnings.filterwarnings('ignore')

# load data set
x_l = np.load('./data/Sign-language-digits-dataset/X.npy')
Y_l = np.load('./data/Sign-language-digits-dataset/Y.npy')
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')

# Join a sequence of arrays along an row axis.
# from 0 to 204 is zero sign and from 205 to 410 is one sign
X = np.concatenate((x_l[204:409], x_l[822:1027]), axis=0)
z_zero = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z_zero, o), axis=0).reshape(X.shape[0], 1)
print("X shape: ", X.shape)
print("Y shape: ", Y.shape)

# Then lets create x_train, y_train, x_test, y_test arrays

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

X_train_flatten = X_train.reshape(
    number_of_train, X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(
    number_of_test, X_test.shape[1]*X_test.shape[2])
print("X train flatten", X_train_flatten.shape)
print("X test flatten", X_test_flatten.shape)

x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)


def dummy(parameter):
    dummy_parameter = parameter + 5
    return dummy_parameter


result = dummy(3)     # result = 8


def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w, b

# calculation of z
# z = np.dot(w.T,x_train)+b


def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


y_head_zero = sigmoid(0)
print(f'y_head {y_head_zero}')


# Forward propagation steps:
# find z = w.T*x+b
# y_head = sigmoid(z)
# loss(error) = loss(y,y_head)
# cost = sum(loss)
def forward_propagation(w, b, x_train, y_train):
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)  # probabilistic 0-1
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    # x_train.shape[1]  is for scaling
    cost = (np.sum(loss))/x_train.shape[1]
    return cost

# In backward propagation we will use y_head
# that found in forward progation
# Therefore instead of writing
# backward propagation method, lets combine forward propagation
# and backward propagation


def forward_backward_propagation(w, b, x_train, y_train):
    # forward propagation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    # x_train.shape[1]  is for scaling
    cost = (np.sum(loss))/x_train.shape[1]
    # backward propagation
    # x_train.shape[1]  is for scaling
    derivative_weight = (
        np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1]
    # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight,
                 "derivative_bias": derivative_bias}
    return cost, gradients

# Updating(learning) parameters


def update(w, b, x_train, y_train, learning_rate, number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print(f'Cost after iteration {i}: {cost:f}')
    # we update(learn) parameters weights and bias
    parameters = {"weight": w, "bias": b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    if 'gradients' in locals():
        return parameters, gradients, cost_list
    else:
        return parameters, None, cost_list


# parameters, gradients,
# cost_list = update(w, b, x_train, y_train,
# learning_rate = 0.009,number_of_iterarion = 200)

# prediction


def predict(w, b, x_test):
    # x_test is a input for forward propagation
    z = np.dot(w.T, x_test)+b
    y_head = sigmoid(z)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(y_head.shape[1]):
        if y_head[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction
# predict(parameters["weight"],parameters["bias"],x_test)


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate,  num_iterations):
    # initialize
    dimension = x_train.shape[0]  # that is 4096
    w, b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(
        w, b, x_train, y_train, learning_rate, num_iterations)

    y_prediction_test = predict(
        parameters["weight"], parameters['bias'], x_test)
    y_prediction_train = predict(
        parameters["weight"], parameters["bias"], x_train)

    # Print train/test Errors
    _train_accuracy = 100 - np.mean(np.abs(y_prediction_train - y_train)) * 100
    _test_accuracy = 100 - np.mean(np.abs(y_prediction_test - y_test)) * 100
    print(f'train accuracy: {_train_accuracy:.2f} %')
    print(f'test accuracy: {_test_accuracy:.2f} %')


logistic_regression(x_train, y_train, x_test, y_test,
                    learning_rate=0.01, num_iterations=150)

logreg = linear_model.LogisticRegression(random_state=42, max_iter=150)
train_accuracy = logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)
test_accuracy = logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)
print(f'test accuracy: {train_accuracy:.2f}')
print(f'train accuracy: {test_accuracy:.2f}')
