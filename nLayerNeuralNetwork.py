# Evaluating the ANN
# https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe

# python -m pip install --upgrade pip
# python -m pip install numpy
# python -m pip install matplotlib
# python -m pip install keras
# python -m pip install tensorflow
# python -m pip install scikit-learn
# pip install scikeras[tensorflow]

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import tensorflow as tf
from keras import activations
# from keras import losses, metrics, optimizers
from keras.initializers import initializers
from keras.layers import Dense  # build our layers library
from keras.models import Sequential  # initialize neural network library
# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score, train_test_split

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


# reshaping
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T


def build_model():
    _model = Sequential()  # initialize neural network

    _model.add(
        Dense(units=8,
              #   kernel_initializer='uniform',
              kernel_initializer=initializers.RandomUniform.__name__,
              activation=activations.relu,
              input_dim=x_train.shape[1]))

    _model.add(
        Dense(units=4,
              kernel_initializer=initializers.RandomUniform.__name__,
              activation=activations.tanh))

    _model.add(
        Dense(units=1,
              kernel_initializer=initializers.RandomUniform.__name__,
              activation=activations.sigmoid))

    _model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        # optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        # loss=losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()])
    # metrics=[metrics.BinaryAccuracy()])
    return _model


classifier = KerasClassifier(model=build_model, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=3)
mean = accuracies.mean()
variance = accuracies.std()
print(f'Accuracy mean: {mean}')
print(f'Accuracy variance: {variance}')
