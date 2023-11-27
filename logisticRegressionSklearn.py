# https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe

# python -m pip install --upgrade pip
# python -m pip install numpy
# python -m pip install matplotlib
# python -m pip install scikit-learn

import warnings

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
from sklearn import linear_model
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


logreg = linear_model.LogisticRegression(random_state=42, max_iter=150)
print(
    f'test accuracy: {(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T))}')

print(
    f'train accuracy: {(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T))}')
