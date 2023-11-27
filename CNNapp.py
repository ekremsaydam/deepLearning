# https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe

# python -m pip install --upgrade pip
# python -m pip install numpy
# python -m pip install matplotlib
# python -m pip install scikit-learn

import warnings

import matplotlib.pyplot as plt
# import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# import warnings
# import os
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from sklearn.model_selection import train_test_split

# filter warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or
# pressing Shift+Enter) will list the files in the input directory

# read train
train = pd.read_csv("./data/digit-recognizer/train.csv")
print(train.shape)
train.head()

# read test
test = pd.read_csv("./data/digit-recognizer/test.csv")
print(test.shape)
test.head()

# put labels into y_train variable
Y_train = train["label"]
# Drop 'label' column
X_train = train.drop(labels=["label"], axis=1)

# visualize number of digits classes
plt.figure(figsize=(15, 7))
g = sns.countplot(Y_train, palette="icefire")
plt.title("Number of digit classes")
Y_train.value_counts()

# # plot some samples
# img = X_train.iloc[0].to_numpy()
# img = img.reshape((28, 28))
# plt.imshow(img, cmap='gray')
# plt.title(str(train.iloc[0, 0]))
# plt.axis("off")
# plt.show()


# # plot some samples
# img = X_train.iloc[3].to_numpy()
# img = img.reshape((28, 28))
# plt.imshow(img, cmap='gray')
# plt.title(str(train.iloc[3, 0]))
# plt.axis("off")
# plt.show()

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
print("x_train shape: ", X_train.shape)
print("test shape: ", test.shape)

# Reshape # 28, 28, 1 => 3D gray scale
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)
print("x_train shape: ", X_train.shape)

print("test shape: ", test.shape)

# Label Encoding

# convert to one-hot-encoding
Y_train = to_categorical(Y_train, num_classes=10)

# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.1, random_state=2)

print("x_train shape", X_train.shape)
print("x_test shape", X_val.shape)
print("y_train shape", Y_train.shape)
print("y_test shape", Y_val.shape)

# Some examples
plt.imshow(X_train[2][:, :, 0], cmap='gray')
plt.show()
