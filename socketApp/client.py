import pickle
import socket

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target  # type: ignore

X_train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.5)

model = RandomForestClassifier()
model.fit(X_train, y_train)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client.connect(('127.0.0.1', 9999))

try:
    myobject = {'key1': 'value1', 'key2': 'value2'}
    serialized = pickle.dumps(model)
    # print(serialized)
    client.sendall(serialized)
finally:
    client.close()
