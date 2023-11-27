import pickle
import socket

from sklearn.datasets import load_digits, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target  # type: ignore


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('127.0.0.1', 9999))

server.listen(1)

while True:
    print('Waiting for connection...')
    client, addr = server.accept()

    try:
        print('Connected')
        data = b''
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            data += chunk

        reveived_object = pickle.loads(data)
        print(f'Received {reveived_object}')

        print(f'Accuracy: {reveived_object.score(X,y)}')
    finally:
        client.close()
