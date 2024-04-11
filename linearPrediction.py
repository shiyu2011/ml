import numpy as np

def linearPrediction(X, Y):
    X = [X**i for i in range(2)]
    X = np.array(X).T
    Y = np.array(Y)
    return np.linalg.inv(X.T @ X) @ X.T @ Y

data = {}
data['hours'] = np.array([1, 2, 3])
data['score'] = np.array([1, 2, 3])

X = data['hours']
Y = data['score']

print(linearPrediction(X, Y))