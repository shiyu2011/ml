import numpy as np

X = np.array([1, 2, 3, 4, 5])

Y = np.array([1, 4, 9, 16, 25])

def analytical_solution(X, Y, m):
    X = [X**i for i in range(m + 1)]
    X = np.array(X).T
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

def polynomial_regression(X, Y, m):
    X = [X**i for i in range(m + 1)]
    X = np.array(X).T
    beta = np.random.rand(m + 1)
    lr = 0.00001
    for i in range(100000):
        beta -= lr * X.T.dot(X.dot(beta)-Y)
    return beta

#test
print(analytical_solution(X, Y, 2))
print(polynomial_regression(X, Y, 2))
    
