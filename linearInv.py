import numpy as np

X = np.array([1 , 2 , 3 , 4 , 5])
Y = np.array([1 , 2 , 3 , 4 , 5])

def analytical_solution(X , Y, m):
    X = [X**i for i in range(m)]
    X = np.array(X).T
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return beta

def iterative_solution(X , Y , m):
    X = [X**i for i in range(m)]
    X = np.array(X).T
    beta = np.zeros(m)
    for i in range(10):
        beta = beta - 0.01 * X.T.dot(X.dot(beta) - Y)


m = 3
beta = analytical_solution(X , Y , m)
print(beta)

beta = iterative_solution(X , Y , m)
print(beta)

