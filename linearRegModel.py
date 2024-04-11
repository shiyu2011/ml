import numpy as np

def linear_regression(X, y):
    # Calculate the coefficients
    # Implement the formula: (X^T * X)^(-1) * X^T * y
    # Return the coefficients
    A = (np.transpose(X) * X)^(-1) * np.transpose(X) * y 
    return A

# Test your function with a simple dataset
X = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

A = linear_regression(X, y)
print(A)