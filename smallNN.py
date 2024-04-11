import numpy as np

# Define the sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Input data for training
# Each row is a training example, each column is a feature.
# There are 4 training examples and 2 features
X = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])
Y = np.array([[0, 1, 1, 0]])

input_size = X.shape[1] # Number of features
output_size = 1
hidden_size = 4
lr = 0.1
epochs = 100000

np.random.seed(1)
# Initialize weights
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size)) # 2 rows, 4 columns
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
bias_hidden = np.random.uniform(size=(1, hidden_size)) # 1 row, 4 columns
bias_output = np.random.uniform(size=(1, output_size))

for epoch in range(epochs):
    #forward pass
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_activations = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_activations, weights_hidden_output) + bias_output
    predicted_ouput = sigmoid(output_layer_input)

    #backward pass
    # Calculate Error
    error = Y.T - predicted_ouput
    d_predicted_ouput = error * sigmoid_derivative(output_layer_input) #4 rows, 1 column
    error_hidden_layer = d_predicted_ouput.dot(weights_hidden_output.T) #4 rows, 4 columns
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_input) #

    weights_hidden_output += lr * hidden_layer_activations.T.dot(d_predicted_ouput)
    bias_output += lr * np.sum(d_predicted_ouput, axis=0, keepdims=True)

    weights_input_hidden += lr * X.T.dot(d_hidden_layer)
    bias_hidden += lr * np.sum(d_hidden_layer, axis=0, keepdims=True)

print(predicted_ouput)