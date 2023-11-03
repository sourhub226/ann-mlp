import time
import numpy as np
import sys

# np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


dataset = np.genfromtxt("mnist_mini.csv", delimiter=",")
X = dataset[:, 1:]  # extract feature columns
Y = dataset[:, 0]  # extract class column

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print(Y_train)
# print(f"The shape of the training set is: {X_train.shape[0]} x {X_train.shape[1]}")
# print(f"The shape of the test set is: {X_test.shape[0]} x {X_test.shape[1]}")

# # rescale data between 0 to 1
X_train = X_train / X_train.max()
X_test = X_test / X_test.max()


# one-hot encode Y_train and Y_test set labels into binary vectors
# 0 becomes [1 0 0 0 0 0 0 0 0]
# 1 becomes [0 1 0 0 0 0 0 0 0]
# ...
# 9 becomes [0 0 0 0 0 0 0 0 0 1]
y_train = np.zeros((Y_train.size, int(Y_train.max()) + 1))
y_train[np.arange(Y_train.size), Y_train.astype(np.int)] = 1.0

y_test = np.zeros((Y_test.size, int(Y_test.max()) + 1))
y_test[np.arange(Y_test.size), Y_test.astype(np.int)] = 1.0

print(y_train)


# Assuming all hidden layers are treated with sigmoid activaiton fn.
# and the final layer is processed with a softmax activation fn.


def sigmoid(x):
    # returns sigmoid applied to `x` element-wise
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    # x : (N x dim) array with N samples. dim=10 for MNIST classification.
    # returns softmax applied to `x` along the first axis.
    exponent = np.exp(x)  # only compute the exponent once
    return exponent / exponent.sum(axis=1, keepdims=True)


def init_layers(batch_size, layer_sizes):
    # batch_size : Number of samples to concurrently feed through the network.
    # layer_sizes : Array of length `N_hidden`. Each entry is the number of neurons in each layer.
    # returns List of empty arrays used to hold hidden layer outputs.

    hidden_layers = [np.empty((batch_size, layer_size)) for layer_size in layer_sizes]
    return hidden_layers


def init_weights(layer_sizes):
    # returns randomly initalized weight matricies based on the layer sizes with numerical values obtained from a normal distribution with mean=0 and standard deviation=1.
    weights = list()
    for i in range(layer_sizes.shape[0] - 1):
        weights.append(
            np.random.uniform(-1, 1, size=[layer_sizes[i], layer_sizes[i + 1]])
        )
    # weights = np.asarray(self.weights)
    return weights


# using stochastic gradient descent (SGD) with mini batches
def feed_forward(batch, hidden_layers, weights):
    # Perform a forward pass of the neural network.
    # batch :(batch_size x dim) matrix of inputs
    # hidden_layers : List of hidden layer outputs
    # weights : Array of weight matricies

    # returns
    # output : Forward pass output of the MLP
    # hidden_layers : List of hidden layer outputs, populated from the forward pass.

    h_l = batch
    hidden_layers[0] = h_l
    for i, weight in enumerate(weights):
        h_l = sigmoid(h_l.dot(weight))
        hidden_layers[i + 1] = h_l
    output = softmax(hidden_layers[-1])
    return output, hidden_layers


def sigmoid_prime(h):
    # returns derivative of sigmoid, based on value of sigmoid.
    return h * (1 - h)


def back_prop(output, batch_y, hidden_layers, weights, batch_size, lr):
    # output : Forward pass output of the MLP
    # batch_y : True labels for the samples in the batch
    # lr : Learning rate for SGD
    # batch_size : Size of a training mini-batch
    # returns array of weight matricies, updated from the backpropagation.

    delta_t = (output - batch_y) * sigmoid_prime(hidden_layers[-1])
    for i in range(1, len(weights) + 1):
        weights[-i] -= lr * (hidden_layers[-i - 1].T.dot(delta_t)) / batch_size
        delta_t = sigmoid_prime(hidden_layers[-i - 1]) * (delta_t.dot(weights[-i].T))
    return weights


def loss(y_pred, y):
    # Compute the loss along the rows, averaging along the number of samples
    return ((-np.log(y_pred)) * y).sum(axis=1).mean()


def accuracy(y_pred, y):
    # Compute the accuracy along the rows, averaging along the number of samples
    return np.all(y_pred == y, axis=1).mean()


def to_categorical(x):
    # Transform probabilities into categorical predictions row-wise, by simply taking the max probability
    categorical = np.zeros((x.shape[0], Y.shape[1]))
    categorical[np.arange(x.shape[0]), x.argmax(axis=1)] = 1
    return categorical


def predict(X):
    # Generate a categorical, one-hot, prediction given an input X
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    init_layers(X.shape[0])
    feed_forward(X)
    return to_categorical(output)


def evaluate(X, Y):
    # Evaluate the performance (accuracy) predicting on X with true labels Y
    prediction = predict(X)
    return accuracy(prediction, Y)


def train(X, Y, layer_sizes, batch_size=8, epochs=25, lr=1.0):
    """
    Train the MLP.

    Parameters
    ----------
    X : array_like
        Forward pass output of the MLP
    Y : array_like
        True labels for the samples in the batch
    layer_sizes :
        Array of length `N_l`. Each entry is the number of neurons in each layer
    batch_size : int
        Size of a training mini-batch
    epochs : int
        Number of iterations to train for
    lr : float
        Learning rate for SGD

    Returns
    -------
    weights : array_like
        Array of weight matricies, updated from the backpropagation.

    """
    n_samples = X.shape[0]

    hidden_layers = init_layers(batch_size, layer_sizes)
    weights = init_weights(layer_sizes)
    for epoch in range(epochs):
        start = time.time()

        shuffle = np.random.permutation(n_samples)
        X_batches = np.array_split(X[shuffle], n_samples / batch_size)
        Y_batches = np.array_split(Y[shuffle], n_samples / batch_size)

        train_loss = 0
        train_acc = 0

        for batch_x, batch_y in zip(X_batches, Y_batches):
            output, hidden_layers = feed_forward(batch_x, hidden_layers, weights)
            train_loss += loss(output, batch_y)
            train_acc += accuracy(to_categorical(output), batch_y)
            weights = back_prop(output, batch_y, hidden_layers, weights, batch_size, lr)

        train_loss = train_loss / len(X_batches)
        train_acc = train_acc / len(X_batches)

        train_time = round(time.time() - start, 3)

        print(
            f"Epoch {epoch+1}: loss = {train_loss.round(3)} | acc = {train_acc.round(3)} | train_time = {train_time} | tot_time = {tot_time}"
        )

    return weights
