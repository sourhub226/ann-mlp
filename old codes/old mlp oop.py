import time
import numpy as np
import sys

# np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


class MLP:
    def __init__(self, X, Y, X_val, Y_val, L=1, N_l=64):
        self.X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.Y = np.squeeze(np.eye(3)[Y.astype(np.int).reshape(-1)])
        self.X_val = np.concatenate((X_val, np.ones((X_val.shape[0], 1))), axis=1)
        self.Y_val = np.squeeze(np.eye(3)[Y_val.astype(np.int).reshape(-1)])
        self.L = L
        self.N_l = N_l
        self.n_samples = self.X.shape[0]
        self.layer_sizes = np.array([self.X.shape[1]] + [N_l] * L + [self.Y.shape[1]])
        self.__init_weights()
        self.train_loss = list()
        self.train_acc = list()
        self.val_loss = list()
        self.val_acc = list()
        # self.metrics = [
        #     self.train_loss,
        #     self.train_acc,
        #     self.val_loss,
        #     self.val_acc,
        # ]

    def __sigmoid(self, x):
        # VCompute the sigmoid
        return 1.0 / (1.0 + np.exp(-x))

    def __softmax(self, x):
        # Compute softmax along the rows of the input
        exponent = np.exp(x)
        return exponent / exponent.sum(axis=1, keepdims=True)

    def __loss(self, y_pred, y):
        # Compute the loss along the rows, averaging along the number of samples
        return ((-np.log(y_pred)) * y).sum(axis=1).mean()

    def __accuracy(self, y_pred, y):
        # Compute the accuracy along the rows, averaging along the number of samples
        return np.all(y_pred == y, axis=1).mean()

    def __sigmoid_prime(self, h):
        # Compute the derivative of sigmoid where h=sigmoid(x)
        return h * (1 - h)

    def __to_categorical(self, x):
        # Transform probabilities into categorical predictions row-wise, by simply taking the max probability
        categorical = np.zeros((x.shape[0], self.Y.shape[1]))
        categorical[np.arange(x.shape[0]), x.argmax(axis=1)] = 1
        return categorical

    def __init_weights(self):
        # Initialize the weights of the network given the sizes of the layers
        self.weights = list()
        for i in range(self.layer_sizes.shape[0] - 1):
            self.weights.append(
                np.random.uniform(
                    -1, 1, size=[self.layer_sizes[i], self.layer_sizes[i + 1]]
                )
            )
        self.weights = np.asarray(self.weights)

    def __init_layers(self, batch_size):
        # Initialize and allocate arrays for the hidden layer activations
        self.__h = [np.empty((batch_size, layer)) for layer in self.layer_sizes]

    def __feed_forward(self, batch):
        # Perform a forward pass of `batch` samples (N_samples x N_features)
        h_l = batch
        self.__h[0] = h_l
        for i, weights in enumerate(self.weights):
            h_l = self.__sigmoid(h_l.dot(weights))
            self.__h[i + 1] = h_l
        self.__out = self.__softmax(self.__h[-1])

    def __back_prop(self, batch_y):
        # Update the weights of the network through back-propagation
        delta_t = (self.__out - batch_y) * self.__sigmoid_prime(self.__h[-1])
        for i in range(1, len(self.weights) + 1):
            self.weights[-i] -= (
                self.lr * (self.__h[-i - 1].T.dot(delta_t)) / self.batch_size
            )
            delta_t = self.__sigmoid_prime(self.__h[-i - 1]) * (
                delta_t.dot(self.weights[-i].T)
            )

    def predict(self, X):
        # Generate a categorical, one-hot, prediction given an input X
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.__init_layers(X.shape[0])
        self.__feed_forward(X)
        return self.__to_categorical(self.__out)

    def evaluate(self, X, Y):
        # Evaluate the performance (accuracy) predicting on X with true labels Y
        prediction = self.predict(X)
        return self.__accuracy(prediction, Y)

    def train(self, batch_size=8, epochs=25, lr=1.0):
        # Train the model with a given batch size, epochs, and learning rate. Store and print relevant metrics.
        self.lr = lr
        self.batch_size = batch_size
        for epoch in range(epochs):
            self.__init_layers(self.batch_size)
            shuffle = np.random.permutation(self.n_samples)
            train_loss = 0
            train_acc = 0
            X_batches = np.array_split(
                self.X[shuffle], self.n_samples / self.batch_size
            )
            Y_batches = np.array_split(
                self.Y[shuffle], self.n_samples / self.batch_size
            )
            for batch_x, batch_y in zip(X_batches, Y_batches):
                self.__feed_forward(batch_x)
                train_loss += self.__loss(self.__out, batch_y)
                train_acc += self.__accuracy(self.__to_categorical(self.__out), batch_y)
                self.__back_prop(batch_y)

            train_loss = train_loss / len(X_batches)
            train_acc = train_acc / len(X_batches)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

            self.__init_layers(self.X_val.shape[0])
            self.__feed_forward(self.X_val)
            val_loss = self.__loss(self.__out, self.Y_val)
            val_acc = self.__accuracy(self.__to_categorical(self.__out), self.Y_val)
            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)

            print(
                f"Epoch {epoch+1}: train_loss = {train_loss.round(3)} | train_acc = {train_acc.round(3)} | val_loss = {val_loss.round(3)} | val_acc = {val_acc.round(3)}"
            )


# START HERE
dataset = np.genfromtxt("iris.csv", delimiter=",")
X = dataset[:, 1:]  # extract feature columns
Y = dataset[:, 0]  # extract class column

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# normalize data between 0 to 1
# X_train = X_train / X_train.max()
# X_test = X_test / X_test.max()

# one-hot encode Y_train and Y_test set labels into binary vectors
# 0 becomes [1 0 0 0 0 0 0 0 0]
# 1 becomes [0 1 0 0 0 0 0 0 0]
# ...
# 9 becomes [0 0 0 0 0 0 0 0 0 1]


model = MLP(X_train, Y_train, X_test, Y_test, L=1, N_l=64)
model.train(batch_size=8, epochs=20, lr=1)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(model.train_loss, label="Train loss")
ax[0].plot(model.val_loss, label="Val loss")
ax[0].legend()
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].grid()

ax[1].plot(model.train_acc, label="Train acc")
ax[1].plot(model.val_acc, label="Val acc")
ax[1].legend()
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].grid()
plt.show()
