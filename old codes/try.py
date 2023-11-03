import numpy as np
import matplotlib.pyplot as plt


class FCLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(
            input_size + output_size
        )
        self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # bias_error = output_error

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class ActivationLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_error, learning_rate):
        return output_error * self.activation_prime(self.input)


# bonus
class FlattenLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, input):
        return np.reshape(input, (1, -1))

    def backward(self, output_error, learning_rate):
        return np.reshape(output_error, self.input_shape)


# bonus
class SoftmaxLayer:
    def __init__(self, input_size):
        self.input_size = input_size

    def forward(self, input):
        self.input = input
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.zeros(output_error.shape)
        out = np.tile(self.output.T, self.input_size)
        return self.output * np.dot(output_error, np.identity(self.input_size) - out)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(x, 0)


def relu_prime(x):
    return np.array(x >= 0).astype("int")


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_pred.size


def sse(y_true, y_pred):
    return 0.5 * np.sum(np.power(y_true - y_pred, 2))


def sse_prime(y_true, y_pred):
    return y_pred - y_true


from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

dataset = np.genfromtxt("datasets/mnist_train.csv", delimiter=",")
X = dataset[:, 1:]  # extract feature columns
Y = dataset[:, 0]


x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0
)

x_train = x_train.astype("float32")
y_train = to_categorical(y_train)


x_test = x_test.astype("float32")
y_test = to_categorical(y_test)


network = [
    FlattenLayer(input_shape=(28, 28)),
    FCLayer(28 * 28, 128),
    ActivationLayer(relu, relu_prime),
    FCLayer(128, 10),
    SoftmaxLayer(10),
]

epochs = 40
learning_rate = 0.01
error_values = []


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def calculate_accuracy(network, x_test, y_test):
    correct = 0
    total = len(x_test)

    for x, y_true in zip(x_test, y_test):
        y_pred = predict(network, x)
        if np.argmax(y_pred) == np.argmax(y_true):
            correct += 1

    accuracy = correct / total
    return accuracy


# Initialize an empty list to store accuracy values for each epoch
accuracy_values = []

# training
for epoch in range(epochs):
    error = 0
    for x, y_true in zip(x_train, y_train):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # error (display purpose only)
        error += mse(y_true, output)

        # backward
        output_error = mse_prime(y_true, output)
        for layer in reversed(network):
            output_error = layer.backward(output_error, learning_rate)

    error /= len(x_train)
    print("%d/%d, error=%f" % (epoch + 1, epochs, error))
    error_values.append(error)
    # accuracy = calculate_accuracy(network, x_train, y_train)

    accuracy = np.all(y_true == output, axis=1).mean()
    print("acc=%f" % (accuracy))
    accuracy_values.append(accuracy)


plt.figure()
plt.plot(range(1, epochs + 1), accuracy_values, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy vs. Number of Epochs")
plt.grid()
plt.show()


plt.figure()
plt.plot(range(1, epochs + 1), error_values, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Error (MSE)")
plt.title("Error vs. Number of Epochs")
plt.grid()
plt.show()
