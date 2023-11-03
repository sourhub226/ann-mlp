import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from activations import *
from layers import *
import seaborn as sns


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_pred.size


dataset = np.genfromtxt("datasets/iris.csv", delimiter=",")
X = dataset[:, 1:]  # extract feature columns
Y = dataset[:, 0]  # extract target column
X = X.astype("float32")
Y = to_categorical(Y)
# print(X.shape)
# print(Y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0
)


epochs = 40
learning_rate = 0.01
neurons_in_hl = 128
error_values = []


network = [
    FlattenLayer(input_shape=(X.shape[1] * 1)),  # ip layer
    DenseLayer(X.shape[1], neurons_in_hl),
    ActivationLayer(relu, relu_prime),
    DenseLayer(neurons_in_hl, Y.shape[1]),  # op layer
    SoftmaxLayer(Y.shape[1]),
]


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


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
    print(f"Epoch {epoch + 1}/{epochs} | error = {error:.6f}")
    error_values.append(error)


train_acc = np.mean(
    [np.argmax(y) == np.argmax(predict(network, x)) for x, y in zip(x_train, y_train)]
)
print(f"\nTraining accuracy: {train_acc*100:.2f}%")
test_acc = np.mean(
    [np.argmax(y) == np.argmax(predict(network, x)) for x, y in zip(x_test, y_test)]
)
print(f"Testing accuracy: {test_acc*100:.2f}%")


plt.figure()
plt.plot(range(1, epochs + 1), error_values, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Error (MSE)")
plt.title("Error vs. Number of Epochs")
plt.grid()
plt.show()


# Initialize variables to store the predicted and true labels
predicted_class = []
true_class = []


for test, true in zip(x_test, y_test):
    pred = predict(network, test)[0]
    idx = np.argmax(pred)
    idx_true = np.argmax(true)
    predicted_class.append(idx)
    true_class.append(idx_true)

# Create the confusion matrix
confusion = confusion_matrix(true_class, predicted_class)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(confusion, annot=True)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
