import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    exponent = np.exp(x)
    return exponent / exponent.sum(axis=1, keepdims=True)


def loss(y_pred, y):
    return ((-np.log(y_pred)) * y).sum(axis=1).mean()


def accuracy(y_pred, y):
    return np.all(y_pred == y, axis=1).mean()


def sigmoid_prime(h):
    return h * (1 - h)


def to_categorical(x):
    categorical = np.zeros((x.shape[0], x.shape[1]))
    categorical[np.arange(x.shape[0]), x.argmax(axis=1)] = 1
    return categorical


def init_weights(layer_sizes):
    weights = []
    for i in range(layer_sizes.shape[0] - 1):
        weights.append(
            np.random.uniform(-1, 1, size=[layer_sizes[i], layer_sizes[i + 1]])
        )
    return np.asarray(weights, dtype=object)


def feed_forward(batch, weights):
    h = [batch]
    for i, weight in enumerate(weights):
        h_l = sigmoid(h[i].dot(weight))
        h.append(h_l)
    out = softmax(h[-1])
    return h, out


def back_prop(h, out, batch_y, weights, lr, batch_size):
    delta_t = (out - batch_y) * sigmoid_prime(h[-1])
    for i in range(1, len(weights) + 1):
        weights[-i] -= lr * (h[-i - 1].T.dot(delta_t)) / batch_size
        delta_t = sigmoid_prime(h[-i - 1]) * (delta_t.dot(weights[-i].T))


def plot_graphs(train_loss, train_acc, val_loss, val_acc):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(train_loss, label="Train loss")
    ax[0].plot(val_loss, label="Val loss")
    ax[0].legend()
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid()

    ax[1].plot(train_acc, label="Train acc")
    ax[1].plot(val_acc, label="Val acc")
    ax[1].legend()
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid()
    plt.show()


def train_MLP(
    X,
    Y,
    X_val,
    Y_val,
    num_hidden=1,
    neurons_in_hidden=64,
    batch_size=8,
    epochs=25,
    lr=1.0,
):
    n_samples, n_features = X.shape
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    Y = np.squeeze(np.eye(10)[Y.astype(np.int64).reshape(-1)])
    print(Y)
    X_val = np.concatenate((X_val, np.ones((X_val.shape[0], 1))), axis=1)
    Y_val = np.squeeze(np.eye(10)[Y_val.astype(np.int64).reshape(-1)])
    layer_sizes = np.array(
        [n_features + 1] + [neurons_in_hidden] * num_hidden + [Y.shape[1]]
    )

    weights = init_weights(layer_sizes)
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(epochs):
        shuffle = np.random.permutation(n_samples)
        train_loss_sum = 0
        train_acc_sum = 0
        X_batches = np.array_split(X[shuffle], n_samples // batch_size)
        Y_batches = np.array_split(Y[shuffle], n_samples // batch_size)

        for batch_x, batch_y in zip(X_batches, Y_batches):
            h, out = feed_forward(batch_x, weights)
            train_loss_sum += loss(out, batch_y)
            train_acc_sum += accuracy(to_categorical(out), batch_y)
            back_prop(h, out, batch_y, weights, lr, batch_size)

        train_loss_avg = train_loss_sum / len(X_batches)
        train_acc_avg = train_acc_sum / len(X_batches)
        train_loss.append(train_loss_avg)
        train_acc.append(train_acc_avg)

        _, val_out = feed_forward(X_val, weights)
        val_loss_avg = loss(val_out, Y_val)
        val_acc_avg = accuracy(to_categorical(val_out), Y_val)
        val_loss.append(val_loss_avg)
        val_acc.append(val_acc_avg)

        print(
            f"Epoch {epoch+1}: train_loss = {train_loss_avg:.3f} | train_acc = {train_acc_avg:.3f} | val_loss = {val_loss_avg:.3f} | val_acc = {val_acc_avg:.3f}"
        )

    plot_graphs(train_loss, train_acc, val_loss, val_acc)
    return weights


# START HERE
dataset = np.genfromtxt("mnist_train.csv", delimiter=",")
X = dataset[:, 1:]  # extract feature columns
Y = dataset[:, 0]  # extract class column

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# # normalize data between 0 to 1
X_train = X_train / X_train.max()
X_test = X_test / X_test.max()


weights = train_MLP(
    X_train,
    Y_train,
    X_test,
    Y_test,
    num_hidden=1,
    neurons_in_hidden=32,
    batch_size=4,
    epochs=20,
    lr=1,
)

# print(f"W* = {weights}")
