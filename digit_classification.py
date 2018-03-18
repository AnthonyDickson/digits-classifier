import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Import datasets, classifiers, and performance metrics
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def plot_digits(data, n=5):
    """ Display some of the digits from the dataset. """
    for i in range(n):
        plot_digit(data[i])


def plot_digit(digit_row):
    """Plots a monochrome heatmap of a digit."""
    sns.heatmap(digit_row.reshape(8, 8), cmap='Greys')
    plt.show()


def step(x):
    """Unit step function."""
    return 1 if x >= 0 else 0


def predict(x):
    """Predict the label for a given x."""
    pred = np.array([])

    for k in range(K):
        w = W[k]
        b = B[k]
        pred = np.append(pred, step(x.dot(w) - b))

    return pred


def validate(X, Y):
    """Tests the model on the given inputs and return a tuple of the error
    rate and accuracy.
    """
    error_rate = 0

    for x, y in zip(X, Y):
        yhat = predict(x)

        e = np.sum(np.abs(y - yhat))
        error_rate += 0 if e == 0 else 1

    # Report test accuracy.
    error_rate /= len(X)

    return error_rate, 1 - error_rate


# Load the digits dataset
digits = datasets.load_digits()
data = digits.data
labels = digits.target

# One-Hot encode the y values.
labels = np.expand_dims(labels, axis=1)
enc = preprocessing.OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray()

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size=0.2,
                                                    random_state=343)

x_dim = data.shape[1]  # Number of parameters we are training on.
K = labels.shape[1]  # Number of classes we are trying to predict.
# Initialise weights and biases.
W = np.random.normal(0, 1, (K, x_dim))
B = np.zeros(K)
# Initialise training parameters
epochs = 50
alpha = 0.4  # Learning rate.
N = len(X_train)  # Number of training examples.
use_batch_training = False

history = {'err': [], 'acc': [], 'val_err': [], 'val_acc': []}

# Start training.
for epoch in range(epochs):
    error_rate = 0  # Epoch error rate over the training set.
    dW = np.zeros((K, x_dim))
    dB = np.zeros(K)

    # Loop through training data.
    for x, y in zip(X_train, y_train):
        # Use current model to predict the label for the current x.
        yhat = predict(x)
        e = np.sum(np.abs(y - yhat))
        error_rate += 0 if e == 0 else 1

        # For each label class in K...
        for k in range(K):
            # Get weights and biases for label class k.
            w = W[k]
            b = B[k]
            # Get the predicted value.
            yhat = step(x.dot(w) - b)
            # Calculate error
            e = y[k] - yhat
            # Calculate weight and bias changes.
            dw = e * x
            db = -e

            if use_batch_training:
                # Accumulate weight and bias changes.
                dW[k] += dw / N
                dB[k] += db / N
            else:
                # Apply weight and bias changes.
                W[k] = W[k] + alpha * dw
                B[k] = B[k] + alpha * db

    if use_batch_training:
        # Apply average weight and bias changes
        for k in range(K):
            W[k] = W[k] + alpha * dW[k]
            B[k] = B[k] + alpha * dB[k]

    # Update error and accuracy metrics.
    error_rate /= N
    accuracy = 1 - error_rate
    history['err'].append(error_rate)
    history['acc'].append(accuracy)

    # Validate using test data and report epoch accuracy.
    val_err, val_acc = validate(X_test, y_test)
    history['val_err'].append(val_err)
    history['val_acc'].append(val_acc)
    print('Epoch # {} acc: {:.3} val_acc: {:.3}'.format(1 + epoch, accuracy, val_acc))

print('Best test accuracy: {:.3}'.format(np.max(history['val_acc'])))

# Plot history of accuracy
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('accuracy (%)')
plt.show()

# Plot history of error
plt.plot(history['err'])
plt.plot(history['val_err'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('error (%)')
plt.show()
