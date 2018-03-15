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
    '''Unit step function.'''
    return 1 if x >= 0 else 0


def predict(x):
    '''Predict the label for a given x.'''
    pred = np.array([])

    for k in range(K):
        w = W[k]
        b = B[k]
        pred = np.append(pred, step(x.dot(w) - b))

    return pred


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
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

x_dim = data.shape[1]
K = labels.shape[1]
# Initialise weights for the k-neuron network, where k in this case is 10.
W = np.random.normal(0, 1, (K, x_dim))
# Initialise bias values for the k-neuron network.
B = np.zeros(K)
# Initialise training parameters
epochs = 10
alpha = 0.2
num_tests = len(X_train)  # the number of rows in X_train

# Start training.
for epoch in range(epochs):
    accuracies = np.array([])
    error_rate = 0

    # Loop through test data.
    for x, y in zip(X_train, y_train):
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
            # Apply weight and bias changes.
            W[k] = w + alpha * dw
            B[k] = b + alpha * db

        # Predict the label for the current x.
        yhat = predict(x)
        # Calculate the error.
        e = np.sum(np.abs(y - yhat))

        if e != 0:
            error_rate += 1

    # Report epoch accuracy.
    error_rate = error_rate / num_tests
    accuracy = 1 - error_rate
    print('Epoch #' + str(1 + epoch) + ' accuracy: ' + str(accuracy))

    if accuracy > 0.99:
        break

# Make predictions on test data.
error_rate = 0

for i in range(len(X_test)):
    x = X_test[i]
    y = y_test[i]
    yhat = predict(x)

    e = np.sum(np.abs(y - yhat))

    if e != 0:
        error_rate += 1

# Report test accuracy.
error_rate /= num_tests
accuracy = 1 - error_rate
print('Test accuracy: ' + str(accuracy))
