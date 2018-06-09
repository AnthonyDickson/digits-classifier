import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Import datasets, classifiers, and performance metrics
import time
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

plot_handles = []
error_plot_handle = None


def data_plot(x, y=None):
    global plot_handles, error_plot_handle

    num_points, num_attributes = x.shape
    im_height = 8
    im_width = 8

    if not plot_handles:
        plt.close('all')
        figure_handle = plt.figure()
        figure_handle.suptitle('Sample of the Model\'s Predictions')
        plt.ion()
        plt.show()
        n = 0

        for r in range(4):
            for c in range(4):
                if n >= num_points:
                    continue

                im = x[n, :].reshape(im_height, im_width)
                n += 1

                ph = figure_handle.add_subplot(4, 4, n)
                plot_handles.append(ph)

                ph.imshow(im)

                ph.xaxis.set_visible(False)
                ph.yaxis.set_visible(False)

        plt.tight_layout(pad=1.5)
        plt.subplots_adjust(top=0.88)

    if not error_plot_handle:
        error_plot_handle = plt.figure()
        error_plot_handle.add_subplot(1, 1, 1)

    for n in range(len(plot_handles)):
        if np.sum(y[n, :]) != 1:
            class_label = "?"
        else:
            class_label = np.argmax(y[n, :])

        plot_handles[n].set_title(class_label)

    axis = error_plot_handle.get_axes()[0]
    axis.clear()
    axis.set_title('Training and Test Error Rate Over {} Epochs'.format(len(history['err'])))
    axis.plot(history['err'])
    axis.plot(history['val_err'])
    axis.legend(['Train', 'Test'])
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Error Rate')
    axis.set_ylim(0.0, 1.0)
    axis.set_xlim(0)
    plt.pause(0.1)
    time.sleep(0.1)


def predict(x):
    global W, b

    y_pred = np.matmul(x, W.transpose()) - b
    return np.heaviside(y_pred, 0)


def error(y, y_pred):
    e = y - y_pred
    e_abs = np.abs(e)
    errors_by_row = np.sum(e_abs, axis=1)
    error_rate = np.count_nonzero(errors_by_row) / y.shape[0]

    return e, error_rate


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

num_params = data.shape[1]
num_labels = labels.shape[1]  # Number of classes we are trying to predict.
# Initialise weights and biases.
W = np.random.normal(0, 1, (num_labels, num_params))
b = np.zeros(num_labels)
# Initialise training parameters
epochs = 200
alpha = 0.4  # Learning rate.
num_examples = len(X_train)  # Number of training examples.
use_batch_training = False

history = {'err': [], 'val_err': []}

# Start training.
for epoch in range(epochs):
    y_pred = predict(X_train)
    e, error_rate = error(y_train, y_pred)

    dW = np.zeros((num_examples, *W.shape))
    db = np.zeros((num_examples, *b.shape))

    for example in range(num_examples):
        for label in range(num_labels):
            dW[example][label] = e[example, label] * X_train[example]
            db[example][label] = -e[example, label]

    W += alpha * dW.mean(axis=0)
    b += alpha * db.mean(axis=0)

    # Update error and accuracy metrics.
    history['err'].append(error_rate)

    # Validate using test data and report epoch accuracy.
    y_pred = predict(X_test)
    _, val_err = error(y_test, y_pred)

    history['val_err'].append(val_err)
    print('Epoch # {} err: {:.3} val_err: {:.3}'.format(1 + epoch, error_rate, val_err))
    data_plot(X_test, y_pred)

best_accuracy = 1.0 - np.min(history['val_err'])
print('Best test accuracy: {:.3}'.format(best_accuracy))

plt.ioff()
plt.show()
