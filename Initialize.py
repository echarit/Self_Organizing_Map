import numpy as np
import matplotlib.pyplot as plot
from scipy.io import loadmat


def read_data():
    """
    A function tailored to read the specific mnist dataset.

    ----------

    :return:
        data
            An m x n numpy array where m is the number of samples and n its dimensionality.
        labels
            An 1D array (no singleton dimensions) of length m containing the data's labels.
        pointers
            A list of size k where k is the number of classes the dataset has.
            The list contains k tuples (a, b) where a and b are indexes
            where the k-th class samples start end end respectively in the data matrix.
            See sort_dataset() to see how pointers is created.
        num_of_classes
            The number of classes the dataset has.
    """
    path = 'mnist-original.mat'
    print('Reading', path)
    dictionary = loadmat(path)
    data = dictionary['data']
    data_labels = dictionary['label']

    data = data.T
    data = data / 255
    data_labels = np.asarray(data_labels, dtype='int32')
    data_labels = np.squeeze(data_labels)
    return data, data_labels


def sort_dataset(dataset, labels):
    """
    Sorts the data set in a way that the first n samples belong to class '0',
    the next k samples to class '1' etc. Use only on an unsorted data set.

    ----------

    :param dataset:
        An m x n numpy array where m is the number of images and n its dimensionality.
    :param labels:
        An 1D array (no singleton dimensions) of length m containing the data's labels.

    :return:
        data
            An m x n numpy array where m is the number of images and n its dimensionality.
        labels
            An 1D array (no singleton dimensions) of length m containing the data's labels.
        indexes
            A list of size k where k is the number of classes the dataset has.
            The list contains k tuples (a, b) where a and b are indexes
            where the k-th class samples start end end respectively in the data matrix.
    """
    classes = int(np.amax(labels)) + 1
    data_dimension = dataset.shape[1]
    numel = dataset.shape[0]
    class_elements = np.zeros((classes,), dtype='int32', order='F')
    counters = np.zeros((classes,), dtype='int32', order='F')
    data_sorted = []

    for i in range(0, classes, 1):  # Creating a list with n matrices, one for each class
        class_elements[i] = np.sum(labels == i)
        data_sorted.append(np.zeros((class_elements[i], data_dimension), dtype='float32', order='F'))
    for i in range(0, numel, 1):  # Adding a sample to the matrix it belongs
        data_sorted[labels[i]][counters[labels[i]]][:] = dataset[i][:]
        counters[labels[i]] += 1
    k = 0
    for i in range(0, classes, 1):
        for j in range(0, counters[i], 1):
            dataset[k, :] = data_sorted[i][j][:]
            labels[k] = i
            k += 1
    indexes = [(0, counters[0])]
    for i in range(1, classes, 1):
        indexes.append((indexes[i-1][1], indexes[i-1][1] + counters[i]))

    return dataset, labels, indexes


def shuffle_data(data, labels):
    """
    Shuffles the dataset.

    ----------

    :param data:
        An m x n numpy array where m is the number of images and n its dimensionality.
    :param labels:
        An 1D array (no singleton dimensions) of length m containing the data's labels.

    :return:
        data
            The suffled m x n numpy dataset array.
        labels
            The suffled 1D array (no singleton dimensions) of the data's labels.
    """
    data = np.concatenate((data, np.expand_dims(labels, axis=1)), axis=1)
    data = np.random.permutation(data)
    labels = data[:, data.shape[1] - 1]
    data = data[:, 0:data.shape[1] - 1]
    return data, labels


def plot_sample(sample):
    """
    Plots an image sample to the screen (usually for debugging purposes).

    ----------

    :param sample:
        An 1D sample for plotting. It's length should be a perfect square (ex. 784 = 28*28).

    :return: Nothing.
    """
    dim = (int(sample.shape[-1] ** 0.5), int(sample.shape[-1] ** 0.5))
    sample = sample.reshape(dim)
    fig, ax = plot.subplots()
    image = plot.imshow(sample, cmap='gray')
    fig.colorbar(image, ax=ax)
    plot.clim(0, 1)  # Color Bar Scale
    plot.show()


if __name__ == '__main__':
    data, labels = read_data()
    plot_sample(data[15100, :])
