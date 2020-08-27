from Initialize import *
from SOM import *


def main():
    # Properly edit this function in the "Initialize.py" if you want to train on your own dataset
    training_set, training_labels = read_data()
    # Change to false if your dataset contains no labels whatsoever or
    # you want a single map for all your data anyway.
    map_for_each_class = True
    pre_trained = True
    if not map_for_each_class:
        training_set, training_labels = shuffle_data(training_set, training_labels)
        path = 'kohonen_map'
        som = SOM(training_set)
        if pre_trained:
            som.load_map(path)
        else:
            som.train(training_set)
            som.save_map(path)
        som.test(training_set)
    else:  # Each class of the dataset gets its own SOM
        training_set, training_labels, indexes = sort_dataset(training_set, training_labels)
        number_of_classes = int(np.amax(training_labels)) + 1
        for i in range(0, number_of_classes, 1):
            partial_dataset = training_set[indexes[i][0]:indexes[i][1]]
            som = SOM(partial_dataset)
            path = str(i)
            if pre_trained:
                som.load_map(path)
                som.save_map(path)
            else:
                som.train(partial_dataset)
                som.save_map(path)
            som.test(partial_dataset)


if __name__ == '__main__':
    main()
