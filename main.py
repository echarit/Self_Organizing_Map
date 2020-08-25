from Initialize import *
from SOM import *


def main():
    # Properly edit this function in the "Initialize.py"
    # if you want to train on your own dataset
    training_set, training_labels, indexes, num_of_classes = read_data()
    # Change to false if your dataset contains no labels whatsoever or
    # you want a single map for all your data anyway.
    map_for_each_class = True
    pre_trained = False
    if not map_for_each_class:
        path = 'Kohonen_map'
        som = SOM(training_set)
        if pre_trained:
            som.load_map(path + '.npy')
        else:
            som.train(training_set)
            som.save(path)
        som.test(training_set)
    else:
        # Each class of the dataset gets its own SOM
        for i in range(0, num_of_classes, 1):
            partial_dataset = training_set[indexes[i][0]:indexes[i][1]]
            som = SOM(partial_dataset)
            path = str(i)
            if pre_trained:
                som.load_map(path + '.npy')
            else:
                som.train(partial_dataset)
                som.save(path)
            som.test(partial_dataset)


if __name__ == '__main__':
    main()
