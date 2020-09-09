import sys
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plot
import time


class SOM:
    """
    An implementation of the Self Organizing Map Neural Network.
    Specifically Kohonen's variation.
    """
    def __init__(self, data_set):
        """The class's constructor."""
        # The map's shape.
        # First two arguments define dimensionality.
        # If 1D map is needed provide a single number else provide 2 numbers
        # The last element is the nodes' dimensionality which should be equal to the data set's.
        # ex. (10, data_set.shape[1]) for a 1D map with 10 nodes,
        #     (10, 10, data_set.shape[1]) for a 2D map with 10 * 10 nodes
        self._grid_shape = (8, 8, data_set.shape[1])
        # The total number of the map's neurons.
        self._total_neurons = int(np.prod(self._grid_shape[:-1]))
        # If the data is grayscale images this field defines it's height and width for plotting.
        self._plot_shape = (int(self._grid_shape[-1] ** 0.5), int(self._grid_shape[-1] ** 0.5))
        # The map's initialization method.
        # 'sampling': Draws random instances from the data to initialize the map
        # 'random': Initializes the map with samples from a multivariate Gaussian PDF.
        self._init_method = 'random'
        # The map's rank (1D or 2D).
        self._grid_rank = len(self._grid_shape) - 1
        # The learning rate.
        self._alpha = 0.01
        self._alpha_constant = 100
        # The radius of the Gaussian neighbourhood function. Fine-tuned initialization.
        self._sigma = int(data_set.shape[1] / 10)
        self._sigma_constant = 100
        # The number of epochs the network is gonna be trained into.
        self._epochs = 200
        # The actual map
        self._grid = self.initialize_grid(data_set)
        # A 2D array containing the neuron's pairwise distances upon the multidimensional grid.
        self._grid_distances = self.initialize_grid_distances()

    def initialize_grid(self, data_set):
        """
            Initializes the map depending on the initialization method.
            'random' Draws random vectors from a multivariate Gaussian distribution
            with the mean sample of the data set's as the mean vector and
            a scaled Identity matrix as the covariance matrix.
            'sampling' draws n random samples from the data set's (without replacement)
            and assigns them to to the map's neurons.
            Stops execution if an invalid initialization method was provided in the constructor.

            ----------

            :param data_set:
                An m x n numpy array where m is the number of samples and n its dimensionality.
            :return: An m x n numpy array with the initialized grid.
        """
        if self._init_method == 'random':
            mean_vector = np.mean(data_set, axis=0)
            covariance_matrix = 0.001 * np.eye(self._grid_shape[-1])
            return rng.multivariate_normal(mean_vector, covariance_matrix, self._total_neurons)
        elif self._init_method == 'sampling':
            indexes = tuple(rng.choice(data_set.shape[0], self._total_neurons, False))
            return data_set[indexes, :]
        else:
            print('Invalid Initialization Method.')
            print('Please Check Argument "init" in architecture list')
            sys.exit(-1)

    def initialize_grid_distances(self):
        """
            Computes the distance between two neurons (x, y) in grid coordinates
            for each pair of neurons in the map's grid.

            ----------

            :return: An m x m numpy array containing the neuron's pairwise grid distances.
        """
        grid_coordinates = np.zeros((self._total_neurons, self._total_neurons))
        for i in range(0, self._total_neurons):
            index_1 = np.asarray(np.unravel_index(i, self._grid_shape[:-1]))
            for j in range(0, self._total_neurons):
                index_2 = np.asarray(np.unravel_index(j, self._grid_shape[:-1]))
                grid_coordinates[i, j] = np.linalg.norm(index_1 - index_2)
        return grid_coordinates

    def find_winner_neuron(self, sample):
        """
        Given a data instance, determines which neuron "wins" the sample
        and returns it's position within the map as a tuple.

        ----------

        :param sample: An 1D array containing a single data sample.
        :return: winner_neuron - A tuple of size 1 containing the position
                 of the winning neuron (single number indexing)
        """
        distances = np.linalg.norm(self._grid - sample, axis=1)
        winner = np.unravel_index(np.argmin(distances), distances.shape)
        return winner

    @staticmethod
    def neighbourhood_function(grid_distance, sigma):
        """
        Defines The Gaussian Neighbourhood Function. See SOM bibliography for more.

        ----------

        :param grid_distance: The similarity measure between two neurons.
            See grid_distance() for more.
        :param sigma: The sigma of the Gaussian similarity as defined in the constructor.
        :return: The neighbourhood function's value for two neurons.
        """
        return np.exp(-0.5 * grid_distance / sigma ** 2)

    def update_alpha(self, n):
        """
        Decaying the learning rate in conjunction with
        the current epoch and the total number of epochs.
        See SOM bibliography for more.

        ----------

        :param n: The current epoch (also known as discrete time).
        :return: The a_n as decayed by time/epoch.
        """
        return self._alpha * np.exp(- 2 * n / self._alpha_constant)

    def update_sigma(self, n):
        """
        Decaying the sigma of the neighbourhood function
        in conjunction with the current epoch and the total number of epochs.
        See SOM bibliography for more.

        ----------

        :param n: The current epoch (also known as discrete time).
        :return: The s_n as decayed by time/epoch.
        """
        return self._sigma * np.exp(-n * np.log(self._sigma) / self._sigma_constant)

    def feed_sample(self, sample, alpha, sigma):
        """
        "Feeds" a sample to the map, then finds the "winner" neuron
        and performs all the necessary updates to the that neuron and its neighbours.

        ----------

        :param sample: The 1D sample provided to the map.
        :param alpha: The learning rate.
        :param sigma: The radius of the Gaussian similarity.
        :return: Nothing.
        """
        winner = self.find_winner_neuron(sample)
        winner_grid_distances = self._grid_distances[winner, :]
        neighbour_values = self.neighbourhood_function(winner_grid_distances, sigma)
        self._grid += alpha * np.multiply(neighbour_values.T, sample - self._grid[winner, :])
        return

    def train(self, training_set):
        """
        Trains the map iterating over the data set for each epoch.
        Plots the Self-Organizing Map one time before training
        and one time after training.

        ----------

        :param training_set: An m x n numpy array where
            m is the number of images and n dimensionality of the data.
        :return: Nothing.
        """
        self.plot_grid()
        print('\nStarted Training Map! Please Wait...')
        start = time.clock()
        for i in range(0, self._epochs, 1):
            new_alpha = self.update_alpha(i)
            new_sigma = self.update_sigma(i)
            for j in range(0, training_set.shape[0], 1):
                self.feed_sample(training_set[j, :], new_alpha, new_sigma)
        stop = time.clock()
        print('Training Time:', (stop - start) / 60, 'minutes')
        self.plot_grid()
        return

    def test(self, dataset):
        """
        Throws a random sample to the map and highlights the neuron that "caught" it.

        ----------

        :param dataset: The dataset from which the test sample will be randomly selected.
        :return: Nothing.
        """
        random_index = rng.choice(dataset.shape[0])
        winner_neuron = self.find_winner_neuron(dataset[random_index, :])
        self.plot_grid(winner_neuron)
        return

    def plot_grid(self, *args):
        """
        Plots a Grid To The Screen (1D or 2D).
        If the function is called by the test function it highlights
        the neuron that "won" the test sample. In that case a tuple
        with the winning neuron's coordinates is passed to the function as argument.
        Currently works only for data set's where n_rows = n_columns.
        ----------

        :param args: A tuple containing the coordinates of the winning neuron.
        :return: Nothing.
        """
        if self._grid_rank == 1:
            reference_tuple = (1,) + self._grid_shape[:-1]
        elif self._grid_rank == 2:
            reference_tuple = self._grid_shape[:-1]
        else:
            print('Error. 3D or higher gird cannot be plotted.')
            return
        winning_neuron = None
        if len(args) > 0:
            winning_neuron = args[0][0]
        for i in range(0, self._total_neurons):
            ax = plot.subplot(reference_tuple[0], reference_tuple[1], i+1)
            ax.set_axis_off()
            neuron = np.reshape(self._grid[i, :], self._plot_shape)
            if i == winning_neuron:
                plot.imshow(neuron, cmap='plasma')
            else:
                plot.imshow(neuron, cmap='gray')
        plot.show()
        return

    def load_map(self, path):
        """
        Loads a pre-trained model. Stops execution if invalid path was provided
        or the loaded map's shape is not the same as the one defined in the constructor.

        ----------

        :param path:
            A string containing the path of the pre-trained map.
        :return: Nothing.
        """
        loaded_map = np.load(path + '_' + str(self._grid_rank) + 'D' + '.npy')
        if loaded_map.shape[0] == self._total_neurons:
            self._grid = loaded_map
        else:
            print('Error! Loaded map shape = ', loaded_map.shape)
            print('does not match the one defined in the constructor:', self._grid.shape)
            sys.exit(-1)
        return

    def save_map(self, class_name):
        """
        Saves a map to the disk.

        ----------

        :param class_name: A string containing the name that the map will be saved as.
        :return: Nothing.
        """
        np.save(class_name + '_' + str(self._grid_rank) + 'D', self._grid)
        return
