import sys
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plot
import random
import time


class SOM:
    """
    An implementation of the Self Organizing Map Neural Network.
    Specifically Kohonen's variation.
    """
    def __init__(self, data_set):
        """The class'es constructor."""
        # The map's shape.
        # First two argumments define dimensionality.
        # If 1D map is needed provide a single number else provide 2 numbers
        # The last element is the nodes' dimensionality which should be equal to that of the data.
        # ex. (10, data_set.shape[1]) for a 1D map with 10 nodes,
        #     (10, 10, data_set.shape[1]) for a 2D map with 10 * 10 nodes
        self.grid_shape = (7, 7, data_set.shape[1])
        # The map's initialization method.
        # 'sampling': Draws random instances from the data to initialize the map
        # 'random': Initializes the map with random noise
        self.init_method = 'random'
        # An ad-hoc argument to "extend" the neighbourhood function.
        # 'simple_chain' means that topologically only the upper, lower, right and left
        #   neighbour of the winning neuron will receive updates regardless
        #   of their neighbourhood function value.
        # 'full_chain: Everyone is neighbour with everyone
        # thus receiving updates proportionally to the neighbourhood function
        self.grid_mode = 'full_chain'
        # Defines the neighbourhood function. Currently only 'Gaussian' is implemented
        self.similarity = 'Gaussian'
        # The actual map
        self.grid = np.zeros(self.grid_shape, dtype='float32')
        # The maps's rank (1D or 2D)
        self.grid_rank = len(self.grid_shape) - 1
        # The learning rate
        self.alpha = 0.01
        # The number of epochs the network is gonna be trained into
        self.epochs = 1
        # The radius of the Gaussian neighbourhood function
        if self.similarity == 'Gaussian':
            self.sigma = int(data_set.shape[1] / 10)

        self.initialize_grid(data_set)

    def initialize_grid(self, data_set):
        """
            Initializes the map depending on the initialization method (And the grid's rank).
            Stops executution if an invalid initialization method
            or grid shape were provided in the constructor.

            ----------

            :param data_set:
                An m x n numpy array where m is the number of images and n the dimensionality of the data.
            :return: Nothing.
        """
        if self.init_method == 'random':
            mean_vector = np.mean(data_set, axis=0)
            covariance_matrix = 0.001 * np.eye(self.grid_shape[-1])
            self.grid = np.random.multivariate_normal(mean_vector, covariance_matrix, self.grid_shape[:-1])
        elif self.init_method == 'sampling':
            if self.grid_rank == 1:  # 1D Grid
                for i in range(0, self.grid_shape[0], 1):
                    index = rng.random_integers(0, data_set.shape[0] - 1)
                    self.grid[i] = data_set[index, :]
            elif self.grid_rank == 2:  # 2D Grid
                for i in range(0, self.grid_shape[0], 1):
                    for j in range(0, self.grid_shape[1], 1):
                        index = rng.random_integers(0, data_set.shape[0] - 1)
                        # Add a coma and : in self.grid[i, j] in case something goes wrong
                        self.grid[i, j] = data_set[index, :]
            else:
                print('Invalid Map Shape. Please Check If Your Map is 1D or 2D Only')
                sys.exit(-1)
        else:
            print('Invalid Initialization Method. Please Check Argument "init" in architecture list')
            sys.exit(-1)

    def find_winner_neuron(self, sample):
        """
        Given a data instance, determines which neuron "wins" the sample
        and returns it's position within the map.

        ----------

        :param sample: An 1D array containing a single data sample.
        :return: winner_neuron - A tuple containing the coordinates of the winning neuron.
        """
        distances = np.zeros(self.grid_shape[:-1], dtype='float32')
        min_distance = 100000
        if self.grid_rank == 1:  # 1D Grid
            for i in range(0, self.grid_shape[0]):
                temp = self.grid[i] - sample
                temp = np.dot(temp.T, temp)
                temp = np.sqrt(temp)
                distances[i] = temp
                if distances[i] < min_distance:
                    winner = (i,)
                    min_distance = distances[winner]
        elif self.grid_rank == 2:  # 2D Grid
            for i in range(0, self.grid_shape[0]):
                for j in range(0, self.grid_shape[1]):
                    temp = self.grid[i, j] - sample
                    temp = np.dot(temp.T, temp)
                    temp = np.sqrt(temp)
                    distances[i, j] = temp
                    if distances[i, j] < min_distance:
                        winner = (i, j)
                        min_distance = distances[winner]
        else:
            print("Higher dimension grid than 2 is not implemented")
            winner = None

        return winner

    @staticmethod
    def grid_distance(winner_neuron, current_neuron):
        """
        Defines the neighbour function.
        In other words how much of a neighbour is one neuron with another.

        ----------

        :param winner_neuron: A tuple of the winner neuron's grid coordinates.
        :param current_neuron: A tuple of the current neuron's grid coordinates.
        :return: similarity - A metric (real number) of the two neurons.
        """
        winner_neuron = np.asarray(winner_neuron, dtype='float32')
        current_neuron = np.asarray(current_neuron, dtype='float32')
        temp = winner_neuron - current_neuron
        similarity = np.dot(temp.T, temp)
        return similarity

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

    def alpha_decay(self, n):
        """
        Decaying the learning rate in conjunction with
        the current epoch and the total number of epochs.
        See SOM bibliography for more.

        ----------

        :param n: The current epoch (also known as discrete time).
        :return: The a_n as decayed by time/epoch.
        """
        # Arbitrarily chosen constant
        constant = 100
        return self.alpha * np.exp(- 2 * n / constant)

    def sigma_decay(self, n):
        """
        Decaying the sigma of the neighbourhood function
        in conjunction with the current epoch and the total number of epochs.
        See SOM bibliography for more.

        ----------

        :param n: The current epoch (also known as discrete time).
        :return: The s_n as decayed by time/epoch.
        """
        # Arbitrarily chosen constant
        constant = 100
        return self.sigma * np.exp(-n * np.log(self.sigma) / constant)

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
        if self.grid_mode == 'full_chain':
            if self.similarity == 'Gaussian':
                if self.grid_rank == 1:  # 1D Grid
                    for i in range(0, self.grid_shape[0]):
                        neighbour_value = self.grid_distance(winner, (i,))
                        neighbourhood_value = self.neighbourhood_function(neighbour_value, sigma)
                        self.grid[i] += alpha * neighbourhood_value * (sample - self.grid[winner])
                elif self.grid_rank == 2:  # 2D Grid
                    for i in range(0, self.grid_shape[0]):
                        for j in range(0, self.grid_shape[1]):
                            neighbour_value = self.grid_distance(winner, (i, j))
                            neighbourhood_value = self.neighbourhood_function(neighbour_value, sigma)
                            self.grid[i, j] += alpha * neighbourhood_value * (sample - self.grid[winner])
        elif self.grid_mode == 'simple_chain':
            self.grid[winner] += alpha * (sample - self.grid[winner])
            if self.similarity == 'Gaussian':
                if self.grid_rank == 1:  # 2D Grid
                    # Updating the left neuron
                    if winner[0] - 1 >= 0:  # Checking if it is inside the map
                        neighbour_value = self.grid_distance(winner, (winner[0] - 1,))
                        neighbourhood_value = self.neighbourhood_function(neighbour_value, sigma)
                        self.grid[winner[0] - 1] += \
                            alpha * neighbourhood_value * (sample - self.grid[winner])
                    # Updating The right neuron
                    if winner[0] + 1 < self.grid_shape[0]:  # Checking if it is inside the map
                        neighbour_value = self.grid_distance(winner, (winner[0] + 1,))
                        neighbourhood_value = self.neighbourhood_function(neighbour_value, sigma)
                        self.grid[winner[0] + 1] += \
                            alpha * neighbourhood_value * (sample - self.grid[winner])
                elif self.grid_rank == 2:  # 2D Grid
                    # Updating the left neuron
                    if winner[0] - 1 >= 0:  # Checking if it is inside the map
                        neighbour_value = self.grid_distance(winner, (winner[0] - 1, winner[1]))
                        neighbourhood_value = self.neighbourhood_function(neighbour_value, sigma)
                        self.grid[winner[0] - 1, winner[1]] += \
                            alpha * neighbourhood_value * (sample - self.grid[winner])
                    # Updating the right neuron
                    if winner[0] + 1 < self.grid_shape[0]:  # Checking if it is inside the map
                        neighbour_value = self.grid_distance(winner, (winner[0] + 1, winner[1]))
                        neighbourhood_value = self.neighbourhood_function(neighbour_value, sigma)
                        self.grid[winner[0] + 1, winner[1]] += \
                            alpha * neighbourhood_value * (sample - self.grid[winner])
                    # Updating the upper neuron
                    if winner[1] - 1 >= 0:  # Checking if it is inside the map
                        neighbour_value = self.grid_distance(winner, (winner[0], winner[1] - 1))
                        neighbourhood_value = self.neighbourhood_function(neighbour_value, sigma)
                        self.grid[winner[0], winner[1] - 1] += \
                            alpha * neighbourhood_value * (sample - self.grid[winner])
                    # Updating The Lower Neuron
                    # Checking if it is inside the map
                    if winner[1] + 1 < self.grid_shape[1]:
                        neighbour_value = self.grid_distance(winner, (winner[0], winner[1] + 1))
                        neighbourhood_value = self.neighbourhood_function(neighbour_value, sigma)
                        self.grid[winner[0], winner[1] + 1] += \
                            alpha * neighbourhood_value * (sample - self.grid[winner])

    def train(self, training_set):
        """
        Trains the map iterating over the dataset for each epoch.
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
        for i in range(0, self.epochs, 1):  # Iterating over the epochs
            new_alpha = self.alpha_decay(i)
            new_sigma = self.sigma_decay(i)
            for j in range(0, training_set.shape[0], 1):  # Iterating over the dataset
                self.feed_sample(training_set[j, :], new_alpha, new_sigma)
        self.plot_grid()
        stop = time.clock()
        print('Testing Time:', (stop - start) / 60, 'minutes')

    def test(self, dataset):
        """
        Throws a random sample to the map and highlights the neuron that "caught" it.

        ----------

        :param dataset: The dataset from which the test sample will be randomly selected.
        :return: Nothing.
        """
        random_index = random.randrange(0, dataset.shape[0])
        winner_neuron = self.find_winner_neuron(dataset[random_index, :])
        self.plot_grid(winner_neuron)

    def plot_grid(self, *args):
        """
        Plots a Grid To The Screen (1D or 2D).
        If the function is called by the test function it highlights
        the neuron that "won" the test sample. In that case a tuple
        with the winning neuron's coordinates is passed to the function as argument.
        Currently works only for datasets where n_rows = n_columns.
        ----------

        :param args: A tuple containing the coordinates of the winning neuron.
        :return: Nothing.
        """
        total_neurons = 1
        for i in range(0, self.grid_rank):
            total_neurons *= self.grid_shape[i]
        dim = int(np.sqrt(self.grid_shape[-1]))
        temp_grid = np.reshape(self.grid, (total_neurons, dim, dim))
        reference_tuple = (1,) + self.grid_shape[:-1] if self.grid_rank == 1 else self.grid_shape[:-1]
        step = total_neurons
        winning_neuron_position = None
        if len(args) > 0:
            winning_neuron_position = 0
            for i in range(0, len(args[0])):
                step //= self.grid_shape[i]
                winning_neuron_position += step*args[0][i]
        for i, neuron in enumerate(temp_grid):
            ax = plot.subplot(reference_tuple[0], reference_tuple[1], i+1)
            ax.set_axis_off()
            if i == winning_neuron_position:
                plot.imshow(neuron, cmap='plasma')
            else:
                plot.imshow(neuron, cmap='gray')
        plot.show()

    def load_map(self, path):
        """
        Loads a pre-trained model. Stops execution if invalid path was provided
        or the loaded map's shape is not the same as the one defined in the constructor.

        ----------

        :param path:
            A string containing the path of the pre-trained map.
        :return: Nothing.
        """
        loaded_map = np.load(path + '_' + str(self.grid_rank) + 'D' + '.npy')
        if self.grid.shape != loaded_map.shape:
            print('Error: Input map dimensionality is:', loaded_map.shape)
            print('The shape defined in the constructor is:', self.grid.shape)
            sys.exit(-1)
        else:
            self.grid = loaded_map

    def save_map(self, class_name):
        """
        Saves a map to the disk.

        ----------

        :param class_name: A string containing the name that the map will be saved as.
        :return: Nothing.
        """
        np.save(class_name + '_' + str(self.grid_rank) + 'D', self.grid)
