from math import exp, sqrt
import numpy as np
from numpy.random import default_rng
from neural.datasets import Datasets


# sigmoid - besides Rectified Linear Unit (ReLU) - is a typical activation function to calculate neuron activation
# which is the output of a neuron layer - and input to the next layer.
# For any real x, sigmoid(x) lies in [0,1]
def sigmoid(x: int | float) -> float:
    return 1 / (1 + exp(-x))


# Xavier' initialization is used to initialize network weights and biases with random numbers of an interval
# parameterized by input and output dimensions,  thus optimized in sensitivity to the sigmaoid activation function
def xavier(n_in: int, n_out: int, seed: int) -> np.ndarray:
    limit = sqrt(6 / (n_in + n_out))
    return default_rng(seed=seed).uniform(-limit, limit, (n_in, n_out))


# class represents a neural network with one neuron layer.
# There are as many output layer neurons as categories. In an ideal model, each data record that is feed to the net
# leads to an activation of 1 for the neuron assigned to the category of the data and 0 for all other neurons.
# In reality, the ideal is never achieved and the mean square error (MSE) of the differences is a metrics for the
# efficiency of the trained network.
class OneLayerNeural:

    # the network dimensions itself by the data feed.
    def __init__(self, data: Datasets):
        self.data = data
        self.neuron_count = data.get_output_dimension()
        # use seeds for test assertions
        self.weights = xavier(data.get_input_dimension(), self.neuron_count, 3042022)
        self.bias = xavier(1, self.neuron_count, 3042022).flatten()

    # Perform a forward propagation step (part of epoch) on a subset (specified as index range) of the train data
    def forward_step_train_data(self, subrange: range) -> list[float]:
        return [sigmoid(self.calc_neuron_input(self.data.x_train[i], n))
                for i in subrange
                for n in range(self.neuron_count)]

    # the input of one neuron is the sum of the (image) data weighted with the current network weights plus a bias.
    # training of the model iteratively optimizes weights and biases in epochs
    # an epoch is the cycle of forward propagation step, backward propagation step and update of the cost function
    # parameters (weights and biases of all layers in general)
    def calc_neuron_input(self, image_data: np.ndarray, neuron: int) -> float:
        return np.dot(image_data, self.weights[:, neuron]) + self.bias[neuron]
