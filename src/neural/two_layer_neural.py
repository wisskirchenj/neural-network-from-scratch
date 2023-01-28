import numpy as np
from neural.datasets import Datasets
from neural.neural_utils import xavier, sigmoid


# class represents a neural network with two neuron layers - the hidden layer has DEFAULT of 64 neurons.
# There are as many output layer neurons as categories.
class TwoLayerNeural:

    # the network dimensions itself by the data feed.
    def __init__(self, data: Datasets, hidden_layer_size=64, eta=0.3):
        self.data = data
        # "learning rate" eta determines convergence of gradient descent (kind of step width factor)
        self.eta = eta
        self.neuron_count = data.get_output_dimension()
        # use seeds for test assertions
        self.weights = [xavier(data.get_input_dimension(), hidden_layer_size),
                        xavier(hidden_layer_size, self.neuron_count)]
        self.bias = [xavier(1, hidden_layer_size).flatten(), xavier(1, self.neuron_count).flatten()]
        # initialize z's list of the neuron activation function input, that is repeatedly used in back propagation
        self.zs = [[] for _ in range(2)]
        # count the epochs performed on training data (sub-)set
        self.epoch_count = 0

    # Perform a forward propagation step (e.g. as part of epoch) on a subset (specified as index range)
    # of the train data or test data
    def forward_step(self, subrange: range, use_train_data=True) -> np.ndarray:
        dataset = self.data.x_train if use_train_data else self.data.x_test
        self.zs[0] = ([self.weights[0].T  @ dataset[i] + self.bias[0] for i in subrange])
        hidden_layer_activations = [sigmoid(z) for z in self.zs[0]]
        self.zs[1] = ([self.weights[1].T  @ hidden_layer_activations[i] + self.bias[1] for i in subrange])
        return np.array([sigmoid(z) for z in self.zs[1]])
