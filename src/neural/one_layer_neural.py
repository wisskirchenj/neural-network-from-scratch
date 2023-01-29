import numpy as np
from neural.neural_network import NeuralNetwork, Datasets, xavier, sigmoid, sigmoid_prime, cost_prime


# class represents a neural network with one neuron layer.
# There are as many output layer neurons as categories.
class OneLayerNeural(NeuralNetwork):

    # the network dimensions itself by the data feed.
    def __init__(self, data: Datasets, eta=0.1):
        super().__init__(data, eta)
        self.weights = xavier(data.get_input_dimension(), data.get_output_dimension())
        self.bias = xavier(1, data.get_output_dimension()).flatten()
        # initialize z's list of the neuron activation function input, that is repeatedly used in back propagation
        self.zs = []

    # Perform a forward propagation step (e.g. as part of epoch) on a subset (specified as index range)
    # of the train data or test data
    def forward_step(self, subrange: range, use_train_data=True) -> np.ndarray:
        dataset = self.data.x_train if use_train_data else self.data.x_test
        self.zs = [self.weights.T  @ dataset[i] + self.bias for i in subrange]
        return np.array([sigmoid(z) for z in self.zs])

    # for all data records in the subrange do a forward step and calculate the gradient of weights and bias at the
    # resulting position for correction in the calling epoch.
    def backward_propagation(self, subrange: range) -> tuple[np.ndarray, np.ndarray]:
        forward_result = self.forward_step(subrange)
        gradient_weight = np.zeros(self.weights.shape)
        gradient_bias = np.zeros(self.bias.shape)
        for i, index in enumerate(subrange):
            bias_contribution = cost_prime(forward_result[i], self.data.y_train[index]) \
                                * sigmoid_prime(self.zs[i]) / len(subrange)
            gradient_bias += bias_contribution
            gradient_weight += np.outer(self.data.x_train[index], bias_contribution)
        return gradient_weight, gradient_bias

    # perform one "Epoch" consisting of a back propagation with forward step and the according update of weights
    # and biases. These are updated "permanently" in the neural network's fields.
    def epoch(self, subrange: range):
        gradient_weight, gradient_bias = self.backward_propagation(subrange)
        self.weights -= self.eta * gradient_weight
        self.bias -= self.eta * gradient_bias
        self.epoch_count += 1
