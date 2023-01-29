import numpy as np
from neural.neural_network import NeuralNetwork, Datasets, xavier, sigmoid, sigmoid_prime, cost_prime


# class represents a neural network with two neuron layers - the hidden layer has DEFAULT of 64 neurons.
# There are as many output layer neurons as categories.
class TwoLayerNeural(NeuralNetwork):

    # the network dimensions itself by the data feed.
    def __init__(self, data: Datasets, hidden_layer_size=64, eta=0.1):
        super().__init__(data, eta)
        self.weights = [xavier(data.get_input_dimension(), hidden_layer_size),
                        xavier(hidden_layer_size, data.get_output_dimension())]
        self.bias = [xavier(1, hidden_layer_size).flatten(), xavier(1, data.get_output_dimension()).flatten()]
        # initialize z's list of the neuron activation function input, that is repeatedly used in back propagation
        self.zs = [[] for _ in range(2)]
        self.hidden_activations = None

    # Perform a forward propagation step (e.g. as part of epoch) on a subset (specified as index range)
    # of the train data or test data
    def forward_step(self, subrange: range, use_train_data=True) -> np.ndarray:
        dataset = self.data.x_train if use_train_data else self.data.x_test
        self.zs[0] = [self.weights[0].T  @ dataset[i] + self.bias[0] for i in subrange]
        self.hidden_activations = np.array([sigmoid(z) for z in self.zs[0]])
        self.zs[1] = [self.weights[1].T @ self.hidden_activations[i] + self.bias[1] for i in range(len(subrange))]
        return np.array([sigmoid(z) for z in self.zs[1]])

    # for all data records in the subrange do a forward step and calculate the gradient of weights and bias at the
    # resulting position for correction in the calling epoch.
    def backward_propagation(self, subrange: range) -> tuple[list[np.ndarray], list[np.ndarray]]:
        forward_result = self.forward_step(subrange)
        gradient_weight = [np.zeros(self.weights[i].shape) for i in range(2)]
        gradient_bias = [np.zeros(self.bias[i].shape) for i in range(2)]
        for i, index in enumerate(subrange):
            bias_contribution = cost_prime(forward_result[i], self.data.y_train[index]) \
                                * sigmoid_prime(self.zs[1][i]) / len(subrange)
            gradient_bias[1] += bias_contribution
            gradient_weight[1] += np.outer(self.hidden_activations[i], bias_contribution)
            bias_contribution = self.weights[1] @ bias_contribution * sigmoid_prime(self.zs[0][i])
            gradient_bias[0] += bias_contribution
            gradient_weight[0] += np.outer(self.data.x_train[index], bias_contribution)
        return gradient_weight, gradient_bias

    # perform one "Epoch" consisting of a back propagation with forward step and the according update of weights
    # and biases. These are updated "permanently" in the neural network's fields.
    def epoch(self, subrange: range):
        gradient_weight, gradient_bias = self.backward_propagation(subrange)
        for i in range(2):
            self.weights[i] -= self.eta * gradient_weight[i]
            self.bias[i] -= self.eta * gradient_bias[i]
        self.epoch_count += 1
