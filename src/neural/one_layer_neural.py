from math import sqrt
import numpy as np
from numpy.random import default_rng
from neural.datasets import Datasets


# Xavier's initialization is used to initialize network weights and biases with random numbers of an interval
# parameterized by input and output dimensions,  thus optimized in sensitivity to the sigmoid activation function
def xavier(n_in: int, n_out: int, seed: int) -> np.ndarray:
    limit = sqrt(6 / (n_in + n_out))
    return default_rng(seed=seed).uniform(-limit, limit, (n_in, n_out))


# sigmoid - besides Rectified Linear Unit (ReLU) - is a typical activation function to calculate neuron activation
# which is the output of a neuron layer - and input to the next layer.
# For any real x, sigmoid(x) lies in [0,1] - function is applied to all elements of a vector
def sigmoid(x_vec: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x_vec))


# derivative of the sigmoid function
def sigmoid_prime(x_vec: np.ndarray) -> np.ndarray:
    return sigmoid(x_vec) * (1 - sigmoid(x_vec))


# calc mean square error on the 1d arrays given
def mse(y_predicted: np.ndarray, y_categorical: np.ndarray) -> float:
    return np.dot(y_predicted - y_categorical, y_predicted - y_categorical) / y_categorical.size


# outer derivative of the cost function - as needed in the gradient descent
def cost_prime(y_predicted: np.ndarray, y_categorical: np.ndarray) -> np.ndarray:
    return 2 * (y_predicted - y_categorical)


# class represents a neural network with one neuron layer.
# There are as many output layer neurons as categories. In an ideal model, each data record that is feed to the net
# leads to an activation of 1 for the neuron assigned to the category of the data and 0 for all other neurons.
# In reality, the ideal is never achieved and the mean square error (MSE) of the differences is a metrics for the
# efficiency of the trained network.
class OneLayerNeural:

    # the network dimensions itself by the data feed.
    def __init__(self, data: Datasets, eta=0.1):
        self.data = data
        # "learning rate" eta determines convergence of gradient descent (kind of step width factor)
        self.eta = eta
        self.neuron_count = data.get_output_dimension()
        # use seeds for test assertions
        self.weights = xavier(data.get_input_dimension(), self.neuron_count, 3042022)
        self.bias = xavier(1, self.neuron_count, 3042022).flatten()
        # initialize z's list of the neuron activation function input, that is repeatedly used in back propagation
        self.zs = []
        # count the epochs performed on training data (sub-)set
        self.epoch_count = 0

    # Perform a forward propagation step (e.g. as part of epoch) on a subset (specified as index range)
    # of the train data or test data
    def forward_step(self, subrange: range, use_train_data=True) -> np.ndarray:
        dataset = self.data.x_train if use_train_data else self.data.x_test
        self.zs = [self.weights.T  @ dataset[i] + self.bias for i in subrange]
        return np.array([sigmoid(z) for z in self.zs])

    # perform one "Epoch" consisting of a back propagation with forward step and the according update of weights
    # and biases. These are updated "permanently" in the neural network's fields.
    def epoch(self, subrange: range):
        gradient_weight, gradient_bias = self.backward_propagation(subrange)
        self.weights -= self.eta * gradient_weight
        self.bias -= self.eta * gradient_bias
        self.epoch_count += 1

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

    # perform a forward step with the currently trained (or untrained) weights and biases
    # and return its mean square error compared to the true categorical data
    def mean_square_error(self, subrange: range, use_train_data=True) -> float:
        y_categorical = self.data.y_train if use_train_data else self.data.y_test
        y_categorical = y_categorical[subrange].flatten()
        y_predicted = self.forward_step(subrange, use_train_data).flatten()
        return mse(y_predicted, y_categorical)

    # perform a forward step on all test data, compare each network categorization
    # (i.e. # of neuron with maximal value = argmax) with the true label (as in one-hit encoded y_test set)
    # and count the correct categorization. Return the ratio of correct categorizations.
    def accuracy_in_test(self) -> float:
        count = 0
        data_size = self.data.y_test.shape[0]
        for i in range(data_size):
            if self.data.y_test[i][np.argmax(self.forward_step(range(i, i + 1), use_train_data=False))] == 1:
                count = count + 1
        return count / data_size

    # perform an epoch on all train data and return accuracy on test data
    def next_epoch_accuracy(self, batch_size=100) -> float:
        train_datasize = self.data.x_train.shape[0]
        for batch in range(0, train_datasize, batch_size):
            self.epoch(range(batch, batch + batch_size))
        return self.accuracy_in_test()
