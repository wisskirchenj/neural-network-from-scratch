from math import sqrt
import numpy as np
from numpy.random import default_rng


# Xavier's initialization is used to initialize network weights and biases with random numbers of an interval
# parameterized by input and output dimensions,  thus optimized in sensitivity to the sigmoid activation function
def xavier(n_in: int, n_out: int, seed=3042022) -> np.ndarray:
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
