from math import sqrt, exp

from numpy import ndarray
from numpy.random import default_rng


def xavier(n_in: int, n_out: int) -> ndarray:
    limit = sqrt(6 / (n_in + n_out))
    return default_rng(seed=3042022).uniform(-limit, limit, (n_in, n_out))


def sigmoid(x: int | float) -> float:
    return 1 / (1 + exp(-x))