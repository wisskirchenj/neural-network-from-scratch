import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import math
import neural.provide_data
from neural.one_layer_neural import OneLayerNeural, sigmoid, xavier, mse, cost_prime, sigmoid_prime
from neural.datasets import Datasets


def setup_neural() -> OneLayerNeural:
    neural.provide_data.PATH_DATA = '.'
    test_data_path = 'test/data/train_subset.csv'
    data = neural.provide_data.read_csv_as_bytes(test_data_path)
    return OneLayerNeural(Datasets(data, data))


class OneLayerNeuralTest(unittest.TestCase):

    @staticmethod
    def test_forward_step_train_data_stage2():
        neural_net = setup_neural()
        expected = [0.42286725173173645, 0.7863175754895444, 0.8539526054946633, 0.5878649450449149,
                    0.25332037818521796, 0.10846218815633128, 0.16132366288535738, 0.5036812915517841,
                    0.3964910811110527, 0.3378884293704012, 0.41855072861332493, 0.7054997389006183,
                    0.7505531230813576, 0.5719368355036794, 0.1489699055476822, 0.27148952600271775,
                    0.36007653088357106, 0.5919441786952234, 0.6768202587658815, 0.5221310346087062]
        assert_array_almost_equal(expected, neural_net.forward_step(range(2)).flatten())

    @staticmethod
    def test_sigmoid_as_stage1():
        assert_array_almost_equal(
            [0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823],
            sigmoid(np.array(range(-1, 3)))
        )

    @staticmethod
    def test_xavier_as_stage():
        print(xavier(2, 3, 3042022).flatten())
        assert_array_almost_equal(
            [-1.07159039, 0.49733594, 0.8762539, 0.25949835, -0.8956624, -1.0133875],
            xavier(2, 3, 3042022).flatten()
        )

    def test_xavier_1dim(self):
        result = xavier(1, 2, 3042022).flatten()
        self.assertEqual(2, len(result))
        assert_array_equal([-1.3834172431039413, 0.6420579432446958], result)

    def test_xavier_limit(self):
        result = xavier(2, 2, 0).flatten()
        self.assertEqual(4, len(result))
        [self.assertGreaterEqual(el, -math.sqrt(6 / 4)) for el in result]
        [self.assertLessEqual(el, math.sqrt(6 / 4)) for el in result]

    def test_mse_stage3(self):
        y1 = np.array([-1, 0, 1, 2])
        y2 = np.array([4, 3, 2, 1])
        self.assertEqual(9.0, mse(y1, y2))

    @staticmethod
    def test_cost_prime_stage3():
        y1 = np.array([-1, 0, 1, 2])
        y2 = np.array([4, 3, 2, 1])
        assert_array_equal([-10, -6, -2, 2], cost_prime(y1, y2))

    @staticmethod
    def test_sigmoid_prime_stage3():
        y1 = np.array([-1, 0, 1, 2])
        assert_array_almost_equal([0.19661193324148185, 0.25, 0.19661193324148185, 0.10499358540350662],
                                  sigmoid_prime(y1))

    def test_mean_square_error(self):
        neural_net = setup_neural()
        self.assertEqual(0.0, neural_net.mean_square_error(neural_net.data.y_train, range(0, 2)))
        forward = neural_net.forward_step(range(0, 2))
        self.assertAlmostEqual(0.2308404313307, neural_net.mean_square_error(forward, range(0, 2)))

    def assert_list_almost_equal(self, expected: list[float], values: list[float]):
        self.assertEqual(len(expected), len(values))
        for i in range(len(expected)):
            self.assertAlmostEqual(expected[i], values[i], delta=0.000000000000001)
