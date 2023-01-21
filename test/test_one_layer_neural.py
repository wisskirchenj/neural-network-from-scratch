import unittest
from numpy.testing import assert_array_equal
import math
import neural.provide_data
from neural.one_layer_neural import OneLayerNeural, sigmoid, xavier
from neural.datasets import Datasets


class OneLayerNeuralTest(unittest.TestCase):

    def test_forward_step_train_data_stage2(self):
        neural.provide_data.PATH_DATA = '.'
        test_data_path = 'test/data/train_subset.csv'
        data = neural.provide_data.read_csv_as_bytes(test_data_path)
        neural_net = OneLayerNeural(Datasets(data, data))
        expected = [0.42286725173173645, 0.7863175754895444, 0.8539526054946633, 0.5878649450449149,
                    0.25332037818521796, 0.10846218815633128, 0.16132366288535738, 0.5036812915517841,
                    0.3964910811110527, 0.3378884293704012, 0.41855072861332493, 0.7054997389006183,
                    0.7505531230813576, 0.5719368355036794, 0.1489699055476822, 0.27148952600271775,
                    0.36007653088357106, 0.5919441786952234, 0.6768202587658815, 0.5221310346087062]
        self.assert_list_almost_equal(expected, neural_net.forward_step_train_data(range(2)))

    def test_sigmoid_as_stage1(self):
        self.assert_list_almost_equal(
            [0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823],
            [sigmoid(i) for i in range(-1, 3)]
        )

    @staticmethod
    def test_xavier_as_stage():
        assert_array_equal(
            [-1.0715903886795484, 0.4973359442973788, 0.8762538954575065,
             0.25949835095465223, -0.8956623974165681, -1.0133874996404446],
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

    def assert_list_almost_equal(self, expected: list[float], values: list[float]):
        self.assertEqual(len(expected), len(values))
        for i in range(len(expected)):
            self.assertAlmostEqual(expected[i], values[i], delta=0.000000000000001)
