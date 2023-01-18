import math
import unittest
from numpy.testing import assert_array_equal
from neural.initialization import xavier, sigmoid


class InitializationTest(unittest.TestCase):

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

    def test_sigmoid_as_stage(self):
        self.assertListEqual(
            [0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823],
            [sigmoid(i) for i in range(-1, 3)]
        )
