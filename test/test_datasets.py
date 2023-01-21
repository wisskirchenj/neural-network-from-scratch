import unittest
from numpy.testing import assert_array_equal
import numpy as np

from neural.datasets import one_hot_encode


class DatasetsTest(unittest.TestCase):

    def test_one_hot_encode(self):
        categorical = np.array([3, 1, 4, 1])
        encoded = one_hot_encode(categorical)
        self.assertTupleEqual((4, 5), encoded.shape)
        assert_array_equal(np.array([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0]]), encoded)
