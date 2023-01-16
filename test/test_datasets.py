import unittest
from numpy.testing import assert_array_equal
import numpy as np

from neural.datasets import Datasets, MAX_VALUE


class DatasetsTest(unittest.TestCase):

    def test_scale_x_sets(self):
        sets = Datasets(np.array([4]), np.array([[8, 4], [12, 6]]), np.array([0]), np.array([0]))
        sets.scale_x_sets()
        self.assertEqual(np.array([4 / MAX_VALUE]), sets.x_train)
        assert_array_equal(
            np.array([[8 / MAX_VALUE, 4 / MAX_VALUE], [12 / MAX_VALUE, 6 / MAX_VALUE]]), sets.x_test)
