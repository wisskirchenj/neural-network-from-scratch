import unittest
from unittest.mock import patch
from numpy.testing import assert_array_equal
from io import StringIO
import os

from numpy import array

import neural.provide_data
from neural.neural_network import main, one_hot_encode


class NeuralNetworkTest(unittest.TestCase):

    @patch('sys.stdout', new_callable=StringIO)
    def test_main(self, mock_stdout):
        if not os.path.exists('../dataSources'):
            # we are probably in CI, where the data is not uploaded
            print('No data found ...')
            return
        neural.provide_data.PATH_DATA = '../dataSources'
        main()
        self.assertEqual(
            '[0.42286725173173645, 0.7863175754895444, 0.8539526054946633, '
            '0.5878649450449149, 0.25332037818521796, 0.10846218815633128, '
            '0.16132366288535738, 0.5036812915517841, 0.3964910811110527, '
            '0.3378884293704012, 0.41855072861332493, 0.7054997389006183, '
            '0.7505531230813576, 0.5719368355036794, 0.1489699055476822, '
            '0.27148952600271775, 0.36007653088357106, 0.5919441786952234, '
            '0.6768202587658815, 0.5221310346087062]\n',
            mock_stdout.getvalue()
        )

    def test_one_hot_encode(self):
        categorical = array([3, 1, 4, 1])
        encoded = one_hot_encode(categorical)
        self.assertTupleEqual((4, 5), encoded.shape)
        assert_array_equal(array([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0]]), encoded)
