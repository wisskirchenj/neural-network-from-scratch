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
            '[0.16862745098039217, 0.403921568627451] '
            + '[-1.0715903886795484, 0.4973359442973788, 0.8762538954575065, '
            + '0.25949835095465223, -0.8956623974165681, -1.0133874996404446] '
            + '[0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823]\n',
            mock_stdout.getvalue()
        )

    def test_one_hot_encode(self):
        categorical = array([3, 1, 4, 1])
        encoded = one_hot_encode(categorical)
        self.assertTupleEqual((4, 5), encoded.shape)
        assert_array_equal(array([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0]]), encoded)
