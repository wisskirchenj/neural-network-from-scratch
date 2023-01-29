import unittest
from numpy.testing import assert_array_almost_equal

import neural.provide_data
from neural.two_layer_neural import TwoLayerNeural, Datasets


def setup_neural() -> TwoLayerNeural:
    neural.provide_data.PATH_DATA = '.'
    train_data = neural.provide_data.read_csv_as_bytes('test/data/train_subset.csv')
    test_data = neural.provide_data.read_csv_as_bytes('test/data/test_subset.csv')
    return TwoLayerNeural(Datasets(train_data, test_data))


class TwoLayerNeuralTest(unittest.TestCase):

    @staticmethod
    def test_forward_step_stage5():
        neural_net = setup_neural()
        expected = [0.081458149746659, 0.7078373861901439, 0.788047435931905, 0.5460663760416861,
                    0.4142727590233215, 0.3900817619848179, 0.30867710854793123, 0.8821701624025713,
                    0.5251529930398801, 0.6860112311436224, 0.1083062442365927, 0.7042764793747258,
                    0.7896127455170074, 0.5666891692622699, 0.42785055300952435, 0.3886295210179232,
                    0.3168363764198969, 0.8657427046740684, 0.5576544572719794, 0.7183489046929683]
        assert_array_almost_equal(expected, neural_net.forward_step(range(2)).flatten())

    def test_epoch_mean_square_stage6(self):
        neural_net = setup_neural()
        subrange = range(2)
        neural_net.epoch(subrange)
        expected = 0.17205149953305524
        self.assertAlmostEqual(expected, neural_net.mean_square_error(subrange))

    def test_next_epoch_accuracy_asin_stage7(self):
        neural_net = setup_neural()
        accuracies = [neural_net.next_epoch_accuracy(batch_size=2) for _ in range(10)]
        self.assert_list_almost_equal([0.0, 0.23076923076923078, 0.23076923076923078,
                                       0.23076923076923078, 0.23076923076923078, 0.23076923076923078,
                                       0.23076923076923078, 0.23076923076923078, 0.23076923076923078,
                                       0.3076923076923077],
                                      accuracies)

    def assert_list_almost_equal(self, expected: list[float], values: list[float]):
        self.assertEqual(len(expected), len(values))
        for i in range(len(expected)):
            self.assertAlmostEqual(expected[i], values[i], delta=0.000000000000001)