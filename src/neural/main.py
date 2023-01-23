from neural.provide_data import load_train_and_test_data
from neural.datasets import Datasets
from neural.one_layer_neural import OneLayerNeural, mse, cost_prime, sigmoid_prime
import numpy


def load_datasets() -> Datasets:
    raw_train, raw_test = load_train_and_test_data()
    return Datasets(raw_train, raw_test)


def setup_network_with_data() -> OneLayerNeural:
    datasets: Datasets = load_datasets()
    return OneLayerNeural(datasets, eta=0.1)


def main():
    neural_network = setup_network_with_data()
    neural_network.epoch(range(2))
    y1 = numpy.array([-1, 0, 1, 2])
    y2 = numpy.array([4, 3, 2, 1])
    print(f'[{mse(y1, y2)}]', cost_prime(y1, y2).tolist(), sigmoid_prime(y1).tolist(),
          f'[{neural_network.mean_square_error(range(2))}]')


if __name__ == '__main__':
    main()
