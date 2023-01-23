from neural.provide_data import load_train_and_test_data
from neural.datasets import Datasets
from neural.one_layer_neural import OneLayerNeural


def load_datasets() -> Datasets:
    raw_train, raw_test = load_train_and_test_data()
    return Datasets(raw_train, raw_test)


def setup_network_with_data() -> OneLayerNeural:
    datasets: Datasets = load_datasets()
    return OneLayerNeural(datasets)


def main():
    neural_network = setup_network_with_data()
    print(neural_network.forward_step(range(2)))


if __name__ == '__main__':
    main()
