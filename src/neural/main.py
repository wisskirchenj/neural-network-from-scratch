import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from neural.provide_data import load_train_and_test_data
from neural.datasets import Datasets
from neural.one_layer_neural import OneLayerNeural
from neural.two_layer_neural import TwoLayerNeural


def plot(loss_history: list, accuracy_history: list, filename='plot'):
    # function to visualize learning process at stage 4
    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel('Epoch number')
    plt.ylabel('Cost')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Cost on train dataframe vs epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe vs epoch')
    plt.grid()
    plt.savefig(f'{filename}.png')


def load_datasets() -> Datasets:
    raw_train, raw_test = load_train_and_test_data()
    return Datasets(raw_train, raw_test)


def setup_network_with_data(eta=0.1) -> OneLayerNeural:
    datasets: Datasets = load_datasets()
    return OneLayerNeural(datasets, eta=eta)


def main_stage4():
    neural_network = setup_network_with_data(eta=0.5)
    print(f'[{neural_network.accuracy_in_test()}]')
    costs = []
    accuracies = []
    for _ in tqdm(range(20)):
        print(neural_network.next_epoch_accuracy())
        costs.append(neural_network.mean_square_error(range(60000)))
        accuracies.append(neural_network.accuracy_in_test())
    plot(costs, accuracies)


def main():
    neural_network = TwoLayerNeural(load_datasets())
    print(neural_network.forward_step(range(2)).flatten().tolist())


if __name__ == '__main__':
    main()
