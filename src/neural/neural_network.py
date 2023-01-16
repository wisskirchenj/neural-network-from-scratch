from provide_data import load_train_and_test_data
from datasets import Datasets
from initialization import xavier, sigmoid

import numpy as np


def create_y_array_from_labels(data: np.ndarray) -> np.ndarray:
    y_values = np.zeros((data.size, data.max(initial=0) + 1))
    rows = np.arange(data.size)
    y_values[rows, data] = 1
    return y_values


def load_datasets() -> Datasets:
    raw_train, raw_test = load_train_and_test_data()

    x_train = raw_train[raw_train.columns[1:]].to_numpy()
    x_test = raw_test[raw_test.columns[1:]].to_numpy()
    y_train = create_y_array_from_labels(raw_train['label'].to_numpy())
    y_test = create_y_array_from_labels(raw_test['label'].to_numpy())

    return Datasets(x_train, x_test, y_train, y_test)


def main():
    datasets: Datasets = load_datasets()
    datasets.scale_x_sets()
    print(
        [datasets.x_train[2, 778], datasets.x_test[0, 774]],
        xavier(2, 3).flatten().tolist(),
        [sigmoid(i) for i in range(-1, 3)]
    )


if __name__ == '__main__':
    main()