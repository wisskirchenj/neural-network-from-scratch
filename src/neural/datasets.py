from numpy import ndarray


MAX_VALUE = 255


def scale(x_array: ndarray) -> ndarray:
    return x_array / MAX_VALUE


class Datasets:

    def __init__(self, x_train: ndarray, x_test: ndarray, y_train: ndarray, y_test: ndarray):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def scale_x_sets(self):
        self.x_train = scale(self.x_train)
        self.x_test = scale(self.x_test)
