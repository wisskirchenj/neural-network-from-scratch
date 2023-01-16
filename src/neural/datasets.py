from numpy import array


MAX_VALUE = 255


def scale(x_array: array) -> array:
    return x_array / MAX_VALUE


class Datasets:

    def __init__(self, x_train: array, x_test: array, y_train: array, y_test: array):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def scale_x_sets(self):
        self.x_train = scale(self.x_train)
        self.x_test = scale(self.x_test)
