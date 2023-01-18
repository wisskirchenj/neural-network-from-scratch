import numpy as np

from neural.initialization import xavier, sigmoid


class OneLayerNeural:

    def __init__(self, n_features: int, n_classes: int):
        self.neuron_count = n_classes
        self.weights = xavier(n_features, n_classes, 3042022)
        self.bias = xavier(1, n_classes, 3042022).flatten()

    def forward(self, image_data: np.ndarray) -> list[float]:
        # Perform a forward step
        return [sigmoid(self.calc_neuron_input(image_data, n)) for n in range(self.neuron_count)]

    def calc_neuron_input(self, image_data: np.ndarray, neuron: int) -> float:
        summands = list(map(lambda x, y: x * y, image_data, self.weights[:, neuron]))
        return sum(summands, self.bias[neuron])
