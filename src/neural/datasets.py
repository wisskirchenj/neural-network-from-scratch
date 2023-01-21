from numpy import ndarray, arange, zeros
from pandas import DataFrame


# pixel intensities range from 0 to 255
MAX_VALUE = 255


# scale the data to make them sensitive to the normalized sigmaoid function.
def scale(x_array: ndarray) -> ndarray:
    return x_array / MAX_VALUE


# One-Hot-Encoding of the categorical 'label' column (that assigns the actual category to the image record)
# The y_train set created here serves as benchmark to train the model, i.e. adapt the network weights and biases
# The y_test set finally can be used to evaluate the quality of the trained model when used on the test data.
def one_hot_encode(data: ndarray) -> ndarray:
    y_values = zeros((data.size, data.max(initial=0) + 1))
    rows = arange(data.size)
    y_values[rows, data] = 1
    return y_values


# A Datasets instance is feed with raw data, that are "enriched" in the init-method and the resulting x- and y-sets
# can be accessed by fields as well as the dimensions. Used as data access object by the neural network
class Datasets:

    def __init__(self, raw_train: DataFrame, raw_test: DataFrame):
        # 2nd to last column of an image data record contains the pixel data
        self.x_train = raw_train[raw_train.columns[1:]].to_numpy()
        self.x_test = raw_test[raw_test.columns[1:]].to_numpy()
        # the first label column of each data record contains the image's category as integer in [0,9]
        self.y_train = one_hot_encode(raw_train['label'].to_numpy())
        self.y_test = one_hot_encode(raw_test['label'].to_numpy())
        self.scale_x_sets()

    def scale_x_sets(self):
        self.x_train = scale(self.x_train)
        self.x_test = scale(self.x_test)

    # the record length of the data (here image data) is the number of input values (pixel data)
    # each input vale is connected to / feeds each neuron of the first layer of the neural network
    def get_input_dimension(self) -> int:
        return self.x_test.shape[1]

    # the output dimension represents the number of categories and equals the number of neurons
    # in the final network layer
    def get_output_dimension(self) -> int:
        return self.y_test.shape[1]

