# IDEA EDU Course

Graduate project implemented in the track 'Machine Learning Algorithms from Scratch' of hyperskill.org JetBrains Academy.

## Technology / External Libraries

- Python 3.11.1, 
- Pandas 1.5.2
- with packages: numpy, matplotlib, tqdm, requests, os, collections, unittest, unittest.mock

## Repository Contents

The sources of main project tasks (7 stages) and pytest testing.

## Installation

Install the wheel file using pip after clone/fork:

> $ pip install dist/neural_network_from_scratch-3.11.1-py3-none-any.whl

## Program description

The project is a first experience in every aspect of creating a fully connected neural network. We will code feedforward
networks and implement backpropagation and weight initialization. We will also create a training loop and visualize the
model's training process. We use Zalando-owned "Fashion MNIST" categorized image data, which are offered split into
separate train and test data sets. The project serves as a solid base to continue with neural network science.

Have fun!

## Project completion

Project was completed on 29.01.23

## Progress

15.01.23 Project started - gitHub repo and project structure setup.

15.01.23: Stage 1 completed - load Zalando-owned "Fashion MNIST" categorized image data as CSV over http, which are offered as
separate train and test data sets. Import the two data sets into Panda DataFrames and distribute them to x- and y-arrays
using One-Hot-Encoding. Implement a Xavier initialization and sigmoid function to initialize neural network weights.

18.01.23: Stage 2 completed - implement a single neuron layer network, initialize network weights and neuron biases using uniformly
distributed random numbers, start training the network - only implementing feedforward step. Have it dry-run with some
few image data.

23.01.23: Stage 3 completed - implement full epoch training cycles for the single neuron layer. An epoch cycle consists
of a forward step, a backward propagation and an update of the networks weights and biases. Iterative epochs train the
model on the training data set.

25.01.23: Stage 4 completed - accuracy measure of category recognition on test data set implemented. Convenience method 
implemented on network to run an epoch on full train data with parameterizable batch size and return accuracy afterwards.
Have 20 training iterations and track and plot results with matplotlib.

28.01.23: Stage 5 completed - setup of a **two layer neural network** with a hidden layer of (default) 64 neurons. This
stage does a forward step with 2 image records of the training set through the network.

29.01.23: Stage 6 completed - run an epoch-cycle for the **two layer neural network** and consider the mean square error.
Use inheritance to derive the 1-layer and 2-layer network from a common base class with implemented methods for both.

29.01.23: Final Stage 7 completed - do full 20 epochs on the 60k train data and measure on test data after
each epoch. Setup is with batch_size=100 and eta=0.5 as with the 1-layer neural in stage 4. The results improve
form accuracy of 85.5 % with 1 layer to 88% with 2 layers, while simultaneously increasing runtime by
about a factor of 4. (~ 13 sec per epoch now)
