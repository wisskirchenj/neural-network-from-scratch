# IDEA EDU Course

Graduate project implemented in the track 'Machine Learning Algorithms from Scratch' of hyperskill.org JetBrains Academy.

## Technology / External Libraries

- Python 3.11.1, 
- Pandas 1.5.2
- with packages: numpy, matplotlib, requests, os, collections, unittest, unittest.mock

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

[//]: # (Project was completed on 08.01.23)

## Progress

15.01.23 Project started - gitHub repo and project structure setup.

15.01.23: Stage 1 completed - load Zalando-owned "Fashion MNIST" categorized image data as CSV over http, which are offered as
separate train and test data sets. Import the two data sets into Panda DataFrames and distribute them to x- and y-arrays
using One-Hot-Encoding. Implement a Xavier initialization and sigmoid function to initialize neural network weights.

18.01.23: Stage 2 completed - implement a single neuron layer, initialize network weights and neuron biases using uniformly
distributed random numbers, start training the network - only implementing feedforward step. Have it dry-run with some
few image data.
