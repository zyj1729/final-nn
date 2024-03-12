![BuildStatus](https://github.com/zyj1729/final-nn/actions/workflows/main.yml/badge.svg?event=push)
# Neural Network Implementations

This repository contains the implementation of a fully-connected Neural Network class and its applications in two distinct examples: an autoencoder for dimensionality reduction on the digits dataset and a classifier for transcription factor binding sites.

## NeuralNetwork Class

The `NeuralNetwork` class is a customizable Python class designed for building and training fully-connected neural networks. The class supports various functionalities including forward pass, backpropagation, parameter updates, and training with early stopping.

### Key Features:

- Customizable architecture: Users can define the network architecture including the number of layers, units per layer, and activation functions.
- Loss functions: Supports mean squared error and binary cross entropy loss functions.
- Early stopping: To prevent overfitting, early stopping is implemented based on validation loss.


## Examples

### Running the Examples

For the autoencoder and the transcription factor binding site classifier, open and run the cells in `Autoencoder and Classifier.ipynb`.


Make sure to adjust the hyperparameters according to your needs and computational resources.


### 1. Autoencoder for Dimensionality Reduction

#### Background

An autoencoder is a type of neural network used to learn efficient codings of unlabeled data. The network consists of two main parts: an encoder that reduces the input data into a smaller, dense representation, and a decoder that reconstructs the input data from this dense representation.

#### Task

The task is to train a `64x16x64` autoencoder on the digits dataset from `sklearn.datasets`. The autoencoder first compresses a 64-dimensional input (8x8 image of a digit) to a 16-dimensional latent space and then reconstructs the original input from this compressed form.

#### Implementation

The digits dataset is loaded and split into training and validation sets. The `NeuralNetwork` class is then used to define and train the autoencoder. The training process is monitored by plotting the training and validation losses, and the performance of the autoencoder is evaluated by calculating the average reconstruction error over the validation set.

### 2. Classifier for Transcription Factor Binding Sites

#### Background

Transcription factors are proteins that bind to specific DNA sequences to control the flow of genetic information from DNA to mRNA. Identifying these binding sites (motifs) is crucial for understanding gene regulation.

#### Task

The task is to implement a multi-layer fully connected neural network to predict whether a short DNA sequence is a binding site for the yeast transcription factor Rap1. The training data is imbalanced, with significantly fewer positive (binding) sequences compared to negative (non-binding) sequences. A sampling scheme is implemented to ensure balanced training.

#### Implementation

Positive Rap1 motif examples and negative examples are read from provided files, and the negative examples are processed to match the length of positive examples. The `sample_seqs` function is used to balance the classes, and the `one_hot_encode_seqs` function is used to convert DNA sequences to a numerical format suitable for the neural network. The `NeuralNetwork` class is then used to define, train, and evaluate the classifier. The model's performance is assessed based on accuracy over a validation set.

## Usage

To use the `NeuralNetwork` class or run the example implementations, clone this repository and open the respective Jupyter Notebooks. Make sure you have the required dependencies installed, including `numpy`, `matplotlib` (for plotting), and `sklearn` (for the digits dataset).

