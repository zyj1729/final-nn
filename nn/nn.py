# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike
import copy

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str, 
        patience: int, 
        progress: int
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size
        self._patience = patience
        self._progress = progress

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        Z_curr = np.dot(A_prev, W_curr.T) + b_curr.T
        if activation == "sigmoid":
            A_curr = self._sigmoid(Z_curr)
        elif activation == "relu":
            A_curr = self._relu(Z_curr)
        else:
            raise Error("Activation function not defined")
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # Initialize the input layer activation with the input matrix
        A_curr = X
        # Initialize a cache to store intermediate Z and A values
        cache = {}

        # Iterate through each layer defined in the network architecture
        for i in range(len(self.arch)):
            # Get the layer information from the architecture
            l = self.arch[i]
            # Extract the current layer's weights
            W_curr = self._param_dict["W" + str(i + 1)]
            # Extract the current layer's biases
            b_curr = self._param_dict["b" + str(i + 1)]
            # Extract the current layer's activation function
            activation = l["activation"]

            # Set the activations from the previous layer as inputs to the current layer
            A_prev = A_curr
            # Perform the single forward pass for the current layer
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)

            # Store the current layer's Z value in cache for backpropagation
            cache['Z' + str(i + 1)] = Z_curr
            # Store the current layer's A value in cache for use in the next layer's forward pass
            cache['A' + str(i + 1)] = A_curr

        # Return the final layer's activations and the cache containing all intermediate values
        return A_curr, cache
            

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # Determine the derivative of the activation function based on the current layer's activation type
        if activation_curr == "sigmoid":
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr == "relu":
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        else:
            # Raise an error if the activation function is not supported
            raise Error("Activation function not defined")

        # Calculate the partial derivative of loss with respect to the current layer's weights
        dW_curr = np.dot(dZ_curr.T, A_prev)
        # Calculate the partial derivative of loss with respect to the current layer's biases
        db_curr = np.sum(dZ_curr, axis = 0).reshape(b_curr.shape)

        # Calculate the partial derivative of loss with respect to the activations of the previous layer
        dA_prev = np.dot(dZ_curr, W_curr)
        # Return the gradients with respect to the activations of the previous layer, the current layer's weights, and biases
        return dA_prev, dW_curr, db_curr
        

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backpropagation of the whole fully connected neural network.

        Args:
            y (ArrayLike):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values from the forward pass.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the most recent forward pass,
                specifically A (activation) and Z (linear combination) matrices for each layer.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this backpropagation pass,
                including gradients with respect to weights (dW) and biases (db) for each layer.
        """
        # Initialize a dictionary to store the gradients
        grad_dict = {}

        # Compute the gradient of the loss function with respect to the activation of the last layer
        if self._loss_func == "binary_cross_entropy":
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func == "mean_squared_error":
            dA_curr = self._mean_squared_error_backprop(y, y_hat)
        else:
            # Raise an error if the loss function is not supported
            raise Error("Loss function not defined")

        # Iterate backwards through the layers of the network to compute gradients
        for i in reversed(range(1, len(self.arch) + 1)):
            # Retrieve the current layer's parameters from the parameter dictionary
            W_curr = self._param_dict['W' + str(i)]
            b_curr = self._param_dict['b' + str(i)]
            # Retrieve the linear combination matrix (Z) for the current layer from the cache
            Z_curr = cache['Z' + str(i)]
            # Retrieve the activation matrix (A) of the previous layer from the cache, or use y_hat for the first layer
            A_prev = cache['A' + str(i - 1)] if i - 1 > 0 else y_hat 

            # Perform a single backpropagation pass for the current layer
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, self.arch[i - 1]['activation'])

            # Store the computed gradients in the gradient dictionary
            grad_dict['dW' + str(i)] = dW_curr
            grad_dict['db' + str(i)] = db_curr

            # Update dA_curr for the next iteration (previous layer in the network)
            dA_curr = dA_prev

        # Return the dictionary containing all computed gradients
        return grad_dict
    
    def _compute_loss(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        if self._loss_func == 'mean_squared_error':
            return self._mean_squared_error(y, y_hat)
        elif self._loss_func == 'binary_cross_entropy':
            return self._binary_cross_entropy(y, y_hat)
        else:
            raise ValueError("Loss function not defined")

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for i in range(1, len(self.arch) + 1):
            self._param_dict['W' + str(i)] -= self._lr * grad_dict['dW' + str(i)]
            self._param_dict['b' + str(i)] -= self._lr * grad_dict['db' + str(i)]

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        # Initalize loss lists
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        
        best_val_loss = np.inf
        epochs_no_improve = 0
        patience = self._patience
        progress = self._progress

        # Compute the number of batches in a given epoch
        num_batches = np.ceil(len(y_train) / self._batch_size)

        # Loop over epochs
        for e in range(self._epochs):

            # Shuffle the training data
            shuffle = np.random.permutation(len(y_train))
            shuffled_X_train = X_train[shuffle]
            shuffled_y_train = y_train[shuffle]

            # Create batches
            X_batches = np.array_split(shuffled_X_train, num_batches)
            y_batches = np.array_split(shuffled_y_train, num_batches)

            # Initialize per-batch loss for this epoch
            train_losses = []

            # Create and go through batches
            for X_batch, y_batch in zip(X_batches, y_batches):

                # Do forward pass and append loss
                output, cache = self.forward(X_batch)
                if self._loss_func == "mean_squared_error":
                    train_losses.append(self._mean_squared_error(y_batch, output))
                elif self._loss_func == "binary_cross_entropy":
                    train_losses.append(self._binary_cross_entropy(y_batch, output))

                # Backpropagate and update parameters
                grad_dict = self.backprop(y_batch, output, cache)
                self._update_params(grad_dict)

            # Compute average training loss for this epoch
            per_epoch_loss_train.append(np.mean(train_losses))

            # Compute validation loss for this epoch
            pred = self.predict(X_val)
            if self._loss_func == "mean_squared_error":
                val_loss = self._mean_squared_error(y_val, pred)
                per_epoch_loss_val.append(val_loss)
            elif self._loss_func == "binary_cross_entropy":
                val_loss = self._binary_cross_entropy(y_val, pred)
                per_epoch_loss_val.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_params = copy.deepcopy(self._param_dict)
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {e + 1} epochs.")
                self._param_dict = best_params
                break

            if e % progress == 0:
                print(f"Finished epoch {e + 1} of {self._epochs}.")
        
        # Return loss lists
        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X)
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        return dA * self._sigmoid(Z) * (1 - self._sigmoid(Z))

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        return dA * (Z > 0).astype(int)

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        y_hat = np.clip(y_hat, 0.00001, 0.99999)
        loss = - np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        y_hat = np.clip(y_hat, 0.00001, 0.99999)
        dA = (- (y / y_hat) + (1 - y) / (1 - y_hat)) / len(y)
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        loss = np.mean((y - y_hat) ** 2)
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        m = y.shape[1]
        dA = -2 * (y - y_hat) / m
        return dA