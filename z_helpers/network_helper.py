import pandas as pd
import random
import numpy as np
from abc import ABC, abstractmethod


USE_BIAS = False
def train_validate_split(data_y, val_ratio=0.2, shuffle=True):
    """
    Splits a dataset into training and validation sets based on the specified ratio
    for each class to maintain class distribution balance across both sets,
    without shuffling the original data order.

    Parameters:
    data_y (numpy.ndarray): An array or list containing class labels for each sample in the dataset.
    val_ratio (float, optional): The proportion of the dataset to include in the validation split.
                                Defaults to 0.2 (20% of the data).

    Returns:
    tuple of lists: A tuple containing two lists:
                   - train_indices (list): Indices of the samples designated for the training set.
                   - val_indices (list): Indices of the samples designated for the validation set.
    """
    def samples_per_class(data_y, indices):
        """
        Calculate the number of samples for each class in a specified subset of a dataset.
        """
        class_count = [0]*10
        for i in indices:
            class_count[data_y[i]] += 1 
        return class_count 
    
    sample_num = len(data_y)
    overall_indices = list(range(sample_num))  # Ordered indices
    overall_class_num = samples_per_class(data_y, overall_indices)
    val_class_num = [int(num*val_ratio) for num in overall_class_num]
    tmp_val_class_num = [0]*10
    
    if shuffle:
        random.shuffle(overall_indices)
    
    train_indices = []
    val_indices = []
    
    for idx in overall_indices:
        tmp_label = data_y[idx]
        if tmp_val_class_num[tmp_label] < val_class_num[tmp_label]:
            val_indices.append(idx)
            tmp_val_class_num[tmp_label] += 1
        else:
            train_indices.append(idx)
    
    return train_indices, val_indices

class DataLoader:
    def __init__(self, X, Y, batch_size, sample_indices, shuffle=True):
        """
        Initializes the DataLoader with data and configuration.

        Args:
        X (numpy.ndarray): The image data array, shape (n_samples, 784).
        Y (numpy.ndarray): The labels array, shape (n_samples,).
        batch_size (int): The number of samples per batch.
        sample_indices (list): The sampled indices included in the dataset.
        shuffle (bool): Whether to shuffle the data before creating batches (default: True).
        """
        self.X = X  # The input data array (data)
        self.Y = Y  # The labels array (labels)
        if len(self.X) != len(self.Y):
            raise ValueError("X and Y must have the same size")
        self.batch_size = batch_size  # Size of each mini-batch
        self.shuffle = shuffle  # Whether to shuffle the data at the start of each epoch
        self.current_index = 0  # Tracks the current position in the data for batching
        
        # If sample_indices are provided, use them to filter the data subset; 
        # otherwise, use the entire dataset by default
        if sample_indices:
            self.indices = sample_indices
        else:
            self.indices = np.arange(X.shape[0])  # Use all indices if no subset is specified

    # Your code: implement the function
    def __iter__(self):
        """
        Resets the iterator and shuffles the data if needed.
        
        This method is called at the beginning of a new iteration (e.g., in a for-loop).
        """
        # Shuffle the indices if needed
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        # Load relevant data
        data = [self.X[index] for index in self.indices]
        labels = [self.Y[index] for index in self.indices]
        
        # Compute batches
        self.data_batch = []
        size = self.batch_size
        x_batch = []
        y_batch = []

        for x, y in zip(data, labels):
            x_batch.append(x)
            y_batch.append(y)
            
            size -= 1
            if size <= 0:
                self.data_batch.append([np.array(x_batch), np.array(y_batch)])
                x_batch, y_batch, size = [], [], self.batch_size
        
        # End conditions
        if len(x_batch) > 0:
            self.data_batch.append([np.array(x_batch), np.array(y_batch)])
        
        self.current = 0    
        self.length = len(self.data_batch)
        return self
        

    # Your code: implement the function
    def __next__(self):
        """
        Returns the next batch of data and labels.

        This method is called repeatedly by the iterator to get the next mini-batch.
        
        Raises:
        StopIteration: If there are no more data samples left to return.
        """
        
        if self.current < self.length:
            batch = self.data_batch[self.current]
            self.current += 1
            return batch
        else:
            raise StopIteration
        
class Tensor:
    """
    Represents a tensor object that is used in automatic differentiation.

    This class is designed to store data and associated operations for forward
    and backward passes in a computational graph. It supports operations on 
    vectors using Numpy arrays.

    Attributes:
        track_gradients (bool): Class level flag to track or not track gradients.
    """

    track_gradients = True  # Class variable to track or not track gradients

    def __init__(self, data, children=[], op=None):
        """
        Initializes a Tensor object.

        Args:
            data (np.array): The forward vector data stored as a Numpy array.
            children (list, optional): List of tensors that are predecessors in the 
                                       computational graph.
            op (Operation, optional): The operation that produced this tensor, used 
                                      during the backward pass.
        """
        self.data = data  # Holds the forward vector data as a Numpy array
        self.grad = 0  # Initializes the gradient to zero
        self.prev = children  # List of previous Tensor objects in the graph
        self.op = op  # Operation that produced this Tensor

    def backward(self, grad=None):
        """
        Performs the backward pass to compute gradients of the tensor with respect to
        some scalar value.

        Args:
            grad (np.array, optional): External gradient passed to this tensor. If None,
                                       gradients are initialized to match the shape of
                                       the tensor's data and set to ones.
        """
        if grad is None:
            self.grad = np.ones_like(self.data)  # Set gradient to one for scalar outputs
        else:
            self.grad = grad  # Use the externally provided gradient

        nodes = [self]  # Start with the current node
        while nodes:
            current_node = nodes.pop()  # Process one node at a time
            if current_node.op:
                current_node.op.backward(current_node)  # Call backward method of the operation
            for input_node in current_node.prev:
                nodes.append(input_node)  # Add input nodes to the list to process them
                
class TensorFunction(ABC):
    """
    Abstract base class for defining tensor operations.
    
    This class is designed to be inherited by specific tensor operation classes, which must
    implement the forward and backward methods. These methods define how the operation is
    performed in the forward pass and how gradients are handled in the backward pass.
    """

    @abstractmethod
    def forward(self, *args):
        """
        Computes the forward pass of the operation.

        This method must be implemented by all subclasses to perform the specific operation.
        It takes variable number of tensor inputs and computes the result of the operation.

        Args:
            *args: A variable-length argument list of tensors compatible with the operation.

        Returns:
            The result of the tensor operation as part of the forward computation.
        """
        pass

    @abstractmethod
    def backward(self, *args):
        """
        Computes the backward pass of the operation.

        This method must be implemented by all subclasses to handle the propagation of gradients
        through this operation. It is called during the backward pass of the automatic differentiation.

        Args:
            *args: A variable-length argument list, typically including gradients along with the
                   tensors involved in the forward operation.
        """
        pass

    def __call__(self, *args):
        """
        Allows the object to be called like a function, using the defined forward method.

        This makes instances of subclasses callable, simplifying usage by abstracting the
        method call. When the instance is called, it automatically triggers the `forward` method.

        Args:
            *args: A variable-length argument list passed to the `forward` method.

        Returns:
            The result of the forward method.
        """
        return self.forward(*args)  # Delegate to the forward method
    
class Add(TensorFunction):
    """
    Implements the addition operation for tensors.

    This class inherits from `TensorFunction` and defines how tensors are added in the forward
    pass and how gradients are propagated back in the backward pass. It supports operations on 
    tensors that may or may not track gradients.
    """

    def forward(self, X, Y):
        """
        Performs the forward pass of the addition operation.

        Args:
            X (Tensor): The first operand tensor.
            Y (Tensor): The second operand tensor.

        Returns:
            Tensor: The result of adding X and Y, with appropriate graph connections if gradients
                    are being tracked.
        """
        if Tensor.track_gradients:
            # Create a new tensor that is the sum of X and Y, tracking the operation and parents in the graph.
            Z = Tensor(X.data + Y.data, children=[X, Y], op=self)
        else:
            # Create a new tensor that is the sum of X and Y without tracking the graph.
            Z = Tensor(X.data + Y.data)
        return Z

    def backward(self, dZ):
        """
        Performs the backward pass of the addition operation.

        This method propagates gradients from the result tensor to the operand tensors.

        Args:
            dZ (Tensor): The derivative of the output tensor with respect to some upstream gradient.
        """
        # Retrieve the tensors involved in the forward operation.
        X = dZ.prev[0]
        Y = dZ.prev[1]

        # Update gradients of X and Y by adding the gradient of the output tensor, respecting existing gradients.
        X.grad += dZ.grad
        Y.grad += dZ.grad
        
class MatMul(TensorFunction):
    # Your code: implement the function
    def forward(self, X, Y):
        if Tensor.track_gradients:
            Z = Tensor(X.data @ Y.data, children=[X, Y], op=self)
        else:
            Z = Tensor(X.data @ Y.data)
        # print(np.max(Z.data))
        return Z
    
    # Your code: implement the function
    def backward(self, dZ):
        X = dZ.prev[0]
        Y = dZ.prev[1]
        
        X.grad += dZ.grad @ Y.data.T
        Y.grad += X.data.T @ dZ.grad

class ReLU(TensorFunction):
    # Your code: implement the function
    def forward(self, X):
        
        array = np.maximum(X.data, 0)
        if Tensor.track_gradients:
            Z = Tensor(np.array(array), children=[X], op=self)
        else:
            Z = Tensor(np.array(array))
        return Z
    
    # Your code: implement the function
    def backward(self, dZ):
        X = dZ.prev[0]
        
        X.grad += np.array(np.where(X.data > 0, dZ.grad, 0))
        
class LinearLayer:
    """
    Represents a linear layer in a neural network with optional activation.

    This layer computes a linear transformation on the input data followed by an activation function. 
    It initializes weights and biases uniformly within a range determined by the input dimension size, 
    to help in stabilizing the learning process.
    """

    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        Initializes the LinearLayer with specified input and output dimensions and activation.

        Args:
            input_dim (int): Dimensionality of the input features.
            output_dim (int): Dimensionality of the output features.
            activation (str, optional): Type of activation to use; defaults to 'relu'.
        """
        # Initialize weights and biases within a range that depends on the input dimension.
        # This is a common heuristic used to help in the convergence of training.
        limit = 1 / np.sqrt(input_dim)
        self.weights = Tensor(np.random.uniform(-limit, limit, (input_dim, output_dim)))
        self.bias = Tensor(np.random.uniform(-limit, limit, (1, output_dim)))
        if USE_BIAS:
            self.param = [self.weights, self.bias]
        else:
            self.param = [self.weights]

        # Initializing operations for matrix multiplication and addition.
        self.matmul = MatMul()
        self.add = Add()

        # Set the activation function; default is ReLU.
        self.activation = ReLU() if activation == 'relu' else None

    # Your code: implement the function
    def forward(self, x):
        """
        Computes the forward pass of the linear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the linear transformation and activation function.
        """         
        A =  self.matmul(x, self.weights)
        if USE_BIAS:
            Z = self.add(A, self.bias)
        else:
            Z = A
            
        if self.activation:
            return self.activation(Z)
        else:
            return Z
        
        
class MLP:
    """
    Represents a flexible Multi-Layer Perceptron (MLP) with any number of hidden layers.

    Attributes:
        layers (list): A list of LinearLayer objects forming the MLP.
        param (list): Aggregated list of parameters from all layers for training.
    """

    def __init__(self, layer_dims):
        """
        Initializes the MLP with the given architecture.

        Args:
            layer_dims (tuple or list of int): Sizes of each layer including input and output.
                                               Example: (784, 128, 64, 10)
        """
        self.layers = []

        for in_dim, out_dim in zip(layer_dims[:-2], layer_dims[1:-1]):
            self.layers.append(LinearLayer(in_dim, out_dim))  # Hidden layers with activation

        # Add final output layer with no activation
        self.layers.append(LinearLayer(layer_dims[-2], layer_dims[-1], activation=""))

        # Flatten all parameters into a single list
        self.param = np.array([layer.param for layer in self.layers]).flatten()
        self.last_acts = self.reset_activations()


    def forward(self, x):
        """
        Runs the network and stores intermediate activations.

        Returns
        -------
        Tensor : output of the final layer
        """
        # print(self.last_acts)
        for i, layer in enumerate(self.layers):
            # print(x.data.shape)
            self.last_acts[i].append(x)  
            x = layer.forward(x)
        return x
    
    def batch_activation_count(self, threshold=0.0):
        """
        For the most-recent forward pass, return the total count of activated neurons for each layer.
        The count is the total number of activations that exceed the threshold in each layer, over the entire dataset.
        """
        total_active_counts = []
        # Iterate over each layer's activations
        for act in self.last_acts:
            # print(len(act), len(act[0].data), type(act[0].data))
            
            non_zeros = 0
            for tensor in act:
                arr = np.array(tensor.data)
                non_zeros += np.count_nonzero(arr)
            non_zeros /= len(act) * len(act[0].data)
                
            # Sum across all samples in the batch for the layer
            total_active_counts.append(non_zeros)  # Total active neurons for the layer

        return total_active_counts
    
    def reset_activations(self):
        """Clear stored activations for a new epoch (or whenever)."""
        self.last_acts = [[] for _ in self.layers]


    
class SGDOptimizer:
    """
    Implements the Stochastic Gradient Descent (SGD) optimization algorithm.

    This optimizer updates the parameters of a model by moving them in the direction of
    the negative gradient of the loss function, scaled by a learning rate. This approach
    is one of the most common methods used for training various types of neural networks.
    """

    def __init__(self, parameters, lr=0.01):
        """
        Initializes the SGDOptimizer with given parameters and learning rate.

        Args:
            parameters (list): List of parameters to optimize.
            lr (float, optional): Learning rate for the optimizer. Default is 0.01.
        """
        self.parameters = parameters  # Model parameters to be optimized
        self.lr = lr  # Learning rate for optimization

    def step(self):
        """
        Performs a single optimization step on parameters.

        This method iterates through all parameters and updates them based on their
        gradients and the learning rate. It adjusts for parameter dimensions as needed.
        """
        for param in self.parameters:
            # Check if the parameter and its gradient dimensions require summing (i.e. Bias)
            if param.data.shape[0] == 1 and param.grad.shape[0] > 1:
                print(f"Sum grad : {np.mean(np.sum(param.grad, axis=0, keepdims=True))}")
                # Apply the learning rate and update the parameter for broadcast compatibility
                param.data -= self.lr * np.sum(param.grad, axis=0, keepdims=True)
            else:
                # print(f"Regular update param :, mean : {np.mean(param.grad)}, max= {np.max(param.grad)}, min= {np.min(param.grad)}")
                # Standard parameter update
                param.data -= self.lr * param.grad  # Update parameter based on its gradient

    def zero_grad(self):
        """
        Resets all parameter gradients to zero.

        This method is typically called after each training step to prevent accumulation
        of gradients across multiple passes.
        """
        for param in self.parameters:
            param.grad = 0  # Reset the gradient for each parameter to zero

class SoftmaxCrossEntropy(TensorFunction):
    """
    Implements the softmax activation followed by the cross-entropy loss function.

    This class provides a method to perform a stable calculation of the softmax activation 
    and compute the cross-entropy loss between the predicted probabilities and the actual labels.
    It is particularly useful for training classification models where the outputs are 
    probabilities that sum to one.
    """

    def forward(self, outputs, labels):
        """
        Computes the forward pass using softmax for predictions and cross-entropy for loss.

        Args:
            outputs (Tensor): The logits predicted by the model (before softmax).
            labels (Tensor): The true labels, one-hot encoded.

        Returns:
            Tensor: The mean cross-entropy loss as a tensor.
        """
        # Perform a stable softmax computation to avoid numerical overflow.
        exps = np.exp(outputs.data - np.max(outputs.data, axis=1, keepdims=True))
        self.softmax = exps / (np.sum(exps, axis=1, keepdims=True) + 1e-8)

        # Compute the cross-entropy loss.
        cross_entropy = -np.sum(labels.data * np.log(self.softmax + 1e-8), axis=1)
        mean_loss = np.mean(cross_entropy)

        # Wrap the mean loss in a Tensor, tracking the computation graph if necessary.
        if Tensor.track_gradients:
            loss = Tensor(mean_loss, children=[outputs, labels], op=self)
        else:
            loss = Tensor(mean_loss)
        return loss

    def backward(self, dL):
        """
        Backpropagates the error through the softmax and cross-entropy loss.

        Args:
            dL (Tensor): The gradient tensor of the loss with respect to the output of this operation.

        Modifies:
            outputs.grad (np.array): Updated gradient of the outputs as computed by the backward pass.
        """
        outputs = dL.prev[0]
        labels = dL.prev[1]
        batch_size = outputs.data.shape[0]
        # Compute the gradient of the loss with respect to the input.
        outputs.grad = (self.softmax - labels.data) / batch_size

class no_grad:
    def __enter__(self):
        # Save the current state (whether gradients are currently being tracked)
        self.prev_state = Tensor.track_gradients
        # Disable gradient tracking
        Tensor.track_gradients = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original gradient tracking state
        Tensor.track_gradients = True
        
# Your code: implement the function
def validation_func(network, val_dataloader):
    """
    Computes the validation accuracy of a neural network.

    This function iterates through a validation dataset using a dataloader,
    calculates the number of correct predictions the network makes, and computes
    the overall accuracy of the network on this validation set.

    Args:
        network: The neural network model to validate.
        val_dataloader: The dataloader that provides batches of validation data.

    Returns:
            - accuracy (float): The proportion of correct predictions.
            - correct_num (int): Total number of correct predictions.
            - total_num (int): Total number of samples in the validation set.
    """
    with no_grad():
        correct_num = 0
        total_num = 0
        for batch_x, batch_y in val_dataloader:
            total_num += batch_y.shape[0]
            output = network.forward(batch_x)
            output_label = np.array([np.argmax(outs) for outs in output.data])
            
            correct_num += sum([1 if label == y else 0 for label, y in zip(output_label, batch_y)])
        
        return float(correct_num/total_num), correct_num, total_num

# Your code: implement the function
def one_hot_encode(labels, num_classes=10):
    """
    Convert an array of numeric labels into a one-hot encoded matrix.

    Args:
        labels (np.array): Array of integer labels.
        num_classes (int, optional): Total number of classes.

    Returns:
        np.array: A matrix of one-hot encoded vectors where each row corresponds to a one-hot vector.
    """
    return [[1 if i == label else 0 for i in range(num_classes)] for label in labels]
    
# Your code: implement the function
def train_func(network, train_dataloader, val_dataloader, optimizer, loss_func, epoch_num, act_threshold = 0.0):
    """
    Trains a neural network using the provided training and validation data loaders.

    This function orchestrates the training process over a specified number of epochs,
    updating model parameters with an optimizer, and evaluates the model performance
    on a validation dataset after each epoch. It records and returns the validation 
    accuracies for each epoch.

    Args:
        network: The neural network model to be trained.
        train_dataloader (iterable): DataLoader providing batches of training data.
        val_dataloader (iterable): DataLoader for providing batches of validation data.
        optimizer: The optimization algorithm (e.g., SGD) to update network weights.
        loss_func: The loss function to be used for training evaluation.
        epoch_num (int): The number of training epochs.

    Returns:
        np.array: An array of validation accuracies for each training epoch.
    """
    validation_acc = []
    training_acc = []
    epoch_act_means = []         # list of list: epoch‑>layer‑>mean‑frac‑active
    epoch_act_totals  = []             # epochs × n_layers  – active counts
    epoch_neuron_tot  = []  
    
    for epoch in range(epoch_num):
        i = 0
        network.reset_activations()

        for x, y in train_dataloader:
            # print(i)
            # print(f"x shape: {x.shape}")
            output = network.forward(Tensor(x))

            labels = Tensor(one_hot_encode(y, num_classes=10))
            loss = loss_func.forward(output, labels)
            if np.isnan(loss.data):
                print("skipping loss {}", loss)
                pass
            i += 1
            loss_func.backward(loss)
            output.backward(output.grad)
            
            optimizer.step()
            optimizer.zero_grad()
            
        mean_activations = network.batch_activation_count(threshold=0.0)
        
        with no_grad():
            train_acc, _, _ = validation_func(network, train_dataloader)
            training_acc.append(train_acc)
            
            val_acc, _, _ = validation_func(network, val_dataloader)
            validation_acc.append(val_acc)

            print(
            f"epoch {epoch:02d} | "
            f"train {train_acc:.4f} | val {val_acc:.4f} | "
            f"active {mean_activations}"
            f"loss {loss.data}"
        )

    return (
        np.array(training_acc),
        np.array(validation_acc),
        np.array(epoch_act_means)
    )

class InfiniteDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iterator = iter(data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.data_loader)
            batch = next(self.iterator)
        return batch

def get_total_batches(batch_size, data_indices):
    # Calculate total batches for the data
    total_samples = len(data_indices)
    total_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
    return total_batches