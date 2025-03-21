import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from random import shuffle, randint
from PIL import Image
import math
from abc import ABC, abstractmethod

# region MANUAL LOADER
def train_validate_split(data_y, val_ratio=0.2):
    """
    Splits a dataset into training and validation sets based on the specified ratio
    for each class to maintain class distribution balance across both sets.

    The function ensures that each class is represented in the validation set
    proportionally to the specified validation ratio. This stratified approach helps in
    maintaining a consistent class distribution between the training and validation datasets.

    Parameters:
    data_y (numpy.ndarray): An array or list containing class labels for each sample in the dataset.
    val_ratio (float, optional): The proportion of the dataset to include in the validation split. This value should be
                                 a float between 0 and 1 indicating the percentage of data to be used as validation.
                                 Defaults to 0.2 (20% of the data).

    Returns:
    tuple of lists: A tuple containing two lists:
                     - train_indices (list): Indices of the samples designated for the training set.
                     - val_indices (list): Indices of the samples designated for the validation set.
    """
    def samples_per_class(data_y, indices):
        """
        Calculate the number of samples for each class in a specified subset of a dataset.

        Parameters:
        data_y (numpy.ndarray): An array containing class labels for each sample in the dataset.
        indices (list): An array or list of indices specifying which samples to consider for the count.

        Return:
        list: A list of integers where each index corresponds to a class label (0 through 9), and the value at each index
            indicates the number of samples of that class present in the specified subset of `data_y`.
        """
        class_count = [0]*10
        for i in indices:
            class_count[data_y[i]] += 1 
        return class_count 
    sample_num = len(data_y)
    overall_indices = [num for num in range(sample_num)]
    overall_class_num = samples_per_class(data_y, overall_indices)
    val_class_num = [int(num*val_ratio) for num in overall_class_num]
    tmp_val_class_num = [0 for num in range(10)]
    shuffle(overall_indices)
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
        self.param = [self.weights, self.bias]

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
        Z = self.add(A, self.bias)
        
        if self.activation:
            return self.activation(Z)
        else:
            return Z
        
        
class MLP:
    """
    Represents a simple Multi-Layer Perceptron (MLP) with two fully-connected layers.

    This class defines a neural network with one hidden layer and one output layer,
    both fully connected. The hidden layer includes an activation function (ReLU),
    while the output layer doesn't have an activation, making it suitable for classification.

    Attributes:
        hidden_layer (LinearLayer): The hidden layer of the MLP.
        output_layer (LinearLayer): The output layer of the MLP.
        param (list): Aggregated list of parameters from both layers for training purposes.
    """

    # Your code: implement the function
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the MLP with specified dimensions for each layer.

        Args:
            input_dim (int): The size of the input feature vector.
            hidden_dim (int): The number of neurons in the hidden layer.
            output_dim (int): The number of neurons in the output layer.
        """
        self.hidden_layer = LinearLayer(input_dim, hidden_dim)
        self.output_layer = LinearLayer(hidden_dim, output_dim, activation="")
        self.param = np.array([self.hidden_layer.param, self.output_layer.param]).flatten()

    # Your code: implement the function
    def forward(self, x):
        """
        Computes the forward pass of the MLP.

        Args:
            x (Tensor): Input tensor to the neural network.

        Returns:
            Tensor: Output tensor after processing through both layers of the MLP.
        """
        hidden = self.hidden_layer.forward(x)  

        return self.output_layer.forward(hidden)  
    
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
                # Apply the learning rate and update the parameter for broadcast compatibility
                param.data -= self.lr * np.sum(param.grad, axis=0, keepdims=True)
            else:
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
def train_func(network, train_dataloader, val_dataloader, optimizer, loss_func, epoch_num):
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
    for epoch in range(epoch_num):
        i = 0
        for x, y in train_dataloader:
            # (x-np.mean(x))/np.sqrt(np.std(x)**2)
            output = network.forward(Tensor(x))

            labels = Tensor(one_hot_encode(y, num_classes=10))
            loss = loss_func.forward(output, labels)
            if np.isnan(loss.data):
                # print(i)
                pass
            i += 1
            loss_func.backward(loss)
            output.backward(output.grad)
            
            optimizer.step()
            optimizer.zero_grad()
            
        with no_grad():
            accuracy, correct_num, total_num = validation_func(network, val_dataloader)
            validation_acc.append(accuracy)

    return validation_acc

# region TORCH LOADER
import os

USE_CPU_ONLY = True
flags = os.environ.get("XLA_FLAGS", "")
if USE_CPU_ONLY:
    flags += " --xla_force_host_platform_device_count=8"
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = flags

from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST
import jax 
import jax.numpy as jnp
from jax import vmap, grad, jit, pmap
from jax.scipy.special import logsumexp
import time
import numpy as np
from jax import config 

def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = jax.random.split(key)
  return scale * jax.random.normal(w_key, (n, m)), scale * jax.random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = jax.random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def relu(x):
  return jnp.maximum(0, x)

def predict(params, image):
  # per-example predictions
  activations = image
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)
    
  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits - logsumexp(logits)

def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)
    
def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)

@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]
    
def torch_loader(batch_size, n_targets):
    def numpy_collate(batch):
        return tree_map(np.asarray, data.default_collate(batch))

    class NumpyLoader(data.DataLoader):
        def __init__(self, dataset, batch_size=1,
                        shuffle=False, sampler=None,
                        batch_sampler=None, num_workers=0,
                        pin_memory=False, drop_last=False,
                        timeout=0, worker_init_fn=None):
            super(self.__class__, self).__init__(dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=numpy_collate,
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                worker_init_fn=worker_init_fn)

    class FlattenAndCast(object):
        def __call__(self, pic):
            return np.ravel(np.array(pic, dtype=jnp.float32))
        
    
    # Define our dataset, using torch datasets
    mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
    training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)
    # Get the full train dataset (for checking accuracy while training)
    train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
    train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

    # Get full test dataset
    mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
    test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
    test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)
    
    return training_generator, (train_images, train_labels), (test_images, test_labels)
    
def torch_train(training_generator, train, test, params):
    train_images, train_labels = train
    test_images, test_labels = test    
    
    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in training_generator:
            print(np.shape(x))
            print(np.shape(y), y[0])
            y = one_hot(y, n_targets)
            params = update(params, x, y)
        epoch_time = time.time() - start_time

    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
    
    
if __name__ == "__main__":
    torch_load = True
    if torch_load:
        layer_sizes = [784, 512, 512, 10]
        step_size = 0.01
        num_epochs = 8
        batch_size = 128
        n_targets = 10
        
        params = init_network_params(layer_sizes, jax.random.key(0))

        batched_predict = vmap(predict, in_axes=(None, 0))

        training_generator, train, test = torch_loader(batch_size, n_targets)
        
        torch_train(training_generator, train, test, params)
    else:    
        # Read all MNIST training data from the file
        mnist_data = pd.read_csv('train.csv')
        # Extract the image data from the data
        mnist_data_x = mnist_data.iloc[:, 1:].values.astype('float')
        # Extract the labels from the data
        mnist_data_y = mnist_data.iloc[:, 0].values
        
        train_indices, val_indices = train_validate_split(mnist_data_y, val_ratio=0.2)

        batch_size = 64
        # Define training set dataloader object
        train_dataloader = DataLoader(mnist_data_x, mnist_data_y, batch_size, train_indices)
        # Define validation set dataloader object
        val_dataloader = DataLoader(mnist_data_x, mnist_data_y, batch_size, val_indices)
        # Define neural network
        network = MLP(28*28, 128, 10)
        # Define optimizer
        optimizer = SGDOptimizer(network.param, lr=0.001)
        # Define loss function
        loss_func = SoftmaxCrossEntropy()

        epoch_num = 30
        val_accuracy_list = train_func(network, train_dataloader, val_dataloader, optimizer, loss_func, epoch_num)
        plt.figure()
        plt.plot([i+1 for i in range(epoch_num)], val_accuracy_list, 'o-')
        plt.xlabel('Epoch')  # Label for the x-axis
        plt.ylabel('Validation Accuracy')  # Label for the y-axis
        plt.grid(True)  # Enable grid for easier visualization of the plot lines
        plt.savefig("mnist training accuracy")