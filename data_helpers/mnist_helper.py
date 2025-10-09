import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from PIL import Image
import math
from abc import ABC, abstractmethod
try:
    import data_helpers.network_helper as network_helper
except ModuleNotFoundError:
    import network_helper
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
    
def torch_mnist_loader(batch_size=1, n_targets=10):
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
    
    # Calculate total number of batches
    total_samples = len(mnist_dataset)
    total_batches = (total_samples + batch_size - 1) // batch_size
    
    training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)
    # Get the full train dataset (for checking accuracy while training)
    train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
    train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

    # Get full test dataset
    mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
    test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
    test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)
    
    return training_generator, (train_images, train_labels), (test_images, test_labels), total_batches


def torch_train(training_generator, train, test, params):
    train_images, train_labels = train
    test_images, test_labels = test    
    
    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in training_generator:
            y = one_hot(y, n_targets)
            params = update(params, x, y)
        epoch_time = time.time() - start_time

    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
    
def average_active_inputs(train_dataloader):
    """
    Returns the mean number of non-zero features per sample across the
    *entire* training set.

    Assumes each batch_x is (batch, features, ...). For images, flatten first.
    """
    total_active_features = 0   # numerator
    total_samples         = 0   # denominator

    for batch_x, _ in train_dataloader:
        # 👉 ensure 2‑D (batch , features) – flatten everything but batch axis
        flat = batch_x.reshape(batch_x.shape[0], -1)

        # non‑zero count per sample
        active_per_sample = (flat != 0).sum(axis=1)      # shape (batch ,)

        total_active_features += active_per_sample.sum() # scalar
        total_samples         += flat.shape[0]

    return total_active_features / total_samples

#region MANUAL LOADER
def mnist_loader_manual(batch_size, shuffle=False, preprocess=True, CNN_preproces=False, cache_dir='./cache/mnist'):
    max_nonzero = 351
    dataset_folder = "data/mnist/"

    if not preprocess:
        os.makedirs(cache_dir, exist_ok=True)
        train_cache_path = os.path.join(cache_dir, '/train.npz')
        test_cache_path  = os.path.join(cache_dir, '/test.npz')

    else:
        if CNN_preproces:
            cache_dir += "/async_CNN"
            os.makedirs(cache_dir, exist_ok=True)

            train_cache_path = os.path.join(cache_dir, 'train.npz')
            test_cache_path  = os.path.join(cache_dir, 'test.npz')
        else:
            cache_dir += "/async_MLP"
            os.makedirs(cache_dir, exist_ok=True)

            train_cache_path = os.path.join(cache_dir, 'train.npz')
            test_cache_path  = os.path.join(cache_dir, 'test.npz')

    if preprocess and os.path.exists(train_cache_path):
        print("Loading cached MNIST dataset")
        data = np.load(train_cache_path)
        mnist_data_x = data['x']
        mnist_data_y = data['y']

        data = np.load(test_cache_path)
        mnist_data_x_test = data['x']
        mnist_data_y_test = data['y']
    else:
        # Read all MNIST training data from the file
        mnist_data = pd.read_csv(dataset_folder + 'mnist_train.csv')
        mnist_data_x = mnist_data.iloc[:, 1:].values.astype('float')
        mnist_data_y = mnist_data.iloc[:, 0].values
        
        # Read all MNIST test data from the file
        mnist_data = pd.read_csv(dataset_folder + 'mnist_test.csv')
        mnist_data_x_test = mnist_data.iloc[:, 1:].values.astype('float')
        mnist_data_y_test = mnist_data.iloc[:, 0].values

        if preprocess:
            if CNN_preproces:
                print("Preprocess MNIST dataset for CNN")                    
                mnist_data_x = preprocess_dataset_CNN(mnist_data_x, max_nonzero)
                mnist_data_x_test = preprocess_dataset_CNN(mnist_data_x_test, max_nonzero)

                # Save cached dataset
                np.savez_compressed(train_cache_path, x=mnist_data_x, y=mnist_data_y)
                np.savez_compressed(test_cache_path, x=mnist_data_x_test, y=mnist_data_y_test)
            else:
                print('Preprocessing MNIST dataset')
                mnist_data_x = preprocess_dataset(mnist_data_x, max_nonzero)
                mnist_data_x_test = preprocess_dataset(mnist_data_x_test, max_nonzero)

                # Save cached dataset
                np.savez_compressed(train_cache_path, x=mnist_data_x, y=mnist_data_y)
                np.savez_compressed(test_cache_path, x=mnist_data_x_test, y=mnist_data_y_test)
    
    # print(mnist_data_x[0], mnist_data_x.shape, mnist_data_x[0].shape)
    
    # Define training set dataloader object
    train_indices, val_indices = network_helper.train_validate_split(mnist_data_y, val_ratio=0.2, shuffle=shuffle)
    train_dataloader = network_helper.DataLoader(mnist_data_x, mnist_data_y, batch_size, train_indices, shuffle=shuffle)    
    val_dataloader = network_helper.DataLoader(mnist_data_x, mnist_data_y, batch_size, val_indices, shuffle=shuffle)
    
    # Define test set dataloader object
    test_indices, _ = network_helper.train_validate_split(mnist_data_y_test, val_ratio=0, shuffle=shuffle)
    test_dataloader = network_helper.DataLoader(mnist_data_x_test, mnist_data_y_test, batch_size, test_indices)
    
    # Calculate total batches for train, val, test data
    total_train_batches = network_helper.get_total_batches(batch_size, train_indices)
    total_val_batches = network_helper.get_total_batches(batch_size, val_indices)
    total_test_batches = network_helper.get_total_batches(batch_size, test_indices)
    # print("total test batches",total_test_batches)
    
    # Infinite dataloader
    # train_dataloader = network_helper.InfiniteDataLoader(train_dataloader)
    # val_dataloader = network_helper.InfiniteDataLoader(val_dataloader)
    # test_dataloader = network_helper.InfiniteDataLoader(test_dataloader)
    
    # !!! WITHOUT PREPROCESSING !!! Compute the maximum of non-zero elements in the input data 
    # max_nonzero = 0
    # for dataset in [train_dataloader, val_dataloader, test_dataloader]:
    #     for x, _ in iter(dataset):
    #         non_zeros = np.array([np.count_nonzero(row) for row in x])
    #         n_nonzeros = max(non_zeros)
    #         max_nonzero = max(n_nonzeros, max_nonzero)
    
    return (train_dataloader, total_train_batches), (val_dataloader, total_val_batches), (test_dataloader, total_test_batches), max_nonzero

def mnist_loader_preprocessed_single(x, max_nonzero):
    """
    Preprocess a single MNIST sample (1D vector).
    Stores (index, value) for non-zero pixels up to max_nonzero.
    """
    processed_data = np.full((max_nonzero, 2), -2.0, dtype=np.float32)
    j = 0
    for i, val in enumerate(x):
        if val != 0:
            processed_data[j] = [i, val]
            j += 1
            if j >= max_nonzero:  # stop if full
                break
    return processed_data

def preprocess_dataset(dataset_x, max_nonzero):
    """
    Apply preprocessing to the whole dataset.
    dataset_x: shape (N, 784)
    Returns: shape (N, max_nonzero, 2)
    """
    N = dataset_x.shape[0]
    processed_dataset = np.zeros((N, max_nonzero, 2), dtype=np.float32)
    for n in range(N):
        processed_dataset[n] = mnist_loader_preprocessed_single(dataset_x[n], max_nonzero)
    return processed_dataset

def mnist_loader_preprocessed_single_CNN(x, max_nonzero, input_dimension=28):
    """
    Preprocess a single MNIST sample (1D vector).
    Stores (0, x, y, value) for non-zero pixels up to max_nonzero.
    """
    processed_data = np.full((max_nonzero, 4), -2.0, dtype=np.float32)
    j = 0
    for i, val in enumerate(x):
        if val != 0:
            x = i // input_dimension    # row
            y = i % input_dimension     # column
            # print(i, x, y)
            processed_data[j] = [0, x, y, val]
            j += 1
            if j >= max_nonzero:  # stop if full
                break
    return processed_data

def preprocess_dataset_CNN(dataset_x, max_nonzero):
    """
    Apply preprocessing to the whole dataset.
    dataset_x: shape (N, 784)
    Returns: shape (N, max_nonzero, 4) ==> tuple(c, x, y, v) -- tuple(0, x, y, v)
    """
    N = dataset_x.shape[0]
    processed_dataset = np.zeros((N, max_nonzero, 4), dtype=np.float32)
    for n in range(N):
        processed_dataset[n] = mnist_loader_preprocessed_single_CNN(dataset_x[n], max_nonzero)
    return processed_dataset

if __name__ == "__main__":
    torch_load = False
    if torch_load:
        layer_sizes = [784, 512, 512, 10]
        step_size = 0.01
        num_epochs = 8
        batch_size = 128
        n_targets = 10
        
        params = init_network_params(layer_sizes, jax.random.key(0))

        batched_predict = vmap(predict, in_axes=(None, 0))

        training_generator, train, test, total_batches = torch_mnist_loader(batch_size, n_targets)
        print(total_batches)
        for x, y in training_generator:
            print(f"Batch x: {x},{type(x)}")
            print(f"Batch y: {y}, {type(y)}")
            break
        # torch_train(training_generator, train, test, params)
    else:    
        batch_size = 128
        # (training_generator, total_train_batches), (validation_generator, total_val_batches), (test_generator, total_test_batches), max_nonzero = mnist_loader_manual(batch_size, shuffle=False, preprocess=True)
        
        # d = iter(test_generator)
        # for _, _ in iter(training_generator):
        #     batchx, batchy = next(d)
        #     print(jnp.array(batchx).shape)

        (training_generator, total_train_batches), (validation_generator, total_val_batches), (test_generator, total_test_batches), max_nonzero = mnist_loader_manual(batch_size, shuffle=False, preprocess=False, CNN_preproces=False)

        avg_non_zero = average_active_inputs(training_generator)
        print(f"Average non‑zero inputs per sample: {avg_non_zero:.2f}")

        # Define neural network
        # layer_dims = (784, 128, 10)
        layer_size = []
        
        # layer_size.append((28*28, 32, 32, 32, 32, 32, 32, 32, 10))
        # layer_size.append((28*28, 64, 64, 64, 64, 64, 64, 64, 10))
        # layer_size.append((28*28, 128, 128, 128, 128, 128, 128, 128, 10))
        layer_size.append((28*28, 32, 32, 32, 32, 32, 32, 10))
        # layer_size.append((28*28, 32, 32, 32, 32, 32, 10))
        # layer_size.append((28*28, 32, 32, 32, 32, 10))
        # layer_size.append((28*28, 32, 32, 32, 10))
        # layer_size.append((28*28, 32, 32, 10))
        # layer_size.append((28*28, 32, 10))
        # layer_size.append((28*28, 64, 64, 64, 64, 64, 64, 10))
        # layer_size.append((28*28, 64, 64, 64, 64, 64, 10))
        # layer_size.append((28*28, 64, 64, 64, 64,10))
        # layer_size.append((28*28, 64, 64, 64, 10))
        # layer_size.append((28*28, 64, 64, 10))
        # layer_size.append((28*28, 64, 10))
        # layer_size.append((28*28, 128, 128, 128, 128, 128, 128, 10))
        # layer_size.append((28*28, 128, 128, 128, 128, 128, 10))
        # layer_size.append((28*28, 128, 128, 128, 128,10))
        # layer_size.append((28*28, 128, 128, 128, 10))
        # layer_size.append((28*28, 32, 10))
        # layer_size.append((28*28, 128, 128, 10))
        
        for layer_dims in layer_size:
            print(f"running {layer_dims}")
            network = network_helper.MLP(layer_dims)
            
            filename = f"tensor_data_{'_'.join(map(str, layer_dims))}_batch{batch_size}"
            output_dir = "tensor_data/MLP"
            os.makedirs(output_dir, exist_ok=True)

            reload = False
            if reload:
                filename = f"tensor_data_{'_'.join(map(str, layer_dims))}_batch{batch_size}"
                output_dir = "tensor_data/MLP"
                file_path = os.path.join(output_dir, f"{filename}.npz")

                # Load the saved tensors
                loaded = np.load(file_path)

                # Copy them back to the model
                for param_tensor, loaded_array in zip(network.param, loaded.values()):
                    param_tensor.data[:] = loaded_array

            # Define optimizer
            optimizer = network_helper.SGDOptimizer(network.param, lr=0.001)
            # Define loss function
            loss_func = network_helper.SoftmaxCrossEntropy()

            epoch_num = 20
            start_time = time.time()
            train_accuracy_list, val_accuracy_list = [], []
            train_accuracy_list, val_accuracy_list, activations = network_helper.train_func(network, training_generator, validation_generator, optimizer, loss_func, epoch_num)
            end_time = time.time()
            
            val_start = time.time()
            test_acc, _, _ = network_helper.validation_func(network, test_generator)
            val_end = time.time()
            
            inference_time = val_end - val_start
            execution_time = end_time - start_time
            print(f"Execution Time: {execution_time:.6f} seconds")
            print(f"Inference Execution Time: {inference_time:.6f} seconds")
            print(f"Final Val Acc: {val_accuracy_list[-1]:.4f} | Final Train Acc: {train_accuracy_list[-1]:.4f} | Test Acc: {test_acc:.4f}")
            plt.figure(figsize=(8, 5))
            epochs = [i + 1 for i in range(epoch_num)]

            plt.plot(epochs, train_accuracy_list, 'o-', label='Training Accuracy')
            plt.plot(epochs, val_accuracy_list, 's-', label='Validation Accuracy')

            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f"Final Val Acc: {val_accuracy_list[-1]:.4f} | Final Train Acc: {train_accuracy_list[-1]:.4f} | Test Acc: {test_acc:.4f}")
            plt.legend()
            plt.grid(True)
            
            # reload = True
            if not reload:
                plt.savefig(os.path.join(output_dir, f"{filename}.png"))

                # print(network.param[0].data.shape)
                datafile = os.path.join(output_dir, f"{filename}.npz")
                print(f"Data saved in {datafile}")
                # Save parameters to .npz in the same folder
                tensor_data = [tensor.data for tensor in network.param]
                np.savez(datafile, *tensor_data)
            plt.close()

            # data = np.load(os.path.join(output_dir, f"{filename}.npz"))
            # print("Filename: {}", data)