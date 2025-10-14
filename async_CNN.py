from asyncio import gather
from math import e
import math
import os
from tqdm import tqdm

from async_MPI import layer_computation
from mpi4py import MPI
# os.environ["JAX_TRACEBACK_FILTERING"] = "on"
os.environ.pop("JAX_TRACEBACK_FILTERING", None)

import jax
import jax.numpy as jnp
from jax import custom_jvp, jit
from jax.tree_util import Partial
from functools import partial
from jax import jacfwd, jacrev
import optax

import torch
import tree_math
import dataclasses
from typing import Generic, Any, Union, TypeVar, Tuple
import time
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from functools import partial

import mpi4jax
from mpi4jax import send, recv, bcast

from dataset_helpers.mnist_helper import mnist_loader_manual
from dataset_helpers.iris_species_helper import torch_iris_loader
from dataset_helpers.network_helper import one_hot_encode
from dataset_helpers.nmnist_helper import torch_nmnist_loader
from dataset_helpers.shd_helper import torch_SHD_loader
from dataset_helpers.cnn_mnist import get_weights_for_rank

from other_helpers.helpers import pad_batch, accuracy, store_training_data
from other_helpers.backpropagation import back_prop
from other_helpers.loss_functions import loss_bpp, mean_loss

from jax.experimental import io_callback

def save_to_file(x, file_idx):
    filename = f"matrix{file_idx}.npy"
    
    if os.path.isfile(filename):
        return 
    
    np.save(filename, np.array(x))
    print(f"Saved tensor to {filename}")

jax.config.update("jax_debug_nans", True)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()      # Real rank
size = comm.Get_size()

split_rank = None           # Rank corresponding to the layer
process_per_layer = None    # Number of processes for each layer
last_rank = None            # Rank of last layer
batch_part = None           # The size of the batch on each process

training_generator = None
validation_generator = None
test_generator = None

@jax.tree_util.register_pytree_node_class
class NeuronStates:
    def __init__(self, values, thresholds, input_residuals, input_order, input_activity, layer_activity, output_activity, last_sent_iteration, input_vector, output_vector, weights_shape, is_conv=False):
        '''
        Shapes are referenced for a layer with weights of shape: (784, 128)

        values: jnp.ndarray              # Current state of the neurons in the layer, shape: (layer_sizes[rank],) __ (128,)
        thresholds: jnp.float32          # An array of thresholds, one per neuron, shape: (layer_sizes[rank],) __ (128,)
        input_residuals: jnp.ndarray     # Sum of all input neurons, shape: (layer_sizes[rank-1],) __ (784,)
        input order                      # Set input neuron to the iteration at which the input is received to record the order of input received, shape: (layer_sizes[rank-1],) __ (784,)
        input activity                   # Count the number of times a input neuron fired, shape: (layer_sizes[rank-1],) __ (784,)
        layer activity                   # Count the number of times a neuron activated in this layer, only used for restrict parameter and threshold, shape: (layer_sizes[rank],) __ (128,)
        output activity                  # For each input neuron stores the hidden neurons that fire, shape: (layer_sizes[rank-1], layer_sizes[rank]) __ (784, 128)  
        '''
        self.values = values
        self.thresholds = thresholds
        self.input_residuals = input_residuals
        self.input_order = input_order
        self.input_activity = input_activity
        self.layer_activity = layer_activity
        self.output_activity = output_activity
        self.last_sent_iteration = last_sent_iteration
        self.input_vector = input_vector
        self.output_vector = output_vector
        self.weights_shape = weights_shape
        self.is_conv = is_conv

    # Tell JAX how to flatten this object
    def tree_flatten(self):
        children = (self.values, self.thresholds, self.input_residuals,
                    self.input_order, self.input_activity, self.layer_activity,
                    self.output_activity, self.last_sent_iteration, 
                    self.input_vector, self.output_vector, self.weights_shape, self.is_conv)
        aux_data = None  # no extra static data
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    

@jax.tree_util.register_pytree_node_class
class ConvNeuronStates():
    """
    ConvNeuronStates extends NeuronStates to include convolutional properties.

    It adds kernel_size, padding, stride which need to be 2 dimensional tuples containing the height and the width.

    Attributes:
        kernel (tuple[int, int]):               The kernel size for the convolutional layer, shape: (2,)
        padding (tuple[int, int]):              The padding for the convolutional layer, shape: (2,)
        stride (tuple[int, int]):               The stride for the convolutional layer, shape: (2,)
        previous_layer (jnp.ndarray):           Records the last received input from the previous layer
        is_conv (bool):                         True if convolutional layer or pooling layer
        pooling (str):                       "" = No pooling, "max" == Max pooling, "avg" == Average pooling
    """
    def __init__(self, neuron_state: NeuronStates,
                 kernel: tuple[int, int],
                 padding: tuple[int, int] | str, # Padding can be either a tuple for padding in each direction or a string, "SAME"=keep same size, "VALID"=No padding
                 stride: tuple[int, int],
                 previous_layer: jnp.ndarray,
                 is_conv: bool = True,
                 pooling: str = "",
                 pool_size: tuple[int, int] = (2, 2),
                 pool_stride: tuple[int, int] = (2, 2)):

        self.neuron_state = neuron_state
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.previous_layer = previous_layer
        self.is_conv = is_conv
        self.pooling = pooling
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        
    def __getattr__(self, name):
        # fallback: delegate attribute access to neuron_state
        return getattr(self.neuron_state, name)

    # PyTree flatten/unflatten
    def tree_flatten(self):
        children = (self.neuron_state,
                    self.previous_layer)
        aux_data = (self.kernel,
                    self.padding,
                    self.stride,
                    self.is_conv,
                    self.pooling,
                    self.pool_size,
                    self.pool_stride)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        kernel, padding, stride, is_conv, pooling, pool_size, pool_stride = aux_data
        neuron_state, previous_layer = children
        return cls(neuron_state, kernel, padding, stride, previous_layer, 
                   is_conv, pooling, pool_size, pool_stride)

@dataclasses.dataclass(frozen=True)
class Params:
    dataset: str 
    random_seed: int
    layer_sizes: tuple[int, ...]
    init_thresholds: float  # Starting thresholds
    num_epochs: int 
    learning_rate: float
    batch_size: int
    load_file: bool
    shuffle_activations: bool           # Shuffle the activations in the network
    restrict: int           # The amount of times a single neuron can fire accross all inputs, if negative then no restriction
    firing_nb: int          # The maximum number of neurons that can fire for one input at each layer
    sync_rate: int          # The number of inputs that needs to be accumulated before firing  
    max_nonzero: int
    shuffle_input:bool      # Shuffle the input data
    threshold_lr: float
    sparsity_impact: float
    rerun: str
    async_layer: int        # The layer that is training asynchronously while all other layers are training sync, if -1 then all layers are async
    max_kernel: int         # The maximum size of flattened kernel
    flat_layer_sizes: tuple[int, ...]

#region Initialization
@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Network:
    '''
    A class representing the neural network architecture.
    
    It initializes the layers depending on the layer_sizes input and stores either a convolution layer or a fully connected layer.
    
    It then initializes the weights/filters for each layer.
    '''
    params: Params
    key: jax.Array
    layers: tuple
    flat_layer_sizes: tuple
    conv_layer_sizes: tuple
    filename: str

    @classmethod
    def build(cls, params:Params, key, layer_sizes: tuple[tuple[int, ...]], flat_layer_sizes: tuple[tuple[int, ...]], conv_layer_sizes: tuple[tuple[int, ...]], th_bias=0.0, debug=False):
        '''
        Parameters:
            key: random key 
            params: Params object containing the parameters for the network
            layer_sizes (tuple of tuples): 
                        - for a fully connected layer each tuple contains a single integer representing the number of neurons in that layer
                        - for a convolutional layer each tuple contains (out_chan, kernel, padding, stride) representing the output channel, kernel size, padding and stride.
        
        '''
        layers = []
        flat_layer_sizes = []
        conv_layer_sizes = []
        previous_layer = jnp.zeros(0)  # placeholder for input shape tracking
        filename = f"_b{params.batch_size}"

        for i, layer in enumerate(layer_sizes):
            if len(layer) == 1: # Linear layer
                if i == 0:
                    prev_size = 1
                else:
                    prev_size = previous_layer if isinstance(previous_layer, int) else previous_layer.flatten().size
                    if rank == 0 and debug:
                        print(f"rank {rank}, Previous layer: {prev_size}")

                key, subkey = jax.random.split(key) 
                thresholds = jax.random.normal(subkey, layer) * params.init_thresholds + th_bias
                empty_neuron_states = NeuronStates(
                                    values=jnp.zeros(layer),
                                    thresholds=thresholds,
                                    input_residuals=jnp.zeros((prev_size,)),
                                    input_order=jnp.full((prev_size,), -1, dtype=int),
                                    input_activity=jnp.zeros((prev_size,), dtype=int),
                                    layer_activity=jnp.zeros((layer[0],), dtype=int),
                                    output_activity=jnp.zeros((prev_size, layer[0])),
                                    last_sent_iteration=0,
                                    input_vector=jnp.zeros((prev_size,)),
                                    output_vector=jnp.zeros((layer[0],)),
                                    weights_shape=(prev_size, layer[0]),
                                    is_conv=False,
                )
                layers.append(empty_neuron_states)
                previous_layer = layer[0]
                filename += f"_L{layer[0]}"
                flat_layer_sizes.append(layer)
                conv_layer_sizes.append(layer)
            else:
                in_chan = previous_layer.shape[0]
                pool_size = (2, 2)
                pool_stride = (2, 2)
                pooling = ""
                if len(layer) > 4:
                    pooling = layer[4]
                    if len(layer) > 5 : pool_size = layer[5]
                    if len(layer) > 6 : pool_stride = layer[6]
                    layer = layer[:4]
                
                if i == 0:
                    previous_layer = jnp.zeros(1)
                    values = jnp.zeros(layer)
                    out_chan, kernel, padding, stride = 1, (0,0), (0,0), (0,0) # Values used as placeholders for the input layer
                    filename += "_C{}x{}x{}".format(*layer)
                    out_chan, h_out, w_out = layer
                else:
                    out_chan, kernel, padding, stride = layer
                    in_shape = previous_layer.shape
                    h_out = (in_shape[1] + 2 * padding[0] - kernel[0]) // stride[0] + 1
                    w_out = (in_shape[2] + 2 * padding[1] - kernel[1]) // stride[1] + 1

                    if rank == 0 and debug:
                        print(f"rank {rank}, previous layer shape: {in_shape}, out shape: {(out_chan, h_out, w_out)}, kernel: {kernel}, padding: {padding}, stride: {stride}")
                    values = jnp.zeros((out_chan, h_out, w_out))  # Initialize values for convolutional layer
                    filename += f"_C{out_chan}x{in_chan}x{kernel[0]}x{kernel[1]}"

                h_out_pool, w_out_pool = h_out, w_out
                if pooling != "":
                    h_out_pool = (h_out - pool_size[0]) // pool_stride[0] + 1
                    w_out_pool = (w_out - pool_size[1]) // pool_stride[1] + 1
                    if pooling == "max":
                        filename += f"_P{pool_size[0]}x{pool_size[1]}"
                    elif pooling == "avg":
                        filename += f"_AvgP{pool_size[0]}x{pool_size[1]}"
                    # print(h_out_pool, w_out_pool)

                key, subkey = jax.random.split(key) 
                thresholds = jax.random.normal(subkey, values.shape) * params.init_thresholds + th_bias
                neuron_state = NeuronStates(
                    values=values,
                    thresholds=thresholds,
                    input_residuals=jnp.zeros(previous_layer.shape),
                    input_order=jnp.full(previous_layer.shape, -1, dtype=int),
                    input_activity=jnp.zeros(previous_layer.shape, dtype=int),
                    layer_activity=jnp.zeros(values.shape, dtype=int),
                    output_activity=jnp.zeros_like(values),  # placeholder, shape matches values
                    last_sent_iteration=0,
                    input_vector=jnp.zeros(previous_layer.shape),
                    output_vector=jnp.zeros(values.shape),
                    weights_shape=(out_chan, in_chan, kernel[0], kernel[1]),
                    is_conv=True
                )

                empty_conv_neuron = ConvNeuronStates(
                    neuron_state=neuron_state,
                    kernel=kernel,
                    padding=padding,
                    stride=stride,
                    previous_layer=previous_layer,
                    pooling=pooling,
                    pool_size=pool_size,
                    pool_stride=pool_stride
                )
                layers.append(empty_conv_neuron)
                previous_layer = jnp.zeros((out_chan, h_out_pool, w_out_pool)) # Shape after pooling 
                flat_layer_sizes.append(previous_layer.shape)
                conv_layer_sizes.append(values.shape) # Needed to gather the thresholds after computation
        return cls(params=params, key=key, layers=tuple(layers), flat_layer_sizes=tuple(flat_layer_sizes), conv_layer_sizes=tuple(conv_layer_sizes), filename=filename)

    def init_weights(self):
        '''
        Initialize the weights for each layer based on the layer sizes.
        
        Returns the weights correponding to the MPI split_rank.
        ''' 
        weights = init_params(self.key, self.layers, self.params.load_file, self.filename)
        return weights
    
    def rerun(self, thresholds):
        if thresholds is not None:
            # print(split_rank, type(self.layers[split_rank]))
            old_layer = self.layers[split_rank]
            layers_list = list(self.layers)

            if isinstance(old_layer, ConvNeuronStates):
                new_neuron_state = old_layer.neuron_state.replace(thresholds=thresholds)
                layers_list[split_rank] = old_layer.replace(neuron_state=new_neuron_state)
            else:
                layers_list[split_rank] = old_layer.replace(thresholds=thresholds)

            self.layers = tuple(layers_list)
        return self.layers[split_rank]
    
    def tree_flatten(self):
        # children are arrays or other pytree objects
        children = (self.params, self.layers, self.key)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params, layers, key = children
        return cls(params=params, layers=layers, key=key)

def init_params(key, layers, load_file=False, filename="", best=False, scale=1e-2):
    # Initialize weights for each layer
    keys = jax.random.split(key, len(layers))

    if split_rank != 0:
        if load_file:
            folder = "tensor_data/CNN/"
            f = "tensor_data"+filename+".npz"
            return get_weights_for_rank(folder+f, split_rank)

        # Random initialization of the weights       
        layer = layers[split_rank]
        weights_shape = layer.weights_shape
        print(weights_shape)
        if layer.is_conv:
            out_ch, in_ch, kh, kw = weights_shape
            dtype=jnp.float32
            
            fan_in = in_ch * kh * kw
            bound = jnp.sqrt(6.0 / fan_in)
            return jax.random.uniform(jax.random.PRNGKey(0), weights_shape, dtype, -bound, bound)
        else:
            return scale * jax.random.normal(keys[split_rank], weights_shape)
    else:
        return jnp.zeros((1,1,1,1)) # Return an empty holder for the weights of the input layer
    
# region INFERENCE
@custom_jvp # If thresholds == 0 then this behaves as a ReLu activation function 
def activation_func(thresholds, activations):
    # return jax.nn.relu(activations)
    return jnp.where(activations > thresholds, activations, 0.0)

@activation_func.defjvp
def activation_func_jvp(primals, tangents, k=1.0):
    # Surrogate gradient, redefine the function for the backward pass
    neuron_states, activations, = primals
    neuron_states_dot, activations_dot, = tangents
    ans = activation_func(neuron_states, activations)
    ans_dot = jnp.where(activations > neuron_states.thresholds, activations, 0.0)
    return ans, ans_dot

def keep_top_k(x, k):
    # Get the top-k values and their indices
    k_safe = min(k, x.shape[0]) #TODO investigate why this function gets compiled for the last layer, without cond throws a shape error
    # jax.lax.cond(k_safe != k,
    #              lambda _: jax.debug.print("Rank {} k safe: {}, k: {}", rank, k_safe, k),
    #              lambda _: None,
    #              None)
    k = k_safe

    _, top_indices = jax.lax.top_k(x, k)

    # Create a mask with 1s at top-k indices, 0 elsewhere
    mask = jnp.zeros(x.shape)
    mask = mask.at[top_indices].set(1)

    out = x * mask
    return out

@partial(jax.jit, static_argnames=['params'])
def process_activated_output(key, arr: jnp.ndarray, params):
    '''
    Processed the output of a layer from (1d array) to (2d array) with [(neuron idx, value)]
    value == 0 are filled with index==-2
    '''
    max_len = params.layer_sizes[split_rank][0]

    # indices of nonzero values (padded with -2)
    idx = jnp.nonzero(arr, size=max_len, fill_value=-2)[0]
    vals = jnp.where(idx != -2, arr[idx], 0)

    # stack before shuffle
    pairs = jnp.stack([idx, vals], axis=1)

    def do_shuffle(pairs):
         # mask: 1 for valid entries, 0 for padded (-2, 0)
        mask = (idx != -2).astype(jnp.int32)
        
        # assign random keys for sorting
        rand_keys = jax.random.uniform(key, (max_len,))

        # ensure valid entries come first, shuffled within themselves
        sort_keys = jnp.where(mask == 1, rand_keys, rand_keys + 2.0)  
        permuted = pairs[jnp.argsort(sort_keys)]

        return permuted

    pairs_out = jax.lax.cond(
        params.shuffle_activations,
        do_shuffle,
        lambda pairs: pairs,
        operand=pairs
    )
    # jax.debug,print("pairs out: {}", pairs_out.shape) # TODO check why this only prints once
    return pairs_out

#region FC computation 
@partial(jax.jit, static_argnames=['params'])
def fc_layer_computation(params, key, neuron_idx, layer_input, weights, neuron_states, iteration=0):    
    c, x, y = neuron_idx
    # jax.debug.print("rank {} has neuron idx: {}", rank, neuron_idx)

    C, H, W = 0, 0, 0
    flat_layer_size = params.flat_layer_sizes[split_rank-1]
    # jax.debug.print("linear flat layer: {}", flat_layer_size)
    if len(flat_layer_size) == 3:
        C, H, W = flat_layer_size
    neuron_idx = c * (H * W) + x * W + y 
    
    activations = jax.lax.cond(neuron_idx < 0,
                            lambda _: neuron_states.values,
                            lambda _: jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values,
                            None
                            )
    # io_callback(lambda arr, name: save_to_file(arr, name), None, weights[neuron_idx], neuron_idx)

    #TODO being able to compute multiple incoming index neurons
    #TODO store the weight residuals of last layer and neuron state of input layer in sparse matrix representation to reduce space utilization because unused
    # jax.lax.cond(jnp.isnan(jnp.array(layer_input)).any(), 
    #                        lambda _: jax.debug.print("Rank {}: layer_input is NaN: {}, idx: {}, iteration:{}", rank, layer_input, neuron_idx, iteration), 
    #                        lambda _: None, None)
    # jax.lax.cond(neuron_idx == -2,
    #                 lambda _: jax.debug.print("{}, iteration: {}, neuron idx: {}", layer_input, iteration, neuron_idx),
    #                 lambda _: None,
    #                 None)

    
    new_input_residuals = jax.lax.cond(neuron_idx < 0,
                            lambda _: neuron_states.input_residuals,
                            lambda _: neuron_states.input_residuals.at[neuron_idx].add(layer_input),
                            None
                            )
    new_input_activity = jax.lax.cond(neuron_idx < 0,
                            lambda _: neuron_states.input_activity,
                            lambda _: neuron_states.input_activity.at[neuron_idx].add(1),
                            None
                            )

    def last_layer_case(_):
        return jnp.array(0), jnp.zeros((activations.shape[0], 4)), NeuronStates(  values=activations, 
                                                                    thresholds=neuron_states.thresholds, 
                                                                    input_residuals=new_input_residuals, 
                                                                    input_order=neuron_states.input_order, 
                                                                    input_activity=new_input_activity,
                                                                    layer_activity=neuron_states.layer_activity,
                                                                    output_activity=neuron_states.output_activity,
                                                                    last_sent_iteration=neuron_states.last_sent_iteration,
                                                                    input_vector=neuron_states.input_vector,
                                                                    output_vector=neuron_states.output_vector,
                                                                    weights_shape=neuron_states.weights_shape,
                                                                    is_conv=neuron_states.is_conv)
    
    
    def hidden_layer_case(_):
        fire = (iteration-neuron_states.last_sent_iteration) >= params.sync_rate # Fire if sync rate reached
        async_fire = jnp.logical_or(params.async_layer < 0, split_rank <= params.async_layer) # Fire if async_layer or no async_layer condition (-1)
        fire = jnp.logical_and(fire, async_fire) 
        fire = jnp.logical_or(fire, neuron_idx < 0) # Fire if last input received

        # APPLY THE SYNC RATE  
        activated_output = jax.lax.cond(fire, 
                                        lambda args: activation_func(args[0], args[1]), 
                                        lambda _: jnp.zeros(activations.shape),
                                        (neuron_states.thresholds, activations))
        
        # APPLY THE FIRING NUMBER        
        activated_output = keep_top_k(activated_output, params.firing_nb) # Get the top k activations
        # jax.debug.print("{}, iteration: {}, neuron idx: {}", activated_output, iteration, neuron_idx)
        
        # APPLY THE RESTRICTION
        penalty = jax.lax.cond(params.restrict[split_rank] <= 0,
                               lambda _: activated_output, 
                               lambda _: activated_output*params.restrict[split_rank], None)
        
        # Store the neurons that activated
        active_indexes = jnp.where(activated_output > 0, 1, 0)
        new_layer_activities = neuron_states.layer_activity + active_indexes # Update the layer activity by adding the active neurons
        
        
        last_neuron_idx = jnp.argmax(neuron_states.input_order) # Last neuron index in the input order
        new_neuron_idx = jax.lax.cond(neuron_idx < 0,
                     lambda _: last_neuron_idx, 
                     lambda _: neuron_idx,
                     None)
        
        new_input_order = neuron_states.input_order.at[new_neuron_idx].set(iteration) # Update the input activity by setting the input neuron to the iteration number        
        
        # jax.debug.print("{} {}", active_indexes.shape, new_input_activities.shape)
        new_output_activity = neuron_states.output_activity.at[new_neuron_idx].add(active_indexes)
        
        new_last_sent_iteration = jax.lax.cond(fire, lambda _: iteration, lambda _: neuron_states.last_sent_iteration, None)

        # Added +1 so that we can differentiate between never activated (0) and activated at iteration 0 (1)
        new_input_vector = neuron_states.input_vector.at[neuron_idx].set(iteration+1)   # Set the input neuron to the iteration at which the input was received
        new_output_vector = jnp.where(activated_output > 0,                             # Set the output neuron to the last iteration at which it activated
                                    iteration+1,
                                    neuron_states.output_vector)
        
        new_neuron_states = NeuronStates(   values=activations - penalty, 
                                            thresholds=neuron_states.thresholds, 
                                            input_residuals=new_input_residuals, 
                                            input_order=new_input_order, 
                                            input_activity=new_input_activity,
                                            layer_activity=new_layer_activities,
                                            output_activity=new_output_activity,
                                            last_sent_iteration=new_last_sent_iteration,
                                            input_vector=new_input_vector,
                                            output_vector=new_output_vector,
                                            weights_shape=neuron_states.weights_shape,
                                            is_conv=neuron_states.is_conv)
        
        nb_valid_elements = jnp.count_nonzero(activated_output)
        processed_output = process_activated_output(key, activated_output, params) 

        # Pad to CNN format
        shaped_activated_output = jnp.pad(processed_output, ((0, 0), (2, 0)), constant_values=-2) 
        return nb_valid_elements, shaped_activated_output, new_neuron_states
    
    cond = split_rank == last_rank
    return jax.lax.cond(cond, last_layer_case, hidden_layer_case, None)

#region Pool computation
@jax.jit
def compact_nonzero_and_pad(events):
    """
    Reorders (c, x, y, v) events so that all v != 0 come first, 
    and pads the rest with -2. Keeps output shape identical to input.

    Args:
        events: jnp.ndarray of shape (N, 4), each row [c, x, y, v].

    Returns:
        compacted: jnp.ndarray of shape (N, 4)
        nonzero_count: number of nonzero v entries (scalar int32)
    """
    values = events[:, 3]
    
    # Boolean mask: 1 for nonzero values
    mask = values != 0

    # Get indices that would sort mask so that True come first
    # (~mask) is used so that True (1) goes before False (0)
    sort_keys = ~mask
    perm = jnp.argsort(sort_keys.astype(jnp.int32))

    # Reorder events
    compacted = events[perm]

    # Count nonzero
    nonzero_count = jnp.sum(mask).astype(jnp.int32)

    # Pad the remaining rows with -2 (only value field, not coords)
    # For zero entries, overwrite everything with -2
    def pad_fn(row, i):
        return jnp.where(i < nonzero_count, row, jnp.full_like(row, -2.0))

    indices = jnp.arange(events.shape[0])
    compacted = jax.vmap(pad_fn)(compacted, indices)

    return nonzero_count, compacted

def unpool(input_matrix, pooled, shape, pool_size=(2, 2), pool_stride=(2, 2)):
    C, H, W = shape
    sh, sw = pool_stride
    kh, kw = pool_size

    # Create upsampled version
    x_up_h = jnp.repeat(pooled, sh, axis=1)
    matrix_upsampled = jnp.repeat(x_up_h, sw, axis=2)
    
    # Pad if necessary to match original dimensions
    pad_h = H - matrix_upsampled.shape[1]
    pad_w = W - matrix_upsampled.shape[2]
    if pad_h > 0 or pad_w > 0:
        matrix_upsampled = jnp.pad(matrix_upsampled, 
                                ((0, 0), (0, pad_h), (0, pad_w)), 
                                constant_values=0)
    
    # Create mask for matching values
    mask = (input_matrix == matrix_upsampled)
    # print(mask)
    # Create priority matrix: prefer top-left positions
    # This gives unique priorities to each position
    priority = jnp.arange(H * W).reshape(1, H, W) * mask
    
    # For each pooling window, keep only the position with highest priority
    def keep_first_max(x):
        return jax.lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, kh, kw),
            window_strides=(1, sh, sw),
            padding="VALID",
        )
    
    max_priorities = keep_first_max(priority)
    max_priorities_upsampled = jnp.repeat(jnp.repeat(max_priorities, sh, axis=1), sw, axis=2)
    
    if pad_h > 0 or pad_w > 0:
        max_priorities_upsampled = jnp.pad(max_priorities_upsampled, 
                                            ((0, 0), (0, pad_h), (0, pad_w)), 
                                            constant_values=-jnp.inf)
    
    # Keep only positions that have the maximum priority in their window
    unique_mask = (priority == max_priorities_upsampled) & mask
    unpooled = matrix_upsampled * unique_mask

    return unpooled

@partial(jax.jit, static_argnums=(1, 2, 3, 4,))
def full_matrix_to_event_array_with_pooling(matrix, shape, pooling="", pool_size=(2, 2), pool_stride=(2, 2)):
    """
    Convert a full (C, H, W) matrix into event array format (N, 4),
    keeping only nonzero values at the beginning, padded with -2s at the end.

    Args:
        matrix: jnp.ndarray of shape (C, H, W)
        shape: tuple (C, H, W) – static shape of the full matrix

    Returns:
        (num_nonzero, padded_events)
        - num_nonzero: scalar, number of nonzero entries
        - padded_events: (C*H*W, 4) array of [c, x, y, value]
                         first num_nonzero rows are valid, rest are -2
    """
    C, H, W = shape
    
    unpooled = jnp.zeros_like(matrix)
    if pooling != "":
        kh, kw = pool_size
        sh, sw = pool_stride

        input_matrix = matrix
        # Compute output dimensions
        out_h = (H - kh) // sh + 1
        out_w = (W - kw) // sw + 1
        # io_callback(lambda arr, name: save_to_file(arr, name), None, input_matrix, 0)
        
        def pool_fn(x):
            # Extract all pooling windows efficiently
            windows = jax.lax.reduce_window(
                x,
                init_value=-jnp.inf if pooling == "max" else 0.0,
                computation=jax.lax.max if pooling == "max" else jax.lax.add,
                window_dimensions=(1, kh, kw),
                window_strides=(1, sh, sw),
                padding="VALID"
            )
            if pooling == "avg":
                windows = windows / (kh * kw)
            return windows

        # Apply pooling channel-wise
        matrix = pool_fn(input_matrix)
        # io_callback(lambda arr, name: save_to_file(arr, name), None, matrix, 1)

        if pooling == "max":
            unpooled = unpool(input_matrix, matrix, shape, pool_size, pool_stride)
        # io_callback(lambda arr, name: save_to_file(arr, name), None, unpooled, 2)
        
        # Update the shape for output computation
        H, W = matrix.shape[1:]
        shape = (C, H, W)
        

    N = C * H * W

    # Coordinate grid
    c_grid, x_grid, y_grid = jnp.meshgrid(
        jnp.arange(C), jnp.arange(H), jnp.arange(W), indexing='ij'
    )
    coords = jnp.stack([c_grid.ravel(), x_grid.ravel(), y_grid.ravel()], axis=-1)  # (N,3)

    values = matrix.ravel()  # (N,)
    nonzero_mask = values != 0
    num_nonzero = jnp.sum(nonzero_mask)

    # Build event array
    out_events = jnp.concatenate([coords, values[:, None]], axis=-1)  # (N,4)

    # Sort: put nonzeros first
    sort_keys = (~nonzero_mask).astype(jnp.int32)  # 0 for nonzero, 1 for zero
    perm = jnp.argsort(sort_keys)  # brings nonzeros to front
    sorted_events = out_events[perm]

    # Pad trailing rows with -2
    sentinel = jnp.full((N, 4), -2, dtype=matrix.dtype)
    padded_events = jnp.where(
        (jnp.arange(N)[:, None] < num_nonzero), sorted_events, sentinel
    )
    # nb_valid_el, compact_out = compact_nonzero_and_pad(padded_events)

    # jax.debug.print("Non zeros after pooling: {}", num_nonzero)
    # return N, out_events
    return num_nonzero, padded_events, unpooled
    # return nb_valid_el, compact_out

@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5,))
def sparse_pool(events, input_shape, mode="max", pool_size=(2, 2), stride=(2, 2), pad_value=0.0):
    """
    Sparse pooling on (c,x,y,value) event rows.

    Args:
        events: jnp.ndarray (N,4) containing (c, x, y, v). Use sentinel rows
                with c < 0 to indicate padding (they will be removed).
        input_shape: (C, H, W) (integers)
        mode: "max" or "avg"
        pool_size: (ph, pw)
        stride: (sh, sw)
        pad_value: value for empty pooled windows (default 0.0)

    Returns:
        pooled_events: jnp.ndarray shape (C * Hp * Wp, 4) with rows (c, x, y, value)
        pooled_shape: (C, Hp, Wp)
    """
    C, H, W = map(int, input_shape)
    ph, pw = map(int, pool_size)
    sh, sw = map(int, stride)
    
    # pooled grid sizes (assumes H and W are divisible by stride; adjust if needed)
    Hp = H // sh
    Wp = W // sw
    # jax.debug.print("Pool output sizes: {} {} {} {} {}", Hp, Wp, C, H, W)

    # if events.shape[0] == 0:
    #     jax.debug.print("No event in sparse pool ")
    #     # No events at all: return all zeros (or pad_value) sized array
    #     Hp = H // sh
    #     Wp = W // sw
    #     coords_full = jnp.stack([
    #         jnp.repeat(jnp.arange(C, dtype=jnp.int32), Hp * Wp),
    #         jnp.tile(jnp.repeat(jnp.arange(Hp, dtype=jnp.int32), Wp), C),
    #         jnp.tile(jnp.arange(Wp, dtype=jnp.int32), C * Hp)
    #     ], axis=-1)
    #     vals = jnp.full((C * Hp * Wp,), pad_value, dtype=jnp.float32)

    #     return jnp.concatenate([coords_full, vals[:, None]], axis=-1), (C, Hp, Wp)

    coords = events[:, :3].astype(jnp.int32)   # (N,3) integers
    values = events[:, 3].astype(jnp.float32)  # (N,)
    # jax.debug.print("in events in sparse pool: {}", events)

    # map each event to pooled cell
    pooled_c = coords[:, 0].astype(jnp.int32)
    pooled_x = (coords[:, 1] // sh).astype(jnp.int32)
    pooled_y = (coords[:, 2] // sw).astype(jnp.int32)

    pooled_idx = pooled_c * (Hp * Wp) + pooled_x * Wp + pooled_y    # integer index (N,)

    num_segments = C * Hp * Wp
    # jax.debug.print("After pooling indexes {} {} {} {} {}", pooled_c, pooled_x, pooled_y, pooled_idx, num_segments)

    # --- pooling ---
    if mode == "max":
        pooled_values = jax.ops.segment_max(values, pooled_idx, num_segments=num_segments)

        # Compute the original index of the max values
        is_max = values == pooled_values[pooled_idx]
        idx = jnp.arange(values.shape[0])
        argmax_idx = jax.ops.segment_min(jnp.where(is_max, idx, values.shape[0]),
                                        pooled_idx,
                                        num_segments=num_segments)
        
        
        # replace -inf (empty segments) with pad_value (e.g., 0.0)
        pooled_values = jnp.where(jnp.isneginf(pooled_values), pad_value, pooled_values)

        # Get the unpooled matrix for bpp
        unpooled = jnp.zeros_like(values)
        unpooled = unpooled.at[argmax_idx].set(pooled_values)
        
        unpooled_matrix = jnp.concatenate([coords, unpooled[:, None]], axis=-1)
    elif mode == "avg":
        # segment_sum returns 0 for empty segments
        sums = jax.ops.segment_sum(values, pooled_idx, num_segments=num_segments)
        # average over full kernel area (include zeros)
        area = ph * pw
        pooled_values = sums / float(area)
        # jax.debug.print("after segment sum and avg {} {} ", sums, pooled_values)
        unpooled_matrix = jnp.zeros(input_shape)
    else:
        raise ValueError("mode must be 'max' or 'avg'")

    # reconstruct coords for every pooled cell in canonical order
    pooled_c_full = jnp.repeat(jnp.arange(C, dtype=jnp.int32), Hp * Wp)
    pooled_x_full = jnp.tile(jnp.repeat(jnp.arange(Hp, dtype=jnp.int32), Wp), C)
    pooled_y_full = jnp.tile(jnp.arange(Wp, dtype=jnp.int32), C * Hp)

    coords_full = jnp.stack([pooled_c_full, pooled_x_full, pooled_y_full], axis=-1)  # (num_segments, 3)
    out = jnp.concatenate([coords_full, pooled_values[:, None]], axis=-1)  # (num_segments, 4)

    nb_valid_el, compact_out = compact_nonzero_and_pad(out)

    return nb_valid_el, compact_out, unpooled_matrix

@partial(jax.jit, static_argnums=(2, 3, 4, 5,))
def output_to_event_array_with_pooling(activated_output, start_indices, end_indices, pooling="", pool_size=(2,2), pool_stride=(2,2)):
    '''
    Transforms the activated output matrix into a list with format (c, x, y, value)
    to send to the next layer.
    
    activated_output: (c, k_h, k_w) - the activated output corresponding to the input neuron
    start_indices: (c, x, y) - the starting indices of the slice in the padded neuron states
    end_indices: (c, h, w) - the shape of the original neuron states
    kernel_padding: (c, k_h_pad, k_w_pad) - the padding of the kernel
    '''
    c, h, w = activated_output.shape

    # Step 1: Create coordinate grid
    c_grid, x_grid, y_grid = jnp.meshgrid(
        jnp.arange(c),
        jnp.arange(h),
        jnp.arange(w),
        indexing='ij'
    )
    # Flatten everything
    coords = jnp.stack([c_grid.ravel(), x_grid.ravel(), y_grid.ravel()], axis=-1)

    # Step 2: Adjust coordinates
    adjusted_coords = coords + jnp.array(start_indices)

    # Step 3: Create masks to filter out-of-bounds coordinates
    is_valid = jnp.all(
        (adjusted_coords >= jnp.zeros((3,))) &
        (adjusted_coords < jnp.array(end_indices)),
        axis=-1
    )
    values = activated_output.ravel()
    values_masked = values * is_valid.astype(values.dtype)

    # Step 4: 
    out_events = jnp.concatenate([adjusted_coords, values_masked[:, None]], axis=-1)
    # jax.debug.print("out events: {}", out_events)
    
    # Step 5: Pad to full size
    target_size = (end_indices[0]) * (end_indices[1]) * (end_indices[2])
    
    nb_valid_el = activated_output.size 
    # Step 6: Apply pooling if needed
    if pooling != "":
        # Combine with pooled values
        nb_valid_el, padded_out_events, unpooled_matrix = sparse_pool(out_events, end_indices, pooling, pool_size, pool_stride)
        # jax.debug.print("after pool el: {} and out shape: {}", nb_valid_el, padded_out_events.shape)
    else:
        pad_to_full_size = jnp.full((target_size-nb_valid_el, 4), -2)
        padded_out_events = jnp.concatenate([out_events, pad_to_full_size])
        unpooled_matrix = out_events
    # jax.debug.print("end indices: {}, target size: {}, pad to full size shape: {}, padded out events shape: {}", end_indices, target_size, pad_to_full_size.shape, padded_out_events.shape)

    return nb_valid_el, padded_out_events, unpooled_matrix
    
#region Conv computation
@partial(jax.jit, static_argnames=['params'])
def conv_layer_computation(params, key, neuron_idx, layer_input, weights, neuron_states, iteration=0):
    '''
    Apply the convolution for an incoming event in the event-driven manner described in "Optimizing event-based neural networks on digital neuromorphic architecture: a comprehensive design space exploration"
    This convolution only supports 'SAME' padding scheme with stride 1
    
    weights: (out_ch, in_ch, k_h, k_w)
    '''
    out_ch, in_ch, k_h, k_w = weights.shape
    c, x, y = neuron_idx
    activations_shape = (out_ch, k_h, k_w)
    activations_size = out_ch * k_h * k_w
    def regular_input(neuron_states):
        # jax.debug.print("rank {} has x: {}, y: {}", rank, x, y)

        # (1) Multiply the input value by the flipped kernel to obtain the output values
        activations = jnp.dot(layer_input, jnp.flip(weights[:, c, :, :], axis=(1, 2))) # Shape (out_ch, k_h, k_w) 
        # jax.debug.print("activations: {}", activations)

        # Check whether the layer fires at this iteration
        fire = (iteration-neuron_states.last_sent_iteration) >= params.sync_rate # Fire if sync rate reached
        async_fire = jnp.logical_or(params.async_layer <= 0, split_rank <= params.async_layer) # Fire if async_layer or no async_layer condition (-1)
        fire = jnp.logical_and(fire, async_fire) 
        fire = jnp.logical_or(fire, jnp.any(neuron_idx < 0)) # Fire if last input received    
        # jax.debug.print("fire: {}", fire)
        
        kernel_h_span, kernel_w_span = k_h//2, k_w//2 
        max_x, max_y = neuron_states.values.shape[1], neuron_states.values.shape[2] # c, h, w
        
        # Pad the values in neuron states to prevent indexing issues
        values_padded = jnp.pad(neuron_states.values, ((0, 0), (kernel_h_span, kernel_h_span), (kernel_w_span, kernel_w_span)))
        thresholds_padded = jnp.pad(neuron_states.thresholds, ((0, 0), (kernel_h_span, kernel_h_span), (kernel_w_span, kernel_w_span)))

        # Compute start indices for slicing and updating, slice padded matrix centered in original matrix's (x, y)  
        start_indices = (0, x, y) 
        slice_shape = activations.shape  # (C, k_h, k_w)
        # jax.debug.print("rank {}, slice shape {}", rank, slice_shape)
        # jax.debug.print("rank {}, neuron idx: {}, start indices: {}, slice shape: {}, values padded shape: {}", rank, neuron_idx, start_indices, slice_shape, values_padded.shape)
        
        # Step 1: Extract the current values from the padded tensor
        current_slice = jax.lax.dynamic_slice(values_padded, start_indices, slice_shape)
        thresholds_slice = jax.lax.dynamic_slice(thresholds_padded, start_indices, slice_shape)
        
        # Step 2: Add the new activations to the extracted slice
        updated_slice = current_slice + activations

        # Step 3: Compute ReLu on the updated slice if fire is True
        activated_output = jax.lax.cond(fire, 
                                        lambda _: activation_func(thresholds_slice, updated_slice),
                                        lambda _: jnp.zeros(updated_slice.shape),
                                        None)
        # jax.debug.print("rank {}, input: {}, activations: {}, updated slice: {}, activated output: {}", rank, layer_input, activations, updated_slice, activated_output)
        
        # APPLY THE RESTRICTION
        penalty = jax.lax.cond( params.restrict[split_rank] <= 0, 
                                lambda _: activated_output, 
                                lambda _: activated_output*params.restrict[split_rank], None)
        
        # Step 4: Compute remaining values
        remaining_value = updated_slice - penalty
        values_padded = jax.lax.dynamic_update_slice(values_padded, remaining_value, start_indices)
        new_values = neuron_states.values.at[:,:,:].set(
                                                values_padded[:, kernel_h_span:max_x+kernel_h_span, kernel_w_span:max_y+kernel_w_span]
                                                )

        # Step 5: Apply pooling and compute the output events 
        nb_valid_elements, out_events, unpooled_matrix = output_to_event_array_with_pooling(activated_output, 
                                                                       start_indices, 
                                                                       new_values.shape,
                                                                       neuron_states.pooling,
                                                                       neuron_states.pool_size,
                                                                       neuron_states.pool_stride)
        # jax.debug.print("rank {}, unpooled matrix {}", rank, unpooled_matrix)
        # jax.debug.print("out shape {}", out_events.shape)
        # jax.debug.print("n state values before {}, after {}, activations {}", jnp.count_nonzero(neuron_states.values), jnp.count_nonzero(new_neuron_states.values), activations)
        # jax.debug.print("rank {} has activated_output shape: {}, out_events shape: {}", rank, activated_output, out_events.shape)
        # jax.debug.print("neuron states: {}, values padded: {}", neuron_states, values_padded)
        
        # Step 6: Update the neuron state
        new_layer_activity = neuron_states.layer_activity.at[   unpooled_matrix[:, 0].astype(jnp.int32), 
                                                                unpooled_matrix[:, 1].astype(jnp.int32), 
                                                                unpooled_matrix[:, 2].astype(jnp.int32)
                                                            ].add(unpooled_matrix[:,3])
        # new_layer_activity = neuron_states.layer_activity
        new_last_sent_iteration = jax.lax.cond(fire, lambda _: iteration, lambda _: neuron_states.last_sent_iteration, None)

        new_neuron_states = ConvNeuronStates(neuron_state=
                                            NeuronStates(values=new_values,
                                                thresholds=neuron_states.thresholds,
                                                input_residuals=neuron_states.input_residuals.at[neuron_idx].add(layer_input),
                                                input_order=neuron_states.input_order,
                                                input_activity=neuron_states.input_activity,
                                                layer_activity=new_layer_activity,
                                                output_activity=neuron_states.output_activity,
                                                last_sent_iteration=new_last_sent_iteration,
                                                input_vector=neuron_states.input_vector,
                                                output_vector=neuron_states.output_vector,
                                                weights_shape=neuron_states.weights_shape,
                                                is_conv=neuron_states.is_conv,
                                            ),
                                            kernel=neuron_states.kernel,
                                            padding=neuron_states.padding,
                                            stride=neuron_states.stride,
                                            previous_layer=neuron_states.previous_layer,
                                            is_conv=neuron_states.is_conv,
                                            pooling=neuron_states.pooling,
                                            pool_size=neuron_states.pool_size,
                                            pool_stride=neuron_states.pool_stride
                                            )
        # jax.debug.print("rank {}, values padded: {}, current slice: {}, updated slice: {}, activated output: {}, remaining: {}, neuron state: {}", rank, values_padded.shape, current_slice.shape, updated_slice.shape, activated_output.shape, remaining_value.shape, neuron_states.values.shape)
        return nb_valid_elements, out_events, new_neuron_states

    def last_input(neuron_states):
        if params.sync_rate == 1:
            C, H, W = params.flat_layer_sizes[split_rank]
            return jnp.array(0), jnp.zeros((C*H*W, 4)), neuron_states

        # For full sync case, fire all neurons that are above the threshold  
        neuron_val = neuron_states.values
        activated_output = activation_func(neuron_states.thresholds, neuron_val)  
        
        # Step 4: Compute remaining values and update the neuron state
        remaining_value = neuron_val - activated_output
        nb_valid_elements, out_events, unpooled = full_matrix_to_event_array_with_pooling(activated_output, activated_output.shape, neuron_states.pooling, neuron_states.pool_size, neuron_states.pool_stride)
        # jax.debug.print("valid el {}, out events {}", nb_valid_elements, jnp.count_nonzero(activated_output))
        # jax.debug.print("{}", iteration)
        # jax.debug.print("out shape {}", out_events.shape)
        new_layer_activity = neuron_states.layer_activity.at[   unpooled[:, 0].astype(jnp.int32), 
                                                                unpooled[:, 1].astype(jnp.int32), 
                                                                unpooled[:, 2].astype(jnp.int32)
                                                            ].add(unpooled[:,3])
        # new_layer_activity = neuron_states.layer_activity

        new_neuron_states = ConvNeuronStates(neuron_state=
                                            NeuronStates(values=remaining_value,
                                                thresholds=neuron_states.thresholds,
                                                input_residuals=neuron_states.input_residuals,#.at[neuron_idx].add(layer_input),
                                                input_order=neuron_states.input_order,
                                                input_activity=neuron_states.input_activity,
                                                layer_activity=new_layer_activity,
                                                output_activity=neuron_states.output_activity,
                                                last_sent_iteration=neuron_states.last_sent_iteration,
                                                input_vector=neuron_states.input_vector,
                                                output_vector=neuron_states.output_vector,
                                                weights_shape=neuron_states.weights_shape,
                                                is_conv=neuron_states.is_conv,
                                            ),
                                            kernel=neuron_states.kernel,
                                            padding=neuron_states.padding,
                                            stride=neuron_states.stride,
                                            previous_layer=neuron_states.previous_layer,
                                            is_conv=neuron_states.is_conv,
                                            pooling=neuron_states.pooling,
                                            pool_size=neuron_states.pool_size,
                                            pool_stride=neuron_states.pool_stride
                                            )
        return nb_valid_elements, out_events, new_neuron_states
    
    nb_valid_elements, out_events, neuron_states = jax.lax.cond(jnp.any(neuron_idx < 0), last_input, regular_input, neuron_states) 
    # jax.debug.print("rank {}, out events shape: {}, first values: {}", rank, out_event.shape, out_event[-5:])
    
    return nb_valid_elements, out_events, neuron_states

#region Forward Pass
@partial(jax.jit, static_argnames=['params', 'layer_computation',])
def conv_predict(params, key, weights, empty_neuron_states, layer_computation, batch_data: jnp.ndarray):
    '''
    CNN inference, each layer sends each event separately in the format: (c, x, y, value)
    -1 means end of data from previous layer
    -2 means placeholder data in the input layer 
    '''
    # jax.debug.print("Rank {} has batch_data shape: {}", rank, batch_data.shape)

    def input_layer(args):
        neuron_states, x = args # x binned is shape (timesteps, channel, height, width)
                                # x not binned is shape (max_nonzero, 4) (x, y, t, c)
        # jax.debug.print("Rank {}, input layer shape: {}", rank, x.shape)
        
        x_p = x
        def send_input(i, carry):
            timestep = carry
            data = x_p[i]
            def send_data(t):
                # combined = jnp.stack([data[3], data[0], data[1], 1.0]) # Sending format (c, x, y, v)
                combined = data

                # jax.debug.print("rank {} sending data {}", rank, combined)
                send(combined, dest=rank+process_per_layer, tag=0, comm=comm)
                return t+1
            
            timestep = jax.lax.cond(
                jnp.any(data != -2),
                send_data,
                lambda _: timestep,
                operand=timestep
            )
            return timestep

        # Initial carry: (timestep=0)
        iteration = jax.lax.fori_loop(0, x_p.shape[0], send_input, (0))

        # Send end signal
        send(jnp.array([-1.0, -1.0, -1.0, -1.0]), dest=rank+process_per_layer, tag=0, comm=comm)
        # jax.debug.print("Rank {}, sent end signal", rank)

        return neuron_states, iteration
    
    def other_layers(args):
        neuron_states, _ = args
        def cond(state): # end of input has been reached -> break the while loop
            _, neuron_idx, _, _= state
            return jnp.all(neuron_idx != -1)
        
        def forward_pass(state):
            neuron_states, _, timestep, iteration = state

            def hidden_layers(args): # Send activation to the next layers
                loop_iterations, activated_output = args

                def send_activation(i, _):
                    combined = activated_output[i]
                    
                    # jax.debug.print("rank {} i: {}, sending {}", rank, i, combined)
                    jax.lax.cond(
                        (combined[3] != 0),
                        lambda _: send(combined, dest=rank+process_per_layer, tag=0, comm=comm),
                        lambda _: [],
                        operand=None
                    )
                    return None

                jax.lax.fori_loop(0, loop_iterations, send_activation, None)
                return None
            
            # Receive neuron values from previous layers and compute the activations
            # input_shape = params.max_kernel + 3
            input_shape = 4 
            input_data = recv(jnp.zeros(input_shape), source=rank-process_per_layer, tag=0, comm=comm)
            # Unpack
            neuron_idx = input_data[:3].astype(jnp.int32) # channel, x, y
            layer_input = input_data[3] # value
            # jax.debug.print("rank {} receving neuron idx {} and value {} at iteration {}", rank, neuron_idx, layer_input, iteration)
            
            loop_iterations, activated_output, new_neuron_states = layer_computation(params, key, neuron_idx, layer_input, weights, neuron_states, iteration)
            # jax.debug.print("rank {}, loop iterations: {}, activated output shape: {}", rank , loop_iterations, activated_output)
            
            neuron_states = new_neuron_states
            
            jax.lax.cond(split_rank == last_rank, lambda _: None, hidden_layers, (loop_iterations, activated_output)) # Don't send if we reach the last layer
            return neuron_states, neuron_idx, timestep, iteration+1
        
        neuron_idx, timestep, iteration =  jnp.zeros(3).astype(jnp.int32), 0, 0
        initial_state = (neuron_states, neuron_idx, timestep, iteration)
        
        # Loop until the rank receives a -1 timestep
        neuron_states, neuron_idx, timestep, iteration = jax.lax.while_loop(cond, forward_pass, initial_state)
        
        # Send -1 to the next rank when all incoming data has been processed
        jax.lax.cond(
            split_rank != last_rank,
            lambda _: send(jnp.array([-1.0, -1.0, -1.0, -1.0]), dest=rank + process_per_layer, tag=0, comm=comm),
            lambda _: [],
            operand=None
        )
        return neuron_states, iteration-1
    
    # Loop over batches, accumulate output values and return them
    def loop_over_batches(_, x):
        neuron_states = empty_neuron_states  
        new_neuron_states, iterations = jax.lax.cond(split_rank==0, input_layer, other_layers, (neuron_states, x))
        
        return None, (new_neuron_states.values, iterations, new_neuron_states)
    _, (all_outputs, all_iterations, all_neuron_states) = jax.lax.scan(loop_over_batches, None, batch_data)    
    
    # Synchronize all ranks before starting the backward pass
    mpi4jax.barrier(comm=comm)
    return all_outputs, all_iterations, all_neuron_states


#region Training helpers
@partial(jax.jit, static_argnames=['params', 'layer_computation', 'conv_layer_sizes'])
def predict_bwd(params, key, conv_layer_sizes, weights, empty_neuron_states, layer_computation, batch_data):
    '''
    B: batch_size
    '''
    all_outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, batch_data)
    
    next_grad= recv(jnp.zeros((batch_part,) + params.flat_layer_sizes[split_rank]), source=rank + process_per_layer, tag=2, comm=comm) # Shape: (B, 128)
    # next_grad = recv(jnp.zeros((batch_part, layer_sizes[split_rank])), source=rank + process_per_layer, tag=2, comm=comm) # Shape: (B, 128)
    # jax.debug.print("Rank {} received next_grad shape: {}", rank, next_grad)

    next_weight_res = jnp.ones((batch_part, params.layer_sizes[split_rank][0], params.layer_sizes[split_rank+1][0])) # Shape: (B, 128, 10)
    # jax.debug.print("Rank {} received next_grad shape: {}, next_weight_res shape: {}", rank, next_grad.shape, next_weight_res.shape)
    (next_weight_res) = jax.lax.cond(split_rank < last_rank - 1, 
                                   lambda _: recv(next_weight_res, source=rank + process_per_layer, tag=3, comm=comm),
                                   lambda _: (next_weight_res), None) 
    # jax.debug.print("Rank {} received next_grad shape: {}", rank, next_weight_res)

    weight_grad, th_grad, weight_res = back_prop(params, all_neuron_states, next_grad, next_weight_res, split_rank)

    if split_rank > 1:
        send_grad = jnp.dot(next_grad, weights.T) # Shape: (B, 128) @ (128, 784) = (B, 784)

        send(send_grad, dest=rank-process_per_layer, tag=2, comm=comm)
        send(weight_res, dest=rank-process_per_layer, tag=3, comm=comm)
    
    # Sparsity loss gradients 
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
    
    scaling = jax.lax.cond(params.sparsity_impact[split_rank] > 0,
                           lambda _: params.sparsity_impact[split_rank] / (all_iterations * batch_part * process_per_layer) ,
                           lambda _: 0.0,
                           None)
    
    input_activity = jnp.sum(all_neuron_states.input_activity, axis=0) # Shape (784)
    layer_activity = jnp.sum(all_neuron_states.layer_activity, axis=0) # Shape (128)
    
    layer_activity = gather_batch(layer_activity, average=False) # Gather the weight gradients from all ranks in the split rank
    input_activity = gather_batch(input_activity, average=False)
    
    sparsity_residuals = scaling * layer_activity # Shape: (128,)
    # jax.debug.print("Rank {}, scaling mean: {}, sparsity_residuals mean: {}, sparsity_residuals sum: {}", rank, scaling, jnp.mean(sparsity_residuals), jnp.sum(sparsity_residuals))
    
    th_sparsity_grad = -sparsity_residuals
    weight_sparsity_grad = jnp.outer(input_activity, sparsity_residuals) # Shape: (784, 128)
    # jax.debug.print("Rank {}, th_sparsity_grad: {}, weight_sparsity_grad: {}", rank, jnp.mean(th_sparsity_grad), np.mean(weight_sparsity_grad))
    
    return all_outputs, iterations, all_neuron_states, (weight_grad, th_grad, weight_sparsity_grad, th_sparsity_grad) 

@partial(jax.jit, static_argnames=['params', 'layer_computation', 'conv_layer_sizes'])
def conv_predict_bwd(params, key, conv_layer_sizes, weights, empty_neuron_states, layer_computation, batch_data):
    all_outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, batch_data)
    
    out_layer_shape = params.flat_layer_sizes[split_rank]
    next_grad = recv(jnp.zeros((batch_part,) + out_layer_shape), source=rank + process_per_layer, tag=2, comm=comm) # Shape: (B, 5, 28, 28) // (B, 5, 14, 14) with pooling
    
    if empty_neuron_states.pooling != "":
        # Upsampling next_grad: from (B, 5, 14, 14) to (B, 5, 28, 28) by copying each element sh*sw number of times
        sh, sw = empty_neuron_states.pool_size
        next_grad = jnp.repeat(next_grad, sh, axis=2)
        next_grad = jnp.repeat(next_grad, sw, axis=3)
        # jax.debug.print("rank {} upsampled weight grad: {}", rank, next_grad.shape)

    if len(params.layer_sizes[split_rank+1]) != 1: # Next layer is convolution layer
        next_weight_shape = (params.layer_sizes[split_rank+1][0], params.layer_sizes[split_rank][0], *params.layer_sizes[split_rank+1][1])
    else:
        next_weight_shape = (np.prod(params.flat_layer_sizes[split_rank]), params.layer_sizes[split_rank+1][0])    

    # jax.debug.print("rank {}, next weight shape: {}", rank, next_weight_shape)
    next_weight_res = jnp.ones(((batch_part,) + next_weight_shape)) # Shape: (B, 128, 10)
    
    # jax.debug.print("rank {} in conv predict has next_weight_res shape: {}", rank, next_weight_res.shape)
    (next_weight_res) = jax.lax.cond(split_rank < last_rank - 1, 
                                    lambda _: recv(next_weight_res, source=rank + process_per_layer, tag=3, comm=comm),
                                    lambda _: (next_weight_res), None) 
    
    
    pad_x, pad_y = params.layer_sizes[split_rank][2]
    strides = params.layer_sizes[split_rank][3]
    def grad_w(x, dy):
        # x: (3, 28, 28)
        # dy: (5, 28, 28)
        # Reshape for conv_general_dilated
        x_padded = jnp.pad(x, ((0,0), (pad_x, pad_x), (pad_y, pad_y))) # shape (3, 28+(pad_x*2), 28+(pad_y*2))
        lhs = x_padded[None, :, :, :]       # (1, 3, 30, 30)
        rhs = dy[None, :, :, :]      # (1, 5, 28, 28)
        
        # Swap axes to match conv weight gradient trick:
        lhs2 = lhs.transpose(1, 0, 2, 3)   # (3, 1, 30, 30)
        rhs2 = rhs.transpose(1, 0, 2, 3)   # (5, 1, 28, 28)
        
        # Convolve
        grad = jax.lax.conv_general_dilated(
            lhs2, rhs2,
            window_strides=strides,
            padding='VALID',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )
        
        # grad: (3, 5, 3, 3) → matches the weights shape
        return grad.transpose(1, 0, 2, 3)
    
    def grad_x(dY, W):
        lhs = jnp.pad(dY, ((0,0), (0,0), (pad_x, pad_x), (pad_y, pad_y))) # shape (B, 2, 28+(pad_x*2), 28+(pad_y*2))
        # Flip kernel spatially
        W_flipped = jnp.flip(W, axis=(2, 3)) 
        rhs = W_flipped.transpose(1, 0, 2, 3) # shape (In, out, k_h, k_w)
        
        return jax.lax.conv_general_dilated(
            lhs=lhs,                # gradient from next layer
            rhs=rhs,                # flipped weights
            window_strides=strides,
            padding='VALID',        # match forward pass
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
        
    input_residuals = all_neuron_states.input_residuals # Shape: (B, 3, 28, 28) 
    # jax.debug.print("rank {} in conv predict has input residuals shape {}, next grad shape: {}, weights shape: {}", rank, input_residuals.shape, next_grad.shape, weights.shape)

    weight_res = jnp.zeros((batch_part,) + weights.shape)

    weight_grad = jax.vmap(grad_w)(input_residuals, next_grad) # Shape: (5, 3, 3, 3)

    th_grad = jnp.zeros(all_neuron_states.values.shape)
    
    if split_rank > 1:
        # send_grad = jnp.zeros((batch_part, *empty_neuron_states.values.shape))
        send_grad = (grad_x)(next_grad, weights)
        # jax.debug.print(f"rank {rank}, weight grad shape: {weight_grad.shape}, send grad shape: {send_grad.shape}")
        send(send_grad, dest=rank-process_per_layer, tag=2,comm=comm)
        send(weight_res, dest=rank-process_per_layer, tag=3,comm=comm)

    weight_sparsity_grad = jnp.zeros(weights.shape)
    th_sparsity_grad = jnp.zeros(conv_layer_sizes[split_rank])
    
    return all_outputs, iterations, all_neuron_states, (weight_grad, th_grad, weight_sparsity_grad, th_sparsity_grad) 

# Define the loss function
@partial(jax.jit, static_argnames=['params', 'layer_computation',])
def loss_fn(params, key, weights, empty_neuron_states, layer_computation, target, batch_data):
    all_outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, batch_data)

    # Compute Loss and loss gradient
    loss, loss_grad = jax.value_and_grad(mean_loss)(all_outputs, target)
    loss_grad /= process_per_layer # Shape (B, 10)
     
    # Compute output gradient and weight gradient
    out_grad, weight_grad = jax.vmap(loss_bpp, in_axes=(None, 0, 0))(weights, all_neuron_states, loss_grad) # Shape (B, 128), (B, 128, 10)
    mean_weight_grad = jnp.mean(weight_grad, axis=0) # Shape: (128, 10)
    mean_weight_grad = jnp.expand_dims(mean_weight_grad, axis=0)  # Shape: (1, 128, 10)
    # Send gradient to previous layers                
    send(out_grad, dest=rank-process_per_layer, tag=2,comm=comm)
    
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)

    total_loss = loss + sparsity_L 

    return (loss, all_outputs, iterations, total_loss), (mean_weight_grad, loss_grad)

def sparsity_loss(params, all_neuron_states, iterations):
    '''
    Compute the sparsity loss based on the input residuals and the weight residuals
    '''
    if params.sparsity_impact[split_rank] <= 0.0:
        return 0, 1, 0
    
    # Gather all the activations at the last layer to compute the sparsity loss
    leader_rank = split_rank * process_per_layer
    # jax.debug.print("Rank {}, activations shape: {}, iterations shape: {}, last layer neuron values: {}", rank, (activations.shape), jnp.mean(iterations.shape), jnp.sum(all_neuron_states.values))

    activations = gather_batch(all_neuron_states.input_residuals, average=False) # Gather the weight gradients from all ranks in the split rank
    iterations = gather_batch(iterations, average=True) # Gather the iterations from all ranks in the split rank
    # jax.debug.print("Rank {}, activations shape: {}, iterations shape: {}, last layer neuron values: {}", rank, (activations.shape), jnp.mean(iterations.shape), jnp.sum(all_neuron_states.values))

    all_iterations = 0.0
    all_activations = 0.0
    sparsity_L = 0.0
    if split_rank != last_rank and rank == leader_rank:
        # jax.debug.print("Rank {}, sending activations {} and iterations {} to the last rank", rank, jnp.sum(activations), jnp.mean(iterations))
        send(jnp.sum(activations), dest=last_rank * process_per_layer, tag=6,comm=comm)
        if rank == 0:
            send(jnp.mean(iterations), dest=last_rank * process_per_layer, tag=6,comm=comm)
    elif split_rank == last_rank and rank == leader_rank:
        for i in range(last_rank):
            # Storing the thresholds
            act_sum = recv(jnp.zeros(1), source=i * process_per_layer, tag=6, comm=comm)
            all_activations = all_activations + act_sum[0] # Sum of all activations in the hidden layers
            
            if i == 0: # Get iterations of input data
                iter_mean = recv(jnp.zeros(1), source=i * process_per_layer, tag=6, comm=comm)
                all_iterations = iter_mean[0]
        all_activations += jnp.sum(activations) # Adding the activations of the last layer

        sparsity_L = all_activations /  (all_iterations * batch_part * process_per_layer)
        # jax.debug.print("Rank {}, sparsity L: {}, all iterations: {}, all activations: {}", rank, sparsity_L, all_iterations, all_activations)
    all_iterations = bcast(all_iterations, root=last_rank*process_per_layer, comm=comm)

    return all_activations, all_iterations, sparsity_L

def split_batch(batch_iterator):
    if rank == 0:
        all_batch_x, all_batch_y = next(batch_iterator)
        all_batch_y = jnp.array(all_batch_y, dtype=jnp.float32)
        all_batch_x = jnp.array(all_batch_x, dtype=jnp.float32)
        # print('shape before pad batch: {}', all_batch_x.shape)
        all_batch_x, all_batch_y = pad_batch(all_batch_x, all_batch_y, batch_part* process_per_layer)
        
        for process in range(process_per_layer):
            if process == 0:
                batch_x = all_batch_x[:batch_part]
                batch_y = all_batch_y[:batch_part]
            else:
                batch_x_to_send = all_batch_x[batch_part*(process):batch_part*(process+1)]
                batch_y_to_send = all_batch_y[batch_part*(process):batch_part*(process+1)]
                # print(f"rank {rank}, Batch_x: {batch_x_to_send.shape}, Batch_y: {batch_y_to_send.shape}")
                
                send(batch_x_to_send, dest=process, tag=4, comm=comm)
                send(batch_y_to_send, dest=process, tag=4, comm=comm)
    else:
        batch_x = recv(jnp.zeros((batch_part, layer_sizes[0])), source=0, tag=4, comm=comm)  
        batch_y = recv(jnp.zeros((batch_part,)), source=0, tag=4, comm=comm) 
    
    return batch_x, batch_y

def gather_batch(data, average=True):
    '''
    Gather all the data from one split_rank onto one rank and resharing the average result to the corresonding split_ranks
    '''
    data = jnp.array(data)
    leader_rank = split_rank * process_per_layer
    if rank == leader_rank:
        avg = data
        for i in range(process_per_layer-1): # Receive the data from all the corresponding ranks in one split rank
            received_data = recv(data, source=rank+i+1, tag=20, comm=comm)
            avg += received_data
        if average:
            avg = avg / process_per_layer
        
        for i in range(process_per_layer-1): # Resharing the average data to all the corresponding ranks
            send(avg, dest=rank+i+1, tag=20, comm=comm)
    else:
        send(data, dest=leader_rank, tag=20, comm=comm)
        avg = recv(data, source=leader_rank, tag=20, comm=comm)
    return avg

def combine_batch(data, average=False):
    '''
    Concatenate all the data from one split_rank onto one rank to reconstruct the batch and resharing the combined result to the corresonding split_ranks
    '''
    data = jnp.array(data)
    # if len(data.shape) < 2:
        
            
    leader_rank = split_rank * process_per_layer
    if rank == leader_rank:
        avg = data
        for i in range(0, process_per_layer-1): # Receive the data from all the corresponding ranks in one split rank
            received_data = recv(data, source=rank+i+1, tag=20, comm=comm)
            avg = jnp.concatenate([avg, received_data], axis=0)
            # if jnp.isnan(avg).any():
            #     jax.debug.print("Rank {}: process: {} NaN detected in avg data: {}", rank, i, avg.shape)
            
        if average:
            # print(f"Rank {rank} combining batches, avg shape: {avg.shape}")
            avg = jnp.mean(avg, axis=0)
            # if jnp.isnan(avg).any():
            #     jax.debug.print("Rank {}: NaN detected in avg data after mean: {}", rank, avg.shape)
            # print(f"Rank {rank} combining batches, avg shape: {avg.shape}")


        for i in range(process_per_layer-1): # Resharing the average data to all the corresponding ranks
            send(avg, dest=rank+i+1, tag=20, comm=comm)
    else:
        send(data, dest=leader_rank, tag=20, comm=comm)
        avg = recv(jnp.zeros((data.shape[1], data.shape[2])), source=leader_rank, tag=20, comm=comm)
        
    return avg


# region TRAINING
def train(params: Params, key, network, weights, empty_neuron_states, layer_computation, opti):     
    """
    tag 0:  forward computation, data format: (previous_layer_neuron_index, neuron_value)
            end of input is encoded with the index -1
    tag 2: backward computation, last layer gradient shape: (layer_sizes[-1], 1)
    tag 3: weight residuals, shape: (layer_sizes[rank], layer_sizes[rank+1])
    tag 4: communication between processes to split the data
    tag 5: weights for storing
    tag 6: activations for sparsity loss
    tag 10: data labels(y)
    tag 20: communications for gathering, sharing and averaging data across split ranks
    """   
    global training_generator
    global validation_generator
    global test_generator
        
    if split_rank == last_rank:
        all_epoch_accuracies = []
        all_validation_accuracies = []
        all_loss = []
    all_mean_iterations = []
    
    if opti == "adam":
        print("adam optimizer selected")
        solver = optax.adam(learning_rate=params.learning_rate)
    elif opti == "adamw":        
        solver = optax.adam(learning_rate=params.learning_rate)
    elif opti == "sgd":
        solver = optax.sgd(learning_rate=params.learning_rate)
    elif opti == "rmsprop":
        solver = optax.rmsprop(learning_rate=params.learning_rate, decay=0.9, eps=1e-8)
    elif opti == "amsgrad":
        solver = optax.amsgrad(learning_rate=params.learning_rate)
    elif opti == "lion":
        print("lion optimizer selected")
        solver = optax.lion(learning_rate=params.learning_rate)
    else: 
        solver = None
    if solver is not None:
        opt_state = solver.init(weights)
    
    th_solver = optax.adam(learning_rate=params.threshold_lr)
    th_opt_state = th_solver.init(empty_neuron_states.thresholds)
    
    mpi4jax.barrier(comm=comm)
    start_time = time.time()

    for epoch in tqdm(range(params.num_epochs)):
        key, subkey = jax.random.split(key) 

        if split_rank == last_rank:
            epoch_correct = 0
            epoch_total = 0
            epoch_loss = []
            
        epoch_iterations = []
        if split_rank == 0:
            batch_iterator = None
            if rank == 0:
                batch_iterator = iter(training_generator)
            
        for i in tqdm(range(total_train_batches)):
            neuron_states = empty_neuron_states
            # threshold_grad = 0.0
            if split_rank == 0:
                # print("batch", i)
                batch_x, batch_y = split_batch(batch_iterator)
                # print(batch_y.shape, type(batch_y), batch_y)
                send(batch_y, dest=last_rank * process_per_layer + rank, tag=10,comm=comm) # Destination rank: last_rank * process_per_layer + rank

                outputs, iterations, all_neuron_states = (conv_predict)(params, subkey, weights, neuron_states, layer_computation, jnp.array(batch_x))
                all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
            else:
                if split_rank==last_rank:
                    # Receive y
                    y = recv(jnp.zeros((batch_part,)), source=rank - (last_rank * process_per_layer), tag=10, comm=comm)  # Source rank opposite operation: rank - (last_rank * process_per_layer)
                    y_encoded = jnp.array(one_hot_encode(y, num_classes=layer_sizes[-1][0]))
                    # print("encoded y: ", y, y_encoded.shape, y_encoded)              
                    (loss, outputs, iterations, total_loss), gradients = (loss_fn)(params, subkey, weights, neuron_states, layer_computation, y_encoded, jnp.zeros((batch_part, 1, 4)))

                    epoch_loss.append(loss)
                    
                    weight_grad = gradients[1]
                                        
                    valid_y, batch_correct = accuracy(i, outputs, y, iterations, False)                 
                    
                    # print(f"Batch correct: {batch_correct}/{batch_size}")

                    epoch_correct += batch_correct
                    epoch_total += valid_y.shape[0]
                    # weight_grad = gather_batch(weight_grad, average=True)
                    weight_grad = combine_batch(weight_grad, average=True) # Gather the weight gradients from all ranks in the split rank
                else:
                    bwd_fn = predict_bwd
                    if empty_neuron_states.is_conv:
                        bwd_fn = conv_predict_bwd
                    
                    outputs, iterations, all_neuron_states, grads = (bwd_fn)(params, subkey, network.conv_layer_sizes, weights, neuron_states, layer_computation, jnp.zeros((batch_part, 1, 4)))
                    weight_grad, threshold_grad, weight_sparsity_grad, threshold_sparsity_grad = grads
                    # print(f"rank {rank}, weight_res: {weight_res[0].tolist()}, shape: {weight_res.shape}")
                    # print("rank weight grad shape: ", rank, weight_grad.shape)
                    # print(f"Rank {rank} finished predict_bwd for batch {i}, outputs shape: {outputs.shape}, iterations: {iterations.shape}, weight_grad shape: {weight_grad.shape}")
                    threshold_grad = gather_batch(threshold_grad, average=True) # Gather the weight gradients from all ranks in the split rank

                    # weight_grad = gather_batch(weight_grad, average=True)
                    weight_grad = combine_batch(weight_grad, average=True) # Gather the weight gradients from all ranks in the split rank
                    
                    if params.sparsity_impact[split_rank] > 0:
                        weight_grad = weight_grad + weight_sparsity_grad
                        threshold_grad = threshold_grad + threshold_sparsity_grad
                    
                    # Update thresholds                    
                    if params.threshold_lr != 0:
                        # print(f"average threshold grad: {jnp.mean(threshold_grad)}")
                        th_updates, th_opt_state = solver.update(threshold_grad, th_opt_state, empty_neuron_states.thresholds)
                        empty_neuron_states.thresholds = jax.nn.sigmoid(optax.apply_updates(empty_neuron_states.thresholds, th_updates))
                    # print(empty_neuron_states.thresholds)
                
                # print("Rank {}, batch {}, mean weight_grad: {}, max weight_grad: {}, min weight_grad: {}".format(rank, i, jnp.mean(weight_grad), jnp.max(weight_grad), jnp.min(weight_grad)))
                # Update weights
                if solver is not None:
                    # Optax optimizer
                    # continue
                    updates, opt_state = solver.update(weight_grad, opt_state, weights)
                    weights = optax.apply_updates(weights, updates)
                else:                
                    # Basic GD
                    weights -= params.learning_rate * weight_grad 
            # if i > 3:
            #     break
            epoch_iterations.append(iterations)
        epoch_iterations = jnp.array(epoch_iterations).flatten()
        mean = jnp.mean(epoch_iterations)
        all_mean_iterations.append(mean)
        all_mean_iterations = gather_batch(all_mean_iterations)
        all_mean_iterations = all_mean_iterations.tolist()
        
        if split_rank != 0:
            jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points and a mean threshold of {}", rank, mean, epoch_iterations.shape[0], jnp.mean(empty_neuron_states.thresholds))
        
        # Inference on the validation set
        val_accuracy, val_mean, _ = batch_predict(params, key, network, weights, empty_neuron_states, layer_computation, dataset="val", save=False, debug=False)
        # val_accuracy, val_mean = 0, 0
        epoch_accuracy = 0.0
        if split_rank == last_rank:
            # Store loss values
            mean_loss = jnp.mean(jnp.array(epoch_loss))
            all_loss.append(mean_loss)
            mean_loss = gather_batch(mean_loss)

            # Store training and validation accuracies
            epoch_accuracy = epoch_correct / epoch_total
            all_epoch_accuracies.append(epoch_accuracy)
            all_validation_accuracies.append(val_accuracy)
            all_epoch_accuracies = gather_batch(all_epoch_accuracies)
            all_validation_accuracies = gather_batch(all_validation_accuracies)
            all_epoch_accuracies, all_validation_accuracies = all_epoch_accuracies.tolist(), all_validation_accuracies.tolist()
            if rank == size-1:
                jax.debug.print("Epoch {} , Training Accuracy: {:.2f}%, Validation Accuracy: {:.2f}%, mean loss: {}, mean val iterations: {}", epoch, all_epoch_accuracies[-1] * 100, val_accuracy * 100, mean_loss, val_mean)
                jax.debug.print("----------------------------\n")
        epoch_accuracy = bcast(epoch_accuracy, root=size-1, comm=comm)
        if epoch_accuracy >= 0.9999:
            break
    # Inference on the test set
    test_accuracy, test_mean, _ = batch_predict(params, key, network, weights, empty_neuron_states, layer_computation, dataset="test", save=False, debug=False)
    # test_accuracy = 0
    
    # Gather the weights and iteration values at the last layer
    layer_weights_sizes = []
    for layer in network.layers:
        layer_weights_sizes.append(layer.weights_shape)
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_iter_th(network, layer_weights_sizes, weights, jnp.array(all_mean_iterations), empty_neuron_states.thresholds)
    


    # Synchronize all MPI processes again
    mpi4jax.barrier(comm=comm)
    end_time = time.time()
    
    MAX_LEN = 256
    result_path = jnp.zeros(MAX_LEN, dtype=jnp.uint8)
    if rank == last_rank * process_per_layer:
        # Execution time
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.6f} seconds")
        result_path_str = store_training_data(
                            size,
                            network, 
                            "train",
                            all_epoch_accuracies, 
                            all_validation_accuracies, 
                            test_accuracy,
                            execution_time,
                            all_iteration_mean,
                            weights_dict,
                            all_loss, 
                            thresholds_dict,
                            opti,
                            "CNN")
        
        encoded = np.frombuffer(result_path_str.encode("utf-8"), dtype=np.uint8)
        if encoded.size > MAX_LEN:
            raise ValueError("result_path too long")
        padded = np.pad(encoded, (0, MAX_LEN - encoded.size), constant_values=0)
        result_path = jnp.array(padded)
    result_path = bcast(result_path, root=last_rank*process_per_layer, comm=comm)
    result_path = bytes(result_path).decode("utf-8").rstrip("\x00")
    mpi4jax.barrier(comm=comm)

    return result_path

def gather_w_iter_th(network, layer_weights_sizes, weights, mean_iterations, thresholds):
    # Gather all the weights and iteration values at the last layer to store them
    leader_rank = split_rank * process_per_layer

    weights_dict = {}
    all_iteration_mean = []
    thresholds_dict = {}
    if split_rank != last_rank and rank == leader_rank:
        send(mean_iterations, dest=last_rank * process_per_layer, tag=5,comm=comm)
        if split_rank != 0:
            send(weights, dest=last_rank * process_per_layer, tag=5,comm=comm)
            send(thresholds, dest=last_rank * process_per_layer, tag=5,comm=comm)
    elif split_rank == last_rank and rank == leader_rank:
        for i in range(last_rank):
            # Storing mean iterations
            iter_mean = recv(mean_iterations, source=i * process_per_layer, tag=5, comm=comm)
            all_iteration_mean.append(iter_mean)
            if i == 0:
                continue
            
            # Storing the weights 
            w = recv(jnp.zeros(layer_weights_sizes[i]), source=i * process_per_layer, tag=5, comm=comm)   
            weights_dict[f"layer_{i}"] = w.tolist()
            
            # Storing the thresholds
            thr = recv(jnp.zeros(network.conv_layer_sizes[i]), source=i * process_per_layer, tag=5, comm=comm)
            thresholds_dict[f"thresholds_{i}"]= thr.tolist()
            
        all_iteration_mean.append(mean_iterations)  # Append the mean iterations of the last layer
        weights_dict[f"layer_{last_rank}"] = weights.tolist()
        all_iteration_mean = all_iteration_mean[1:] # Don't keep the value of the input layer
        print("all iteration mean: rank", rank, all_iteration_mean)

    return weights_dict, all_iteration_mean, thresholds_dict


# region Inference loop
def batch_predict(params, key, network, weights, empty_neuron_states, layer_computation, dataset:str="train", save=True, debug=True):    
    global training_generator
    global validation_generator
    global test_generator    
    
    mpi4jax.barrier(comm=comm)
    start_time = time.time()
    
    if dataset == "train":
        total_batches = total_train_batches
        if split_rank == 0:
            batch_iterator = None
            if rank == 0:
                print(f"Inference on the training set...")
                batch_iterator = iter(training_generator)
    elif dataset == "val":
        total_batches = total_val_batches
        if split_rank == 0:
            batch_iterator = None
            if rank == 0:
                print(f"Inference on the validation set...")
                batch_iterator = iter(validation_generator)
    elif dataset == "test":
        total_batches = total_test_batches
        if split_rank == 0:
            batch_iterator = None
            if rank == 0:
                print(f"Inference on the test set...")
                batch_iterator = iter(test_generator)
    else:
        print("INVALID DATASET")
        return
        
    if split_rank == last_rank:
        epoch_correct = 0
        epoch_total = 1
            
    epoch_iterations = []
    for i in tqdm(range(total_batches)):
        neuron_states = empty_neuron_states
        if split_rank == 0:                 
            batch_x, batch_y = split_batch(batch_iterator)
            # print(f"batch {i}")
            # print(f"rank {rank} batch x shape: {batch_x.shape}, batch y shape: {batch_y.shape}, batch y: {batch_y}")
            
            # input_tensor = jnp.arange(1, 26, dtype=jnp.float32).reshape(1, 1, 5, 5)

            # batch, in_ch, h, w = input_tensor.shape
            # all_events = []

            # for b in range(batch):
            #     events = []
            #     for c in range(in_ch):
            #         for x in range(h):
            #             for y in range(w):
            #                 events.append((c, x, y, float(input_tensor[b, c, x, y])))
            #     all_events.append(events)

            # # Convert to array with batch dimension
            # events_array = jnp.array(all_events)  # shape (batch, N_events, 4)

            # print("Events shape:", events_array.shape)
            # print(events_array)
            # batch_x = events_array
            # batch_x = input_data[None, None, :, :]
            # outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, jnp.array(batch_x))
            # break

            outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, jnp.array(batch_x))

            # Send label to the last layer
            send(batch_y, dest=last_rank * process_per_layer + rank, tag=10,comm=comm)
        else:
            # outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, jnp.zeros((batch_part, layer_sizes[0])))
            batch_data = jnp.zeros((batch_part, 1, 4))

            outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, batch_data)
            # jax.debug.print("Rank {} All neuron states values shape: {}, output shape : {}", rank, all_neuron_states.values.shape, outputs.shape)

            if split_rank == last_rank:
                y = recv(jnp.zeros((batch_part,)), source=rank - (last_rank * process_per_layer), tag=10, comm=comm)   
                
                valid_y, batch_correct = accuracy(i, outputs, y, iterations, print=False)                 
                
                epoch_correct += batch_correct
                epoch_total += valid_y.shape[0]
        epoch_iterations.append(iterations)
        # jax.debug.print("Rank {}, iterations: {}", rank, iterations)
        # if i > 3:
        #     break
    
    # print(f"Shape iterations before flattening: {jnp.array(epoch_iterations).shape}")
    epoch_iterations = jnp.array(epoch_iterations).flatten()
    mean = jnp.mean(epoch_iterations)
    # print(f"Rank {rank} finished epoch with mean {mean} with {epoch_iterations.shape} iterations")

    if split_rank != 0:
        mean = gather_batch(mean)
    # jax.debug.print("Rank {}, all iterations shape: {}", rank, (epoch_iterations.shape[0]))
    
    if rank != 0 and debug:
        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, epoch_iterations.shape[0]*process_per_layer)
    
    epoch_accuracy = -1.0
    if split_rank == last_rank:
        epoch_accuracy = epoch_correct / epoch_total
        epoch_accuracy = gather_batch(epoch_accuracy)
        if debug:
            jax.debug.print("Epoch Accuracy: {:.2f}%", epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")
    
    
    layer_weights_sizes = []
    for layer in network.layers:
        layer_weights_sizes.append(layer.weights_shape)
    print(f"rank {rank}: {layer_weights_sizes}")
    
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_iter_th(network, layer_weights_sizes, weights, mean, empty_neuron_states.thresholds)
    # jax.debug.print("rank {} all iterations mean: {}, shape: {}", rank, all_iteration_mean, (all_iteration_mean.shape))
    
    # Synchronize all MPI processes again
    mpi4jax.barrier(comm=comm)
    end_time = time.time()

    if rank == last_rank * process_per_layer:
        execution_time = end_time - start_time

        if debug:            
            print(f"Execution Time: {execution_time:.6f} seconds")
        if save:
            accuracies = {"train": [-1], "val": [-1], "test": [-1]}
            if dataset in accuracies:
                accuracies[dataset] = [epoch_accuracy]

            store_training_data(size,
                                network, 
                                "inference",
                                accuracies["train"], 
                                accuracies["val"], 
                                accuracies["test"][0],
                                execution_time,
                                all_iteration_mean,
                                weights_dict,
                                [],
                                thresholds_dict,
                                None,
                                "CNN")
    return epoch_accuracy, mean, end_time - start_time

def list_to_tuple_deep(obj):
    if isinstance(obj, list):
        return tuple(list_to_tuple_deep(item) for item in obj)
    return obj

# region Main
def rerun_init(data_file_path, new_epoch_nb, dataset, th_lr=0, th_impact=0, async_layer=-1, max_kernel=0, network=None):
    with open(data_file_path, "r") as f:
        stored_data = json.load(f)

    load_file = stored_data["loadfile"]
    shuffle = stored_data["shuffle data"]
    shuffle_input = stored_data["shuffle input"]
    firing_nb = stored_data["firing number"]
    sync_rate = stored_data["synchronization rate"]
    layer_sizes = list_to_tuple_deep(stored_data["layer_sizes"])
    batch_size = stored_data["batch_size"]
    learning_rate = stored_data["learning rate"]
    init_thresholds = stored_data["thresholds"]["thresholds_1"][0][0][0]
    threshold_dict = stored_data["thresholds"]
    restrict = tuple(stored_data["restrict"])
    sparsity_impact = stored_data["threshold impact"]
    threshold_lr = stored_data["threshold lr"]
    weights_dict = stored_data["weights"]

    params = Params(
        dataset=dataset,
        random_seed=random_seed,
        layer_sizes=layer_sizes, 
        init_thresholds=init_thresholds, 
        num_epochs=new_epoch_nb, 
        learning_rate=learning_rate, 
        batch_size=batch_size,
        load_file=load_file,
        shuffle_activations=shuffle,
        restrict=restrict,
        firing_nb=firing_nb,
        sync_rate=sync_rate,
        max_nonzero=max_nonzero,
        shuffle_input=shuffle_input,
        threshold_lr=th_lr,
        sparsity_impact=th_impact,
        rerun=data_file_path,
        async_layer=async_layer,
        max_kernel=max_kernel,
        flat_layer_sizes=()
    )

    if split_rank > 0:
        thresholds = None
        if split_rank < last_rank:
            thresholds = jnp.array(threshold_dict["thresholds_"+str(split_rank)])
        weights = jnp.array(weights_dict["layer_"+str(split_rank)])
        neuron_states = network.rerun(thresholds)
    else:
        weights = jnp.zeros((1,1,1,1))
        neuron_states = network.rerun(None)
                
    return params, weights, neuron_states

def get_split_rank():
    global split_rank 
    global process_per_layer
    global last_rank
    global batch_part
    
    last_rank = len(layer_sizes)-1
    process_per_layer = size // (last_rank+1)
    split_rank = rank // process_per_layer
    batch_part = batch_size // process_per_layer

    print(f"Rank {rank}, split rank: {split_rank}, batch part: {batch_part}, process per layer: {process_per_layer}, last rank: {last_rank}")

if __name__ == "__main__":
    STOP_flag = 0
    random_seed = 42
    key = jax.random.key(random_seed)
    
    dataset = 'mnist'
    # dataset = 'shd'
    # dataset = 'nmnist'
    
    # Network structure and parameters
    all_layers = [] 
    # MNIST layers
    # all_layers.append((28*28, 32, 10))
    # all_layers.append((28*28, 64, 10))    
    # all_layers.append((28*28, 128, 10))    
    
    # SHD layers 
    # all_layers.append((700, 128, 20))    
    # all_layers.append((700, 64, 20))    
    # all_layers.append((700, 32, 20))    
    
    # NMNIST layers
    # all_layers.append(( (2, 34, 34), # (channel, height, width)
    #                     (3, (3,3), (1,1), (1,1)), # (out_channel, kernel_size, padding, stride)
    #                     # (2, (5,5), (2,2), (1,1)), 
    #                     # (64,),
    #                     (32,), # Fully connected layer
    #                     (10,)))
    
    # MNIST layers
    # all_layers.append(( (1, 28, 28),                  # (channel, height, width)
    #                     (3, (3,3), (1,1), (1,1)),   # (out_channel, kernel_size, padding, stride)
    #                     (5, (3,3), (1,1), (1,1)), 
    #                     # (64,),
    #                     (128,), # Fully connected layer
    #                     (10,)))
    
    all_layers.append(( (1, 28, 28),                                    # (channel, height, width)
                        (3, (3,3), (1,1), (1,1), ""),                       # Conv layer                (out_channel, kernel_size, padding, stride)
                        (5, (3,3), (1,1), (1,1), "max", (2,2), (2,2)),  # Conv layer With Pooling   (out_channel, kernel_size, padding, stride, pooling type, pool_size, pool_stride)
                        # (64,),
                        (128,), # Fully connected layer
                        (10,)))
    
    # all_layers.append(( (1, 28, 28), # (channel, height, width)
    #                     (32, (3,3), (1,1), (1,1)), # (out_channel, kernel_size, padding, stride)
    #                     (64, (3,3), (1,1), (1,1)), 
    #                     # (64,),
    #                     (32,), # Fully connected layer
    #                     (10,)))

    # VGG 16
    all_layers.append(( (1, 28, 28),                                    # (channel, height, width)
                        (64, (3,3), (1,1), (1,1), ""),                  # Conv layer With Pooling   (out_channel, kernel_size, padding, stride, pooling type, pool_size, pool_stride)
                        (64, (3,3), (1,1), (1,1), "max"),
                        
                        (128, (3,3), (1,1), (1,1), ""),   
                        (128, (3,3), (1,1), (1,1), "max"),   
                        
                        (256, (3,3), (1,1), (1,1), ""),   
                        (256, (3,3), (1,1), (1,1), ""),   
                        (256, (3,3), (1,1), (1,1), "max"),   

                        (512, (3,3), (1,1), (1,1), ""),   
                        (512, (3,3), (1,1), (1,1), ""),   
                        (512, (3,3), (1,1), (1,1), "max"),   

                        (512, (3,3), (1,1), (1,1), ""),   
                        (512, (3,3), (1,1), (1,1), ""),   
                        (512, (3,3), (1,1), (1,1), "max"),   

                        (4096,),
                        (4096,), # Fully connected layer
                        (10,)))

    layer_sizes = all_layers[0]
    
    max_kernel = 0
    for layer in layer_sizes:
        if len(layer) == 4:
            _, k, _, _ = layer
            k_prod = (k[0]+k[0]-1) * (k[1]+k[1]-1)
            if k_prod > max_kernel:
                max_kernel = k_prod
    
    best = False
    load_file = False
    batch_size = 36
    restrict = (0,) * len(layer_sizes)

    
    if size % len(layer_sizes) != 0:
        print(f"Error: layer_sizes ({len(layer_sizes)}) must match number of MPI ranks ({size})")
        sys.exit(1)
    
    get_split_rank() # Compute the split rank for training/inference with multiple processes per batch

    if batch_size % process_per_layer != 0:
        print(f"Error: one batch ({batch_size}) must be divisible by the number of processes per layer ({process_per_layer})")
        sys.exit(1)
    
    
    init_thresholds = 0.0 #float(jnp.sqrt(2))
    
    rerun = "network_results/training/42_ep20_batch36_784_128_64_10_acc0.967_adam_.json"
    rerun = "network_results/nmnist/training/42_ep5_batch36_(2, 34, 34)_(1, (3, 3), (1, 1), (1, 1))_(10,)_acc0.867_adam_.json"
    rerun = None
    async_layer = -1
    # async_layer = 1
    
    cont = True
    # for i in range(5):#[0.0001, 0.001, 0.01]: #TODO rerun sigmoid 4 because multi layer training missed the dependency between 2 hidden layers' activations
        # for th_lr in [0.0001, 0.001, 0.01]:
    while cont:
            # Initialize parameters (input data for rank 0 and weights for other ranks)
            total_train_batches, total_val_batches, total_test_batches = 0, 0, 0
            if split_rank != 0:
                batch_iterator = None
                max_nonzero = 0
            if split_rank == 0:
                max_nonzero = 0
                if rank == 0:
                    # Load the data 
                    match dataset:
                        case "mnist":
                            loader = partial(mnist_loader_manual, CNN_preproces=True)
                        case "shd":
                            loader = torch_SHD_loader
                        case "nmnist":
                            loader = torch_nmnist_loader
                        case _:
                            raise ValueError(f"Unknown dataset: {dataset}")
                    # Load the data 
                    (training_generator, total_train_batches), (validation_generator, total_val_batches), (test_generator, total_test_batches), max_nonzero = loader(batch_size=batch_size, shuffle=False)
                    
                    batch_x, batch_y = next(iter(training_generator))
                    print("Batch data shapes: ",batch_x.shape, batch_y.shape)
                    if len(batch_x.shape) == 5 and batch_x.shape[2:] != layer_sizes[0]:
                        STOP_flag = 1
            
            STOP_flag = bcast(jnp.array([STOP_flag]), root=0, comm=comm)
            if STOP_flag:
                print(f"Error: make sure that the input layer has the same dimensions as the data.")
                sys.exit(1)
                
            # Broadcast total_batches to all other ranks
            total_train_batches, total_val_batches, total_test_batches = bcast(jnp.array([total_train_batches, total_val_batches, total_test_batches]), root=0 , comm=comm)                
            max_nonzero = bcast(jnp.array([max_nonzero]), root=0 , comm=comm)
            max_nonzero = max_nonzero.tolist()[0]
            
            params = Params(
                dataset=dataset,
                random_seed=random_seed,
                layer_sizes=layer_sizes, 
                init_thresholds=init_thresholds, 
                num_epochs=10, 
                learning_rate=0.0001, 
                batch_size=batch_size,
                load_file=load_file,
                shuffle_activations=False,
                restrict=restrict,
                firing_nb=2000,
                sync_rate=1000000,
                max_nonzero=max_nonzero,
                shuffle_input=False,
                threshold_lr=0.0, 
                sparsity_impact=(0.0, 0.0, 0.0, 0.0, 0.0), # Beta sparse
                rerun="",
                async_layer=async_layer,
                max_kernel=max_kernel,
                flat_layer_sizes=()
            )
            key, subkey = jax.random.split(key) 
            network = Network.build(params, key, layer_sizes=layer_sizes, flat_layer_sizes=(), conv_layer_sizes=(), th_bias=0.0)
            weights = network.init_weights()
            empty_neuron_states = network.layers[split_rank]

            if rerun is not None:
                new_epoch_number = 10 # Number of training epoch to run again
                th_lr, beta = 0.0, 0.0
                
                # if async_layer == -1 or async_layer >= last_rank:
                #     cont = False
                #     continue
                # else:
                #     async_layer += 1
                
                # if i % 2:
                #     new_epoch_number = 1
                #     beta = 0.01
                
                params, weights, empty_neuron_states = rerun_init(rerun, new_epoch_number, dataset, th_lr, beta, async_layer=async_layer,max_kernel=max_kernel, network=network)
                if len(layer_sizes) != len(params.layer_sizes):
                    print(f"Error: rerun file {rerun} has different layer sizes than the current network structure {layer_sizes}.")
                    sys.exit(1)

            # print(rank, empty_neuron_states.values.shape)
            params = dataclasses.replace(params, flat_layer_sizes=network.flat_layer_sizes)
            # network.params = params

            if rank == 0:
                print(f"Number of training batches: {total_train_batches}, validation batches: {total_val_batches}, test batches: {total_test_batches}")
                print(params)
                        
            t = 100
            all_time = 0
            # for i in range(t):
            #     _, _, ex_time = batch_predict(params, key, weights, empty_neuron_states, "val", save=False, debug=True)
            #     all_time += ex_time
            # print("average execution time : {}", all_time/t)
            layer_computation = fc_layer_computation
            if empty_neuron_states.is_conv:
                layer_computation = conv_layer_computation
            
            # print(f"rank {rank}, is conv: {empty_neuron_states.is_conv}, weights shape: {weights.shape}")
            # batch_predict(params, key, network, weights, empty_neuron_states, layer_computation, 'test', save=True, debug=True)
            result_path = train(params, key, network, weights, empty_neuron_states, layer_computation, "adam")
            # rerun = result_path
            # print(rerun)
            break
