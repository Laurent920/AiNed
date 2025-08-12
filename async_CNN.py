from asyncio import gather
from math import e
import math
import os

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

import tree_math
import dataclasses
from typing import Generic, Any, Union, TypeVar, Tuple
import time
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

import mpi4jax
from mpi4jax import send, recv, bcast

from data_helpers.mnist_helper import torch_mnist_loader_manual
from data_helpers.iris_species_helper import torch_iris_loader
from data_helpers.network_helper import one_hot_encode
from data_helpers.nmnist_helper import torch_nmnist_loader
from data_helpers.shd_helper import torch_SHD_loader

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

@tree_math.struct
class Neuron_states:
    '''                            
    Stores the state of the neurons in a layer.
    
    Shapes are referenced for a layer (784, 128).
         
    Attributes:
        values (jnp.ndarray):                    Current state of the neurons in the layer, shape: (layer_sizes[rank],) __ (128,)
        thresholds (jnp.float32):                An array of thresholds, one per neuron, shape: (layer_sizes[rank],) __ (128,)
        input_residuals (jnp.ndarray):           Sum of all inputs for each neuron, shape: (layer_sizes[rank-1],) __ (784,)
        weight_residuals (dict[str, jnp.ndarray]):
            - "input order"                      Set input neuron to the iteration at which the input is received to record the order of input received, shape: (layer_sizes[rank-1],) __ (784,)
            - "input activity"                   Count the number of times a input neuron fired, shape: (layer_sizes[rank-1],) __ (784,)
            - "layer activity"                   Count the number of times a neuron activated in this layer, only used for restrict parameter and threshold, shape: (layer_sizes[rank],) __ (128,)
            - "output activity"                  For each input neuron stores the hidden neurons that fire, shape: (layer_sizes[rank-1], layer_sizes[rank]) __ (784, 128)
        last_sent_iteration (int):               The last iteration at which this layer sent data, used to determine if the layer should fire when using synchronization
        weights_shape (tuple[int, ...]):         Shape of the weights for this layer, used to initialize the weights, shape: (layer_sizes[rank-1], layer_sizes[rank]) __ (784, 128)
    '''
    values: jnp.ndarray                                  
    thresholds: jnp.ndarray           
    input_residuals: jnp.ndarray      
    weight_residuals: dict[str, jnp.ndarray]
    last_sent_iteration: int
    weights_shape: tuple[int, ...]
    is_conv: bool = False

@tree_math.struct
class Conv_Neuron():
    """
    Conv_Neuron extends Neuron_states to include convolutional properties.

    It adds kernel_size, padding, stride which need to be 2 dimensional tuples containing the height and the width.

    Attributes:
        kernel (tuple[int, int]):                The kernel size for the convolutional layer, shape: (2,)
        padding (tuple[int, int]):               The padding for the convolutional layer, shape: (2,)
        stride (tuple[int, int]):                The stride for the convolutional layer, shape: (2,)
        previous_layer (jnp.ndarray):            Records the last received input from the previous layer.
    """
    neuron_state: Neuron_states
    kernel: tuple[int, int]
    padding: tuple[int, int]
    stride: tuple[int, int]
    previous_layer: jnp.ndarray
    max_pool: tuple[int, int] = (2, 2)
    is_conv: bool = True
    
    def __getattr__(self, name):
        # Called only when the attribute is not found in Conv_neuron, it will look for the attribute in neuron_state
        return getattr(self.neuron_state, name)

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
    shuffle: bool           # Shuffle the dataset
    restrict: int           # The amount of times a single neuron can fire accross all inputs, if negative then no restriction
    firing_nb: int          # The maximum number of neurons that can fire for one input at each layer
    sync_rate: int          # The number of inputs that needs to be accumulated before firing  
    max_nonzero: int
    shuffle_input:bool      # Shuffle the data in each layer to simulate async individual neurons
    threshold_lr: float
    threshold_impact: float
    rerun: str
    async_layer: int # The layer that is training asynchronously while all other layers are training sync, if -1 then all layers are async
    max_kernel: int # The maximum size of flattened kernel
    flat_layer_sizes: tuple[int, ...]

#region Initialization
@tree_math.struct
@dataclasses.dataclass(frozen=True)
class Network:
    '''
    A class representing the neural network architecture.
    
    It initializes the layers depending on the layer_sizes input and stores either a convolution layer or a fully connected layer.
    
    It then initializes the weights/filters for each layer.
    '''
    def __init__(self, key, params:Params, layer_sizes: tuple[tuple[int, ...]], debug=False):
        '''
        Parameters:
            key: random key 
            params: Params object containing the parameters for the network
            layer_sizes (tuple of tuples): 
                        - for a fully connected layer each tuple contains a single integer representing the number of neurons in that layer
                        - for a convolutional layer each tuple contains (out_chan, kernel, padding, stride) representing the output channel, kernel size, padding and stride.
        
        '''
        self.params = params
        self.layers = []        
        previous_layer = 0  # Initialize previous layer as zero
        for i, layer in enumerate(layer_sizes):
            if len(layer) == 1:
                if i == 0:
                    previous_layer = 1 # Use the smallest array because residuals are not needed for input layer
                else:
                    if isinstance(previous_layer, int):
                        pass
                    if isinstance(previous_layer, jax.Array):
                        flat_previous_layer =  previous_layer.flatten().size
                        if rank == 0 and debug:
                            print(f"rank {rank}, Previous layer: {previous_layer.shape}, flattened: {flat_previous_layer}")
                        previous_layer = flat_previous_layer
                        
                key, subkey = jax.random.split(key) 
                thresholds = jax.random.normal(subkey, layer) * params.init_thresholds

                empty_neuron_states = Neuron_states(
                                        values=jnp.zeros(layer), 
                                        thresholds=thresholds, 
                                        input_residuals=jnp.zeros((previous_layer,)),
                                        weight_residuals={"input order": jnp.full((previous_layer,), -1, dtype=int), 
                                                        "input activity": jnp.full((previous_layer,), 0, dtype=int), 
                                                        "layer activity": jnp.zeros((layer[0],), dtype=int), 
                                                        "output activity": jnp.zeros((previous_layer, layer[0]))},
                                        last_sent_iteration=0,
                                        weights_shape=(previous_layer, layer[0])
                                        )
                self.layers.append(empty_neuron_states)
                previous_layer = layer[0]
            else:
                if i == 0:
                    previous_layer = jnp.zeros(1)
                    values = jnp.zeros(layer)
                    out_chan, kernel, padding, stride = 1, (0,0), (0,0), (0,0) # Values used as placeholders for the input layer
                else:
                    out_chan, kernel, padding, stride = layer
                    in_shape = previous_layer.shape
                    h_out = (in_shape[1] + 2 * padding[0] - kernel[0]) // stride[0] + 1
                    w_out = (in_shape[2] + 2 * padding[1] - kernel[1]) // stride[1] + 1
                    if rank == 0 and debug:
                        print(f"rank {rank}, previous layer shape: {previous_layer.shape}, out shape: {(out_chan, h_out, w_out)}, kernel: {kernel}, padding: {padding}, stride: {stride}")
                    values = jnp.zeros((out_chan, h_out, w_out))  # Initialize values for convolutional layer
                
                in_chan = previous_layer.shape[0]
                key, subkey = jax.random.split(key) 
                thresholds = jax.random.normal(subkey, values.shape) * params.init_thresholds
                empty_conv_neuron = Conv_Neuron(
                                    neuron_state=Neuron_states(
                                                    values=values, 
                                                    thresholds=thresholds, 
                                                    input_residuals=jnp.zeros(previous_layer.shape),
                                                    weight_residuals={"input order": jnp.full(previous_layer.shape, -1, dtype=int), 
                                                                    "input activity": jnp.full(previous_layer.shape, 0, dtype=int), 
                                                                    "layer activity": jnp.zeros(values.shape, dtype=int), 
                                                                    "output activity": previous_layer},
                                                    last_sent_iteration=0,
                                                    weights_shape=(out_chan, in_chan, kernel[0], kernel[1])),  
                                    kernel=kernel,
                                    padding=padding,
                                    stride=stride,
                                    previous_layer=previous_layer
                                    )   
                
                self.layers.append(empty_conv_neuron)
                previous_layer = values
        self.key = key
        self.layers = tuple(self.layers) # Convert to tuple to allow jit

    def init_weights(self):
        '''
        Initialize the weights for each layer based on the layer sizes.
        
        Returns the weights correponding to the MPI split_rank.
        ''' 
        weights = init_params(self.key, self.layers, self.params.load_file)
        print(f"Rank {split_rank} initialized weights: {weights.shape}")
        return weights


def init_params(key, layers, load_file=False, best=False, scale=1e-2):
    # Initialize weights for each layer
    keys = jax.random.split(key, len(layers))

    if split_rank != 0:
        if load_file:
            for l in layers:
                if type(l) is not Neuron_states:
                    print(f"Error Not Implemented: the network contains at least one convolutional layer, loading pretrained is not implemented yet.")
                    sys.exit(1)
            
            print("Loading the weight file...")
            filename = f"tensor_data_{'_'.join(map(str, layer_sizes))}_batch{batch_size}.npz"
            if best:
                filename = "best_" + filename
            filepath = os.path.join("tensor_data", filename)
            w_data = np.load(filepath)
            for i, k in enumerate(w_data.files):
                if i == split_rank-1:
                    weights = jnp.array(w_data[k])
                    print(weights.shape)
                    return weights

        # Random initialization of the weights       
        layer = layers[split_rank]
        if type(layer) is Neuron_states:
            weights_shape = layer.weights_shape
        else:
            weights_shape = layer.neuron_state.weights_shape
        weights = scale * jax.random.normal(keys[split_rank], weights_shape)    
        # if rank == 1:
        #     kernel = jnp.array([[1, 0, -1],
        #                         [1, 0, -1],
        #                         [1, 0, -1]
        #                        ], dtype=jnp.float32)
        #     out_ch = 2
        #     in_ch = 1
        #     weights = jnp.stack([kernel for _ in range(out_ch * in_ch)])
        #     weights = weights.reshape(out_ch, in_ch, 3, 3) 
        #     print(weights)
        return weights
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


def update_new_values(values, active_indexes, new_input_activities):
    # update_row = jnp.where(new_input_activities)[0]     # Indices where new_input_activities is True
    # update_col = jnp.where(active_indexes == 1)[0]      # Indices where active_indexes == 1
    update_row = jnp.nonzero(new_input_activities, size=new_input_activities.shape[0], fill_value=-1)[0] # Shape (784,1)
    update_col = jnp.nonzero(active_indexes, size=active_indexes.shape[0], fill_value=-1)[0] # Shape (128, 1)
    
    # Generate all combinations (row, col) using broadcasting
    row_idx, col_idx = jnp.meshgrid(update_row, update_col, indexing="ij")  # shape (784, 128)
    row_idx = row_idx.flatten()
    col_idx = col_idx.flatten()
    
    # Create a mask to ignore any (-1, *) or (*, -1) pairs to avoid any dynamic conditions
    valid_mask = jnp.logical_and(row_idx != -1, col_idx != -1)
    
    row_idx = jnp.where(valid_mask, row_idx, -1)  # set to dummy valid index (-1, -1)
    col_idx = jnp.where(valid_mask, col_idx, -1)
    
    values = values.at[row_idx, col_idx].set(1)
    
    values = values.at[-1, -1].set(new_input_activities[-1, 0]) # Setting the (-1, -1) element to its correct value because over-writted by dummy index  
    return values

@partial(jax.jit, static_argnames=['params'])
def fc_layer_computation(neuron_idx, layer_input, weights, neuron_states, params, iteration=0):    
    c, x, y = neuron_idx
    # jax.debug.print("rank {} has neuron idx: {}", rank, neuron_idx)

    C, H, W = 0, 0, 0
    flat_layer_size = params.flat_layer_sizes[split_rank-1]
    if len(flat_layer_size) == 3:
        C, H, W = flat_layer_size
    neuron_idx = c * (H * W) + x * W + y 
    
    activations = jax.lax.cond(neuron_idx < 0,
                            lambda _: neuron_states.values,
                            lambda _: jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values,
                            None
                            )
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
                            lambda _: neuron_states.weight_residuals["input activity"],
                            lambda _: neuron_states.weight_residuals["input activity"].at[neuron_idx].add(1),
                            None
                            )

    def last_layer_case(_):
        return jnp.zeros((activations.shape[0], 4)), Neuron_states(
                                            values=activations, 
                                            thresholds=neuron_states.thresholds, 
                                            input_residuals=new_input_residuals, 
                                            weight_residuals={"input order": neuron_states.weight_residuals["input order"], 
                                                              "input activity": new_input_activity,
                                                              "layer activity": neuron_states.weight_residuals["layer activity"],
                                                              "output activity": neuron_states.weight_residuals["output activity"]},
                                            last_sent_iteration=neuron_states.last_sent_iteration,
                                            weights_shape=neuron_states.weights_shape
                                            )
    
    def hidden_layer_case(_):
        fire = (iteration-neuron_states.last_sent_iteration) >= params.sync_rate # Fire if sync rate reached
        async_fire = jnp.logical_or(params.async_layer <= 0, split_rank == params.async_layer) # Fire if async_layer or no async_layer condition (-1)
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

        layer_activity = neuron_states.weight_residuals["layer activity"]
        
        # APPLY THE RESTRICTION
        penalty = jax.lax.cond(params.restrict[split_rank] <= 0, 
                               lambda _: jnp.zeros(activated_output.shape), 
                               lambda _: activated_output*params.restrict[split_rank], None)
        
        # Store the neurons that activated
        active_indexes = jnp.where(activated_output > 0, 1, 0)
        new_layer_activities = layer_activity + active_indexes # Update the layer activity by adding the active neurons
        
        last_neuron_idx = jnp.argmax(neuron_states.weight_residuals["input order"]) # Last neuron index in the input order
        new_neuron_idx = jax.lax.cond(neuron_idx < 0,
                     lambda _: last_neuron_idx, 
                     lambda _: neuron_idx,
                     None)
        
        new_input_activities = neuron_states.weight_residuals["input order"].at[new_neuron_idx].set(iteration) # Update the input activity by setting the input neuron to the iteration number        
        
        # jax.debug.print("{} {}", active_indexes.shape, new_input_activities.shape)
        # new_values = update_new_values(neuron_states.weight_residuals["output activity"], active_indexes, new_input_activities) # Update input activity before updating the values
        new_output_activity = neuron_states.weight_residuals["output activity"].at[new_neuron_idx].add(active_indexes)
        
        new_weight_residuals = {"input order": new_input_activities, 
                                "input activity": new_input_activity,
                                "layer activity": new_layer_activities,
                                "output activity": new_output_activity}

        new_last_sent_iteration = jax.lax.cond(fire, lambda _: iteration, lambda _: neuron_states.last_sent_iteration, None)

        new_values = activations - activated_output - penalty
        # jax.debug.print("Rank {}, neuron idx: {}, new values: {}, active output: {}, penalty: {}, iteration: {}", rank, neuron_idx, new_values, active_output, penalty, iteration)
        new_neuron_states = Neuron_states(values=new_values, 
                                          thresholds=neuron_states.thresholds, 
                                          input_residuals=new_input_residuals, 
                                          weight_residuals=new_weight_residuals,
                                          last_sent_iteration=new_last_sent_iteration,
                                          weights_shape=neuron_states.weights_shape)
        
        indexes = jnp.arange(activated_output.shape[0])
        zeros = jnp.zeros_like(indexes)

        shaped_activated_output = jnp.stack([zeros, zeros, indexes, activated_output], axis=-1)
        # jax.debug.print("rank {}, shaped activated output shape: {}", rank, shaped_activated_output.shape)
        return shaped_activated_output, new_neuron_states
    
    cond = split_rank == last_rank #jnp.logical_or(split_rank == last_rank, neuron_idx < 0)
    return jax.lax.cond(cond, last_layer_case, hidden_layer_case, None)

@partial(jax.jit, static_argnames=['end_indices'])
def activated_output_to_send_array(activated_output, start_indices, end_indices, kernel_padding):
    '''
    Transforms the activated output matrix into a list with format (c, x, y, value)
    to send to the next layer.
    
    activated_output: (c, k_h, k_w) - the activated output corresponding to the input neuron
    start_indices: (c, x, y) - the starting indices of the slice in the padded neuron states
    end_indices: (c, h, w) - the shape of the original neuron states
    kernel_padding: (c, k_h_pad, k_w_pad) - the padding of the kernel
    '''
    c, h, w = activated_output.shape

    # Create coordinate grid
    c_grid, x_grid, y_grid = jnp.meshgrid(
        jnp.arange(c),
        jnp.arange(h),
        jnp.arange(w),
        indexing='ij'
    )

    # Flatten everything
    coords = jnp.stack([c_grid.ravel(), x_grid.ravel(), y_grid.ravel()], axis=-1)

    # Step 1: Adjust coordinates
    adjusted_coords = coords + jnp.array(start_indices)

    # Step 2: Create masks to filter out-of-bounds coordinates
    bottom_filter = jnp.array(kernel_padding) - jnp.array(start_indices)
    end_indices_arr = jnp.array(end_indices)

    # Boolean mask: is each adjusted coordinate inside valid range?
    is_above_bottom = adjusted_coords >= bottom_filter  # shape (N, 3)
    is_below_end = adjusted_coords < end_indices_arr        # shape (N, 3)

    # Only keep values that are within bounds in all (c, x, y)
    valid_mask = jnp.all(is_above_bottom & is_below_end, axis=-1)  # shape (N,)

    # Step 3: Apply mask to values
    values = activated_output.ravel()
    values_masked = values * valid_mask.astype(values.dtype)  # shape (N,)

    # Step 4: Concatenate adjusted coords and masked values
    out_events = jnp.concatenate([adjusted_coords, values_masked[:, None]], axis=-1)  # shape (N, 4)
    # jax.debug.print("rank {}, cgrid shape: {}, xgrid shape: {}, ygrid shape: {}, coord shape: {} start indices shape: {}, out_events shape: {}", rank, c_grid.shape, x_grid.shape, y_grid.shape, adjusted_coords.shape, start_indices, out_events.shape)
    # if params.sync_rate != 1:
    target_size = (end_indices[0] + 1) * (end_indices[1] + 1) * (end_indices[2] + 1)
    # Preallocate big array filled with -2
    padded = jnp.full((target_size, 4), 0, dtype=out_events.dtype)
    # Replace the first N rows with actual data
    padded = padded.at[:out_events.shape[0], :].set(out_events)
    out_events = padded
    
    return out_events
    

@partial(jax.jit, static_argnames=['params'])
def conv_layer_computation(neuron_idx, layer_input, weights, neuron_states, params, iteration=0):
    '''
    Apply the convolution for an incoming event in the event-driven manner described in "Optimizing event-based neural networks on digital neuromorphic architecture: a comprehensive design space exploration"
    This convolution only supports 'SAME' padding scheme with stride 1
    
    weights: (out_ch, in_ch, k_h, k_w)
    '''
    out_ch, in_ch, k_h, k_w = weights.shape
    c, x, y = neuron_idx
    def regular_input(neuron_states):
        # jax.debug.print("rank {} has x: {}, y: {}", rank, x, y)

        activations = jax.lax.cond(jnp.any(neuron_idx < 0), # Shape (out_ch, k_h, k_w) 
                                lambda _: jnp.zeros((out_ch, k_h, k_w)),
                                lambda _: jnp.dot(layer_input, jnp.flip(weights[:, c, :, :], axis=(1, 2))), 
                                None
                                )
        # Check whether the layer fires at this iteration
        fire = (iteration-neuron_states.last_sent_iteration) >= params.sync_rate # Fire if sync rate reached
        async_fire = jnp.logical_or(params.async_layer <= 0, split_rank == params.async_layer) # Fire if async_layer or no async_layer condition (-1)
        fire = jnp.logical_and(fire, async_fire) 
        fire = jnp.logical_or(fire, jnp.any(neuron_idx < 0)) # Fire if last input received    
        
        kernel_h_span, kernel_w_span = k_h//2, k_w//2 
        max_x, max_y = neuron_states.values.shape[1], neuron_states.values.shape[2] # c, h, w
        
        # Pad the values in neuron states to prevent indexing issues
        values_padded = jnp.pad(neuron_states.values, ((0, 0), (kernel_h_span, kernel_h_span), (kernel_w_span, kernel_w_span)))
        thresholds_padded = jnp.pad(neuron_states.thresholds, ((0, 0), (kernel_h_span, kernel_h_span), (kernel_w_span, kernel_w_span)))

        # Compute start indices for slicing and updating (x-kernel_h_span)   + kernel_h_span
        start_indices = (0, x, y)
        slice_shape = activations.shape  # (C, k_h, k_w)

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
        
        # APPLY THE RESTRICTION
        penalty = jax.lax.cond(params.restrict[split_rank] <= 0, 
                               lambda _: jnp.zeros(activated_output.shape), 
                                lambda _: activated_output*params.restrict[split_rank], None)
        
        # Step 4: Compute remaining values
        remaining_value = updated_slice - activated_output - penalty
        values_padded = jax.lax.dynamic_update_slice(values_padded, remaining_value, start_indices)
  
        # Step 5: Update the neuron state     
        new_last_sent_iteration = jax.lax.cond(fire, lambda _: iteration, lambda _: neuron_states.last_sent_iteration, None)
        neuron_states = neuron_states.replace(
                            neuron_state = neuron_states.neuron_state.replace(
                                values = neuron_states.values.at[:,:,:]
                                    .set(values_padded[:, kernel_h_span:max_x+kernel_h_span, kernel_w_span:max_y+kernel_w_span]),
                                    last_sent_iteration=new_last_sent_iteration,
                                    input_residuals = neuron_states.input_residuals.at[neuron_idx].add(layer_input)
                                    ))
        

        
        # jax.debug.print("rank {}, values padded: {}, current slice: {}, updated slice: {}, activated output: {}, remaining: {}, neuron state: {}", rank, values_padded, current_slice, updated_slice, activated_output, remaining_value, neuron_states.values)
        # jax.debug.print("\n____________________________________________________________________________________")
        # Step 5: Output the events
        padding = (0, kernel_h_span, kernel_w_span)
        out_events = activated_output_to_send_array(activated_output, start_indices, neuron_states.values.shape, padding)
    
        # jax.debug.print("rank {} has activated_output shape: {}, out_events shape: {}", rank, activated_output.shape, out_events.shape)
        # jax.debug.print("neuron states: {}, values padded: {}", neuron_states, values_padded)

        return out_events, neuron_states

    def last_input(neuron_states):
        neuron_val = neuron_states.values
        # For full sync case, fire all neurons that are above the threshold
        activated_output = activation_func(jnp.zeros(neuron_val.shape), neuron_val)  
        
        # Step 4: Compute remaining values and update the neuron state
        remaining_value = neuron_val - activated_output
        neuron_states = neuron_states.replace(
                            neuron_state = neuron_states.neuron_state.replace(
                                values = neuron_states.values.at[:,:,:]
                                    .set(remaining_value)))
        
        out_events = activated_output_to_send_array(activated_output, (0,0,0), neuron_states.values.shape, (0,0,0))
        return out_events, neuron_states
    return jax.lax.cond(jnp.any(neuron_idx < 0), last_input, regular_input, neuron_states)
    
@partial(jax.jit, static_argnames=['params'])
def predict(params, key, weights, empty_neuron_states, token, batch_data: jnp.ndarray):
    #region JAX loop
    def input_layer(args):
        token, neuron_states, x = args # x is shape (input_layer_size,)
        
        x_p = preprocess_to_sparse_data_padded(x, params.max_nonzero) # shape (max_nonzero, 2)
        if params.shuffle_input:
            perm = jax.random.permutation(key, x_p.shape[0])
            x_p = x_p[perm]
            
        def send_input(i, carry):
            token, count = carry
            data = x_p[i]
            def send_data(t):
                return send(data, dest=rank+process_per_layer, tag=0, comm=comm, token=t), count + 1

            def skip_data(t):
                return t, count
            
            token, count = jax.lax.cond(
                jnp.any(data != -2),
                send_data,
                skip_data,
                operand=token
            )
            return token, count

        # Initial carry: (token, iteration=0)
        token, iteration = jax.lax.fori_loop(0, x_p.shape[0], send_input, (token, 0))

        # Send end signal
        token = send(jnp.array([-1.0, 0.0]), dest=rank+process_per_layer, tag=0, comm=comm, token=token)

        return token, jnp.zeros(()), neuron_states, iteration
    
    def other_layers(args):
        token, neuron_states, _ = args
        def cond(state): # end of input has been reached -> break the while loop
            _, _, _, neuron_idx, _= state            
            return neuron_idx != -1
        
        def forward_pass(state):
            token, layer_input, neuron_states, neuron_idx, iteration = state
            
            def hidden_layers(input): # Send activation to the next layers
                token, activated_output = input

                def send_activation(i, token):
                    out_val = activated_output[i]
                    return jax.lax.cond(
                        out_val != 0,
                        lambda t: send(jnp.array([i, out_val]), dest=rank+process_per_layer, tag=0, comm=comm, token=t),
                        lambda t: t,
                        operand=token
                    )

                token = jax.lax.fori_loop(0, activated_output.shape[0], send_activation, token)
                return token
            
            # Receive neuron values from previous layers and compute the activations
            (neuron_idx, layer_input), token = recv(jnp.zeros((2,)), source=rank-process_per_layer, tag=0, comm=comm, token=token)
            activated_output, new_neuron_states= layer_computation(neuron_idx.astype(int), layer_input, weights, neuron_states, params, iteration)
            
            neuron_states = new_neuron_states
            
            token = jax.lax.cond(split_rank == last_rank, lambda input: input[0], hidden_layers, (token, activated_output)) # Don't send if we reach the last layer
            return token, layer_input, neuron_states, neuron_idx, iteration+1
        
        neuron_idx = 0
        layer_input = jnp.zeros(())
        initial_state = (token, layer_input, neuron_states, neuron_idx, 0)
        
        # Loop until the rank receives a -1 neuron_idx
        token, layer_input, neuron_states, neuron_idx, iteration = jax.lax.while_loop(cond, forward_pass, initial_state)
        
        # Send -1 to the next rank when all incoming data has been processed
        token = jax.lax.cond(
            split_rank != last_rank,
            lambda t: send(jnp.array([-1.0, 0.0]), dest=rank + process_per_layer, tag=0, comm=comm, token=t),
            lambda t: t,
            operand=token
        )
        return token, layer_input, neuron_states, iteration-1
    
    # Loop over batches, accumulate output values and return them
    def loop_over_batches(token, x):
        neuron_states = empty_neuron_states  
        token, layer_input, new_neuron_states, iterations = jax.lax.cond(split_rank==0, input_layer, other_layers, (token, neuron_states, x))
        
        return token, (new_neuron_states.values, iterations, new_neuron_states)
    
    token, (all_outputs, all_iterations, all_neuron_states) = jax.lax.scan(loop_over_batches, token, batch_data)    
    
    # Synchronize all ranks before starting the backward pass
    token = mpi4jax.barrier(comm=comm, token=token)

    return token, all_outputs, all_iterations, all_neuron_states

def group_send_data(params, data, neuron_states):
    '''
    For nmnist data has format (x, y, t, p), p is the polarity (0, 1) and we use it as the channel
    '''
    next_layer = params.layer_sizes[split_rank+1]
    x, y, _, c = data
    
    def next_conv(_):
        next_kernel = next_layer[2]
        k_x, k_y = next_kernel
        C, H, W = params.flat_layer_sizes
        
        position_array = jnp.array([c, x, y])
        data_around_position = neuron_states.values[c, x-k_x+1:x+k_x-1, y-k_y+1:y+k_y-1]
        flat_data_around_position = data_around_position.reshape(-1)
        
        pad_width = params.max_kernel - ((k_x+k_x-1) * (k_y+k_y-1))
        padded_data = jnp.pad(flat_data_around_position, (0, pad_width), constant_values=0)
        
        combined = jnp.concatenate([position_array, padded_data], axis=0)
        return combined
        
    def next_fc(_):
        '''
        if the current layer is a fc layer then c and x will contain the value -1 while y contains the true index
        '''
        C, H, W = 0, 0, 0
        flat_layer_size = params.flat_layer_sizes[split_rank]
        if len(flat_layer_size) == 3:
            C, H, W = flat_layer_size
        
        index = c * (H * W) + x * W + y 
        
        position_array = jnp.array([-1, -1, index])
        padded_data = jnp.zeros(params.max_kernel)
        
        combined = jnp.concatenate([position_array, padded_data], axis=0)
        return combined
        
    message = jax.lax.cond(
        len(next_layer) == 1,
        next_fc,
        next_conv,
        None
    )
    
    return message
    

@partial(jax.jit, static_argnames=['params', 'layer_computation',])
def conv_predict(params, key, weights, empty_neuron_states, token, batch_data: jnp.ndarray, layer_computation):
    '''
    CNN inference, each layer sends each event separately in the format: (c, x, y, value)
    -1 means end of data from previous layer
    -2 means placeholder data in the input layer 
    '''
    # jax.debug.print("Rank {} has batch_data shape: {}", rank, batch_data.shape)

    #region JAX loop
    def input_layer(args):
        token, neuron_states, x = args  # x binned is shape (timesteps, channel, height, width)
                                        # x not binned is shape (max_nonzero, 4)
        # jax.debug.print("Rank {}, input layer shape: {}", rank, x.shape)
        
        x_p = x
        def send_input(i, carry):
            token, timestep = carry
            data = x_p[i]
            def send_data(t):
                # Pack the timestep and the matrix in a 1D array
                # timestep_array = jnp.array([timestep], dtype=jnp.int32)  # shape (1,)
                # flat_data = jnp.reshape(data, (-1,))               # shape (2*34*34,)
                # combined = jnp.concatenate([timestep_array, flat_data], axis=0)
                
                # combined = jnp.array([data[3], data[0], data[1], jnp.array(1.0)])
                # jax.debug.print("rank {}, each data shape: {}{}{}{}{}", rank, data[0].shape, data[1].shape, data[2].shape, data[3].shape, jnp.array(1.0).shape)
                
                # combined = jnp.stack([data[3], data[0], data[1], jnp.array(1.0, dtype=jnp.float32)])
                combined = jnp.stack([data[0], data[1], data[2], data[2]])
                combined = data
                # combined = group_send_data(params, data, neuron_states)
                
                token = send(combined, dest=rank+process_per_layer, tag=0, comm=comm, token=t)
                return token, timestep+1

            def skip_data(t):
                return t, timestep
            
            token, timestep = jax.lax.cond(
                jnp.any(data != -2),
                send_data,
                skip_data,
                operand=token
            )
            return token, timestep

        # Initial carry: (token, timestep=0)
        token, iteration = jax.lax.fori_loop(0, x_p.shape[0], send_input, (token, 0))

        # Send end signal
        # end_signal = jnp.concatenate([jnp.array([-1, -1, -1, -1], dtype=jnp.int32), jnp.zeros(params.max_kernel)], axis=0)
        token = send(jnp.array([-1.0, -1.0, -1.0, -1.0]), dest=rank+process_per_layer, tag=0, comm=comm, token=token)
        # jax.debug.print("Rank {}, sent end signal", rank)

        return token, neuron_states, iteration
    
    def other_layers(args):
        token, neuron_states, _ = args
        
        # input_shape = params.flat_layer_sizes[split_rank-1]
        # input_shape_flat = math.prod(input_shape) + 1
        
        # jax.debug.print("rank {} input shape: {}, input shape flat: {}", rank, (input_shape).shape, (input_shape_flat).shape)
        # return token, jnp.zeros(()), neuron_states, 1
        def cond(state): # end of input has been reached -> break the while loop
            _, _, neuron_idx, _, _= state
            return jnp.all(neuron_idx != -1)
        
        def forward_pass(state):
            token, neuron_states, _, timestep, iteration = state
            
            def hidden_layers(input): # Send activation to the next layers
                token, activated_output = input

                def send_activation(i, token):
                    # Pack the timestep and the matrix in a 1D array
                    # timestep_array = jnp.array([0], dtype=jnp.int32)  # shape (1,)
                    # flat_data = jnp.reshape(activated_output, (-1,))  # shape 
                    # combined = jnp.concatenate([timestep_array, flat_data], axis=0)
                    combined = activated_output[i]
                    
                    # jax.debug.print("rank {} sending {}", rank, combined)
                    return jax.lax.cond(
                        (combined[3] != 0),
                        lambda t: send(combined, dest=rank+process_per_layer, tag=0, comm=comm, token=t),
                        lambda t: t,
                        operand=token
                    )

                    # out_val = activated_output[i]
                    # return jax.lax.cond(
                    #     out_val != 0,
                    #     lambda t: send(jnp.array([i, out_val]), dest=rank+process_per_layer, tag=0, comm=comm, token=t),
                    #     lambda t: t,
                    #     operand=token
                    # )

                token = jax.lax.fori_loop(0, activated_output.shape[0], send_activation, token)
                return token
            
            # Receive neuron values from previous layers and compute the activations
            # input_shape = params.max_kernel + 3
            input_shape = 4 
            input_data, token = recv(jnp.zeros(input_shape), source=rank-process_per_layer, tag=0, comm=comm, token=token)
            # Unpack
            neuron_idx = input_data[:3] # channel, x, y
            layer_input = input_data[3] # value
            
            # jax.debug.print("rank {} at iteration {}", rank, iteration)
            
            # jax.debug.print("rank {} input data: {}",rank , (input_data))
            # jax.debug.print("rank {} stop condition not met: {}",rank , jnp.all(neuron_idx != -1))
            # data_position = input_data[:3].astype(int)
            # k_x, k_y = empty_neuron_states.kernel
            # data_shape = (k_x+k_x-1) * (k_y+k_y-1)
            # layer_input = jnp.reshape(input_data[3:params.max_kernel], data_shape)
            
            activated_output, new_neuron_states = layer_computation(neuron_idx.astype(int), layer_input, weights, neuron_states, params, iteration)
            # jax.debug.print("rank {} activated output shape: {}",rank , neuron_states.values)
            
            # activated_output, new_neuron_states= layer_computation(timestep.astype(int), layer_input, weights, neuron_states, params, iteration)
            # activated_output, new_neuron_states = jnp.full((1, 4), 1.0), neuron_states
            
            neuron_states = new_neuron_states
            
            token = jax.lax.cond(split_rank == last_rank, lambda args: args[0], hidden_layers, (token, activated_output)) # Don't send if we reach the last layer
            return token, neuron_states, neuron_idx, timestep, iteration+1
        
        neuron_idx, timestep, iteration =  jnp.zeros(3), 0, 0
        initial_state = (token, neuron_states, neuron_idx, timestep, iteration)
        
        # Loop until the rank receives a -1 timestep
        token, neuron_states, neuron_idx, timestep, iteration = jax.lax.while_loop(cond, forward_pass, initial_state)
        
        # Send -1 to the next rank when all incoming data has been processed
        
        # end_signal = jnp.concatenate([jnp.array([-1, -1, -1, -1], dtype=jnp.int32), jnp.zeros(empty_neuron_states.values.shape).reshape(-1)], axis=0)
        # jax.debug.print("Rank {}, sent end signal", rank)

        token = jax.lax.cond(
            split_rank != last_rank,
            lambda t: send(jnp.array([-1.0, -1.0, -1.0, -1.0]), dest=rank + process_per_layer, tag=0, comm=comm, token=t),
            lambda t: t,
            operand=token
        )
        return token, neuron_states, iteration-1
    
    # Loop over batches, accumulate output values and return them
    def loop_over_batches(token, x):
        neuron_states = empty_neuron_states  
        token, new_neuron_states, iterations = jax.lax.cond(split_rank==0, input_layer, other_layers, (token, neuron_states, x))
        
        return token, (new_neuron_states.values, iterations, new_neuron_states)
    token, (all_outputs, all_iterations, all_neuron_states) = jax.lax.scan(loop_over_batches, token, batch_data)    
    
    # Synchronize all ranks before starting the backward pass
    token = mpi4jax.barrier(comm=comm, token=token)
    return token, all_outputs, all_iterations, all_neuron_states

def preprocess_to_sparse_data_padded(x, max_nonzero):
    # Pre-allocate max possible
    processed_data = jnp.full((max_nonzero, 2), -2.0)
    def neuron_state_preprocess(processed_data):        
        def body_fn(i, carry):
            processed_data, j = carry
            val = x[i]
            processed_data, j = jax.lax.cond(
                val != 0,
                lambda _: (processed_data.at[j].set(jnp.array([i, val])), j + 1),
                lambda _: (processed_data, j),
                operand=None
            )
            return processed_data, j

        init_val = (processed_data, 0)
        processed_data, _ = jax.lax.fori_loop(0, x.shape[0], body_fn, init_val)
        return processed_data
    
    if x[0].shape == ():
        return neuron_state_preprocess(processed_data)
    else:
        return processed_data


#region Training helpers
def z_gradient(weight_res, next_grad):
    '''
    vmap computation with weight_res shape: (784, 128) and next_grad shape: (128,)
    '''
    next_grad_expanded = jnp.expand_dims(next_grad, axis=0)  # Shape: (1, 128)

    # Perform element-wise multiplication
    z_grad = weight_res * next_grad_expanded # shape: (784, 128)
    return z_grad

@jit
def compute_w_residuals(input_activity, output_activity):
    '''
    Compute the weights that activated and need to be updated by taking into account previous timesteps influence
    input_activity: shape (784, ) containing last iteration number or -1 if never fired
    output_activity:   shape (784, 128)
    '''
    # Preprocess the input activity by computing the ordering of the indices
    activity_ordered = jnp.argsort(input_activity)
    
    def body(i, carry):
        activates, output_activity = carry
        
        # Use the ordered activity
        j = activity_ordered[i]
        # jax.debug.print("i: {}, j: {}, input_activity[j]: {}", i, j, input_activity[j])
        def update_if_active_fn(carry):
            activates, output_activity = carry

            # Extract row i
            vals = output_activity[j]  # shape: (128,)
            
            # Case 1: neuron_val == 0 and activates[j] == 1 → set output to 1, meaning this neuron activated in later timesteps
            condition = (vals == 0) & (activates == 1)
            replace_vals = jnp.where(vals > 0, 1, 0) # Only have values 0 or 1 in the residuals
            new_vals = jnp.where(condition, 1, replace_vals)
            output_activity = output_activity.at[j].set(new_vals)

            # Case 2: neuron_val == 1 and activates[j] == 0 → set activates[j] = 1, meaning first activation of this neuron
            update_activates = (vals == 1) & (activates == 0)
            new_activates = jnp.where(update_activates, 1, activates)
            activates = new_activates
            
            return activates, output_activity

        # jax.debug.print("j: {}, j type: {}, input_activity[j]: {}", j, type(j), input_activity[j])
        return jax.lax.cond(
            input_activity[j]>-1,
            update_if_active_fn,
            lambda carry: carry,
            operand=(activates, output_activity)
        )

    # Initial state
    activates = jnp.zeros((output_activity.shape[1],), dtype=jnp.int32) #(128,)

    # Reverse loop with fori_loop
    n = input_activity.shape[0] # 784
    activates, output_activity = jax.lax.fori_loop(
        0, jnp.sum(input_activity!=-1), # Don't loop over the non relevant values
        lambda idx, carry: body(n - 1 - idx, carry),  # reversed order
        (activates, output_activity)
    )
    return output_activity

def recompute_w_residuals(current_res, next_res):
    """
    Recompute the weight residuals of the current layer by taking into account the weight residuals of the next layer.
    Basically if one row (neuron) in the next layer is all zeros (=neuron never activated), then the corresponding column in the current layer should be set to zero. 
    
    current_res: (128, 64) — one batch element
    next_res: (64, 10) 
    """

    mask = jnp.all(next_res == 0, axis=1)  # shape (64,)
    # jax.debug.print("mask type: {}", mask.type)
    numeric_mask = (~mask).astype(current_res.dtype)  # invert to keep columns where mask is False
    # if rank == 1:
    #     jax.debug.print("numeric_mask shape: {}, mask: {}", numeric_mask.shape, jnp.sum(numeric_mask))
    
    # Broadcast to shape (128, 64)
    full_mask_a = jnp.expand_dims(numeric_mask, axis=0)  # (1, 64)
    full_mask = jnp.broadcast_to(full_mask_a, current_res.shape)  # (128, 64)
    # jax.debug.print("rank {}: {}, {}", rank, full_mask_a, full_mask)

    out = (current_res * full_mask).astype(current_res.dtype)
    are_equal = jnp.all(out == current_res)
    
    # jax.lax.cond(
    #     are_equal,
    #     lambda flag: jax.debug.print("Rank {}, flag: {}, next_res: {}", rank, flag, jnp.max(next_res)),
    #     lambda flag: jax.debug.print("Rank {}, out and cur_res are diff: {}", rank, flag),
    #     operand=are_equal
    # )

    return out

def apply_restrict_to_residuals(params, weight_residuals, layer_activity):
    '''
    Applying to the 1 values in the residuals: 1+(1-alpha)^(n*(n+1)/2)
    '''
    exponent = (layer_activity*(layer_activity+1)/2).astype(jnp.int32)
    new_layer_activity = jnp.where(params.restrict[split_rank] > 0, 1+jnp.power((1-params.restrict[split_rank]), exponent), 1)
    mul_res = jnp.broadcast_to(new_layer_activity, weight_residuals.shape)
    out = weight_residuals * mul_res

    return out

@partial(jax.jit, static_argnames=['params', 'layer_computation',])
def predict_bwd(params, key, batch_data, layer_computation, weights, empty_neuron_states, token):
    '''
    B: batch_size
    '''
    token, all_outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, token, batch_data, layer_computation)
    
    next_layer_shape = params.flat_layer_sizes[split_rank]
    next_grad, token = recv(jnp.zeros((batch_part,) + next_layer_shape), source=rank + process_per_layer, tag=2, comm=comm) # Shape: (B, 128)
    # jax.debug.print("Rank {} received next_grad shape: {}", rank, next_grad)
    
    # "input order": Shape (B, 784, 1), "output activity": Shape (B, 784, 128)
    weight_res = jax.vmap(compute_w_residuals, in_axes=(0, 0))(all_neuron_states.weight_residuals["input order"], all_neuron_states.weight_residuals["output activity"]) # Shape: (B, 784, 128)
    # weight_res = weight_res["output activity"] # incorrect residual but faster for testing
    jax.debug.print("Rank {} weight_res shape: {}, weight_res max: {}", rank, weight_res.shape, jnp.max(weight_res))

    
    next_weight_res = jnp.ones((batch_part,) + weights.shape) # Shape: (B, 128, 10)
    # jax.debug.print("Rank {} received next_grad shape: {}, next_weight_res shape: {}", rank, next_grad.shape, next_weight_res.shape)
    (next_weight_res, token) = jax.lax.cond(split_rank < last_rank - 1, 
                                lambda _: recv(next_weight_res, source=rank + process_per_layer, tag=3, comm=comm),
                                lambda _: (next_weight_res, token), None) 
    # jax.debug.print("Rank {} received next_grad shape: {}", rank, next_weight_res)

    # weight_res = jax.lax.cond(split_rank < last_rank - 1,
    #                             lambda args: jax.vmap(recompute_w_residuals, in_axes=(0, 0))(args[0], args[1]), # Shape: (B, 784, 128)
    #                             lambda _: weight_res,
    #                             (weight_res, next_weight_res))    
    
    weight_res = jax.vmap(apply_restrict_to_residuals, in_axes=(None, 0, 0))(params, weight_res, all_neuron_states.weight_residuals["layer activity"])
    
    # Perform element-wise multiplication
    z_grad = jax.vmap(z_gradient, in_axes=(0, 0))(weight_res, next_grad) # Shape: (B, 784, 128)
    
    x = all_neuron_states.input_residuals # Shape (B, 784)
    x_reshaped = x[..., jnp.newaxis]   # Shape becomes (B, 784, 1)
    
    weight_grad = x_reshaped * z_grad # (B, 784, 128)
    
    # jax.debug.print("weight_grad: {}, x: {}, z_grad: {}, next_grad_expanded: {}, weight_res: {}", jnp.isnan(weight_grad).any(), jnp.isnan(x).any(), jnp.isnan(z_grad).any(), jnp.isnan(next_grad_expanded).any(), jnp.isnan(weight_res).any())
    mean_weight_grad = jnp.mean(weight_grad, axis=0) # (784, 128)

    # jax.debug.print("x {}, x_reshaped{}", x.shape, x)
    # jax.debug.print("next_grad_expanded {}, {}", next_grad_expanded.shape, next_grad_expanded)
    # jax.debug.print("weight residuals {}, {}", weight_res.shape, weight_res)
    # jax.debug.print("z_grad {}, {}", z_grad.shape, z_grad)
    # jax.debug.print("weight_grad {}, mean_weight_grad{}", weight_grad.shape, mean_weight_grad.shape)

    layer_activity = jnp.where(all_neuron_states.weight_residuals["layer activity"] > 0, 1, 0)
    th_grad = -jnp.mean(next_grad * layer_activity, axis=0)  # Shape: (128)
    # jax.debug.print("Rank {}, th_grad shape: {}, th_grad: {}", rank, th_grad.shape, (th_grad)) # Shape: (128,)

    if split_rank > 1:
        send_grad = jnp.dot(next_grad, weights.T) # Shape: (B, 128) @ (128, 784) = (B, 784)

        token = send(send_grad, dest=rank-process_per_layer, tag=2,comm=comm, token=token)
        token = send(weight_res, dest=rank-process_per_layer, tag=3,comm=comm, token=token)
    
    # Sparsity loss gradients 
    token, all_activations, all_iterations, sparsity_L = sparsity_loss(params, token, all_neuron_states, iterations)
    
    scaling = jax.lax.cond(params.threshold_impact > 0,
                           lambda _: params.threshold_impact / (all_iterations * batch_part * process_per_layer) ,
                           lambda _: 0.0,
                           None)
    
    input_activity = jnp.sum(all_neuron_states.weight_residuals["input activity"], axis=0) # Shape (784)
    layer_activity = jnp.sum(all_neuron_states.weight_residuals["layer activity"], axis=0) # Shape (128)
    
    token, layer_activity = gather_batch(token, layer_activity, average=False) # Gather the weight gradients from all ranks in the split rank
    token, input_activity = gather_batch(token, input_activity, average=False)
    
    sparsity_residuals = scaling * layer_activity # Shape: (128,)
    # jax.debug.print("Rank {}, scaling mean: {}, sparsity_residuals mean: {}, sparsity_residuals sum: {}", rank, scaling, jnp.mean(sparsity_residuals), jnp.sum(sparsity_residuals))
    
    th_sparsity_grad = -sparsity_residuals
    weight_sparsity_grad = jnp.outer(input_activity, sparsity_residuals) # Shape: (784, 128)
    # jax.debug.print("Rank {}, th_sparsity_grad: {}, weight_sparsity_grad: {}", rank, jnp.mean(th_sparsity_grad), np.mean(weight_sparsity_grad))
    
    return token, all_outputs, iterations, all_neuron_states, (weight_grad, th_grad, weight_sparsity_grad, th_sparsity_grad, weight_res) 

@partial(jax.jit, static_argnames=['params', 'layer_computation',])
def conv_predict_bwd(params, key, batch_data, layer_computation, weights, empty_neuron_states, token):
    token, all_outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, token, batch_data, layer_computation)
    
    layer_shape = params.flat_layer_sizes[split_rank]
    next_grad, token = recv(jnp.zeros((batch_part,) + layer_shape), source=rank + process_per_layer, tag=2, comm=comm) # Shape: (B, 128)
    
    weight_res = jnp.zeros((batch_part,) + weights.shape)
    weight_grad = jnp.zeros((batch_part,) + weights.shape)
    th_grad = jnp.zeros(all_neuron_states.values.shape)
    
    weight_sparsity_grad = jnp.zeros(weights.shape)
    th_sparsity_grad = jnp.zeros(layer_shape)
    
    return token, all_outputs, iterations, all_neuron_states, (weight_grad, th_grad, weight_sparsity_grad, th_sparsity_grad, weight_res) 

# Define the loss function
def softmax_cross_entropy_with_logits(logits, labels):
    # Compute the softmax in a numerically stable way
    logits_max = jnp.max(logits, axis=0, keepdims=True)
    exps = jnp.exp(logits - logits_max)
    softmax = exps / (jnp.sum(exps, axis=0, keepdims=True) + 1e-8)
    # Compute the cross-entropy loss
    cross_entropy = -jnp.sum(labels * jnp.log(softmax + 1e-8), axis=0)
    # jax.debug.print("logits {}, max: {}, cross entropy: {}", logits, logits_max, cross_entropy)
    return cross_entropy

def mean_loss(logits, labels):
    batched_softmax_cross_entropy = jax.vmap(softmax_cross_entropy_with_logits, in_axes=(0, 0))
    losses = batched_softmax_cross_entropy(logits, labels)
    return jnp.mean(losses)

def output_gradient(weights, loss_grad):
    return jnp.dot(weights, loss_grad)

def output_weight_grad(loss_grad, all_residuals):
    '''
    vmap computation with loss_grad shape: (10,) and all_residuals shape: (128,)
    '''
    # Expand dimensions of loss_grad 
    loss_grad_expanded = jnp.expand_dims(loss_grad, axis=1)  # Shape: (10, 1)

    # Broadcast and perform element-wise multiplication
    weight_grad = loss_grad_expanded * all_residuals  # Shape: (10, 128)
    return weight_grad.T

@partial(jax.jit, static_argnames=['params', 'layer_computation',])
def loss_fn(params, key, batch_data, weights, empty_neuron_states, token, layer_computation, target):
    token, all_outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, token, batch_data, layer_computation)
    # jax.debug.print("output shape: {}, mean output: {}, target shape: {}", all_outputs.shape, jnp.mean(all_outputs), target.shape)
    # jax.debug.print("Rank {}, nb_input: {}, nb_iteration: {}", rank, 
    #                         jax.vmap(lambda n_s: jnp.sum(jnp.where(n_s != 0, 1, 0)))(all_neuron_states.input_residuals), 
    #                         jax.vmap(lambda n_s: n_s)(iterations))#(all_neuron_states.weight_residuals["input activity"])) #jnp.sum(jnp.where(n_s !=0, 1, 0))
    # loss = jnp.mean((all_outputs - target) ** 2)
    # N = all_outputs.shape[0]  
    # loss_grad = (2 / N) * (all_outputs - target)
    all_residuals = all_neuron_states.input_residuals # Shape: (B, 128)
    # jax.debug.print("weight shape: {} {}", all_neuron_states.thresholds[0], all_neuron_states.thresholds)
    # jax.debug.print("threshold loss shape: {} {}", threshold_loss, threshold_grad)
    
    # jax.debug.print("regularized average iterations: {},  {}/{}", reg_avg_iterations, jnp.mean(iterations), jnp.max(iterations))
    loss, loss_grad = jax.value_and_grad(mean_loss)(all_outputs, target)
    loss_grad /= process_per_layer # Shape (B, 10)
    # total_loss = loss + threshold_loss[0]
    # jax.debug.print("rank {}, loss: {}", rank, loss)
    
    out_grad = jax.vmap(output_gradient, in_axes=(None, 0))(weights, loss_grad) # Shape (B, 128)
    # jax.debug.print("rank {}, loss: {}, loss_grad shape: {}, out_grad shape: {}", rank, loss, loss_grad.shape, out_grad.shape)
    
    weight_grad =  jax.vmap(output_weight_grad, in_axes=(0, 0))(loss_grad, all_residuals) #Shape (B, 128, 10)
    mean_weight_grad = jnp.mean(weight_grad, axis=0) # Shape: (128, 10)
    # jax.debug.print("Rank {}, all_residuals shape: {}, threshold shape: {}, weight grad shape: {}", rank, all_residuals.shape, all_neuron_states.thresholds.shape, weight_grad.shape)
    
    # jax.debug.print("loss: {}, loss gradient: {}", loss, loss_grad.shape)
    # jax.debug.print("out grad {}, {}", out_grad.shape, out_grad)
    # jax.debug.print("all residuals {}, {}", all_residuals.shape, all_residuals.dtype)
    # jax.debug.print("weight_grad {}, mean_weight_grad{}", weight_grad.shape, mean_weight_grad.shape)
    
    # Send gradient to previous layers                
    token = send(out_grad, dest=rank-process_per_layer, tag=2,comm=comm, token=token)
    
    token, all_activations, all_iterations, sparsity_L = sparsity_loss(params, token, all_neuron_states, iterations)

    total_loss = loss + params.threshold_impact * sparsity_L 

    return (loss, all_outputs, iterations, total_loss), (out_grad, weight_grad, loss_grad, weight_grad)

def sparsity_loss(params, token, all_neuron_states, iterations):
    '''
    Compute the sparsity loss based on the input residuals and the weight residuals
    '''
    if params.threshold_impact <= 0.0:
        return token, 0, 1, 0
    
    # Gather all the activations at the last layer to compute the sparsity loss
    leader_rank = split_rank * process_per_layer
    # jax.debug.print("Rank {}, activations shape: {}, iterations shape: {}, last layer neuron values: {}", rank, (activations.shape), jnp.mean(iterations.shape), jnp.sum(all_neuron_states.values))

    token, activations = gather_batch(token, all_neuron_states.input_residuals, average=False) # Gather the weight gradients from all ranks in the split rank
    token, iterations = gather_batch(token, iterations, average=True) # Gather the iterations from all ranks in the split rank
    # jax.debug.print("Rank {}, activations shape: {}, iterations shape: {}, last layer neuron values: {}", rank, (activations.shape), jnp.mean(iterations.shape), jnp.sum(all_neuron_states.values))

    all_iterations = 0.0
    all_activations = 0.0
    sparsity_L = 0.0
    if split_rank != last_rank and rank == leader_rank:
        # jax.debug.print("Rank {}, sending activations {} and iterations {} to the last rank", rank, jnp.sum(activations), jnp.mean(iterations))
        token = send(jnp.sum(activations), dest=last_rank * process_per_layer, tag=6,comm=comm, token=token)
        if rank == 0:
            token = send(jnp.mean(iterations), dest=last_rank * process_per_layer, tag=6,comm=comm, token=token)
    elif split_rank == last_rank and rank == leader_rank:
        for i in range(last_rank):
            # Storing the thresholds
            act_sum, token = recv(jnp.zeros(1), source=i * process_per_layer, tag=6, comm=comm, token=token)
            all_activations = all_activations + act_sum[0] # Sum of all activations in the hidden layers
            
            if i == 0: # Get iterations of input data
                iter_mean, token = recv(jnp.zeros(1), source=i * process_per_layer, tag=6, comm=comm, token=token)
                all_iterations = iter_mean[0]
        all_activations += jnp.sum(activations) # Adding the activations of the last layer

        sparsity_L = all_activations /  (all_iterations * batch_part * process_per_layer)
        # jax.debug.print("Rank {}, sparsity L: {}, all iterations: {}, all activations: {}", rank, sparsity_L, all_iterations, all_activations)
    all_iterations, token = bcast(all_iterations, root=last_rank*process_per_layer, comm=comm, token=token)

    return token, all_activations, all_iterations, sparsity_L

def share_split_rank_data(token, data):
    data = jnp.array(data)
    leader_rank = split_rank * process_per_layer
    if rank == leader_rank:
        for i in range(process_per_layer-1): # Sharing the data to all the corresponding ranks
            token = send(data, dest=rank+i+1, tag=20, comm=comm, token=token)
    else:
        data, token = recv(data, source=leader_rank, tag=20, comm=comm, token=token)        
    return token, data

def split_batch(token, batch_iterator):
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
                
                token = send(batch_x_to_send, dest=process, tag=4, comm=comm, token=token)
                token = send(batch_y_to_send, dest=process, tag=4, comm=comm, token=token)
    else:
        batch_x, token = recv(jnp.zeros((batch_part, layer_sizes[0])), source=0, tag=4, comm=comm, token=token)  
        batch_y, token = recv(jnp.zeros((batch_part,)), source=0, tag=4, comm=comm, token=token) 
    
    return token, batch_x, batch_y

def gather_batch(token, data, average=True):
    '''
    Gather all the data from one split_rank onto one rank and resharing the average result to the corresonding split_ranks
    '''
    data = jnp.array(data)
    leader_rank = split_rank * process_per_layer
    if rank == leader_rank:
        avg = data
        for i in range(process_per_layer-1): # Receive the data from all the corresponding ranks in one split rank
            received_data, token = recv(data, source=rank+i+1, tag=20, comm=comm, token=token)
            avg += received_data
        if average:
            avg = avg / process_per_layer
        
        for i in range(process_per_layer-1): # Resharing the average data to all the corresponding ranks
            token = send(avg, dest=rank+i+1, tag=20, comm=comm, token=token)
    else:
        token = send(data, dest=leader_rank, tag=20, comm=comm, token=token)
        avg, token = recv(data, source=leader_rank, tag=20, comm=comm, token=token)
    return token, avg

def combine_batch(token, data, average=False):
    '''
    Concatenate all the data from one split_rank onto one rank to reconstruct the batch and resharing the combined result to the corresonding split_ranks
    '''
    data = jnp.array(data)
    # if len(data.shape) < 2:
        
            
    leader_rank = split_rank * process_per_layer
    if rank == leader_rank:
        avg = data
        for i in range(0, process_per_layer-1): # Receive the data from all the corresponding ranks in one split rank
            received_data, token = recv(data, source=rank+i+1, tag=20, comm=comm, token=token)
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
            token = send(avg, dest=rank+i+1, tag=20, comm=comm, token=token)
    else:
        token = send(data, dest=leader_rank, tag=20, comm=comm, token=token)
        avg, token = recv(jnp.zeros((data.shape[1], data.shape[2])), source=leader_rank, tag=20, comm=comm, token=token)
        
    return token, avg


# region TRAINING
def train(token, params: Params, key, network_layers, weights, empty_neuron_states, layer_computation, opti):     
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
    
    token = mpi4jax.barrier(comm=comm, token=token)
    start_time = time.time()

    for epoch in range(params.num_epochs):
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
            
        for i in range(total_train_batches):
            neuron_states = empty_neuron_states
            # threshold_grad = 0.0
            if split_rank == 0:
                # print(i)
                token, batch_x, batch_y = split_batch(token, batch_iterator)
                # print(batch_y.shape, type(batch_y), batch_y)
                token = send(batch_y, dest=last_rank * process_per_layer + rank, tag=10,comm=comm, token=token) # Destination rank: last_rank * process_per_layer + rank

                token, outputs, iterations, all_neuron_states = (conv_predict)(params, subkey, weights, neuron_states, token, jnp.array(batch_x), layer_computation)
                token, all_activations, all_iterations, sparsity_L = sparsity_loss(params, token, all_neuron_states, iterations)
            else:
                if split_rank==last_rank:
                    # Receive y
                    y, token = recv(jnp.zeros((batch_part,)), source=rank - (last_rank * process_per_layer), tag=10, comm=comm, token=token)  # Source rank opposite operation: rank - (last_rank * process_per_layer)
                    y_encoded = jnp.array(one_hot_encode(y, num_classes=layer_sizes[-1][0]))
                    # print("encoded y: ", y, y_encoded.shape, y_encoded)              
                    (loss, outputs, iterations, total_loss), gradients = (loss_fn)(params, subkey, jnp.zeros((batch_part, 1, 4)), weights, neuron_states, token, layer_computation, y_encoded)

                    epoch_loss.append(loss)
                    
                    weight_grad = gradients[1]
                                        
                    valid_y, batch_correct = accuracy(i, outputs, y, iterations, False)                 
                        
                    epoch_correct += batch_correct
                    epoch_total += valid_y.shape[0]
                    # token, weight_grad = gather_batch(token, weight_grad, average=True)
                    token, weight_grad = combine_batch(token, weight_grad, average=True) # Gather the weight gradients from all ranks in the split rank
                else:
                    bwd_fn = predict_bwd
                    if empty_neuron_states.is_conv:
                        bwd_fn = conv_predict_bwd
                    
                    token, outputs, iterations, all_neuron_states, grads = (bwd_fn)(params, subkey, jnp.zeros((batch_part, 1, 4)), layer_computation, weights, neuron_states, token)
                    weight_grad, threshold_grad, weight_sparsity_grad, threshold_sparsity_grad, weight_res = grads
                    # print(f"rank {rank}, weight_res: {weight_res[0].tolist()}, shape: {weight_res.shape}")

                    # print(f"Rank {rank} finished predict_bwd for batch {i}, outputs shape: {outputs.shape}, iterations: {iterations.shape}, weight_grad shape: {weight_grad.shape}")
                    token, threshold_grad = gather_batch(token, threshold_grad, average=True) # Gather the weight gradients from all ranks in the split rank

                    # token, weight_grad = gather_batch(token, weight_grad, average=True)
                    token, weight_grad = combine_batch(token, weight_grad, average=True) # Gather the weight gradients from all ranks in the split rank
                    
                    if params.threshold_impact > 0:
                        weight_grad = weight_grad + weight_sparsity_grad
                        threshold_grad = threshold_grad + threshold_sparsity_grad
                    # Update thresholds
                    # print(f"new thresholds: {jnp.mean(empty_neuron_states.thresholds)}, threshold_grad: {jnp.mean(threshold_grad)}")
                    # empty_neuron_states.thresholds = jax.nn.sigmoid(empty_neuron_states.thresholds - (threshold_grad * params.threshold_lr))
                    
                    if params.threshold_lr != 0:
                        # print(f"average threshold grad: {jnp.mean(threshold_grad)}")
                        th_updates, th_opt_state = solver.update(threshold_grad, th_opt_state, empty_neuron_states.thresholds)
                        empty_neuron_states.thresholds = jax.nn.sigmoid(optax.apply_updates(empty_neuron_states.thresholds, th_updates))
                    # print(empty_neuron_states.thresholds)
                
                # print("Rank {}, batch {}, mean weight_grad: {}, max weight_grad: {}, min weight_grad: {}".format(rank, i, jnp.mean(weight_grad), jnp.max(weight_grad), jnp.min(weight_grad)))
                # Update weights
                if solver is not None:
                    # Optax optimizer
                    updates, opt_state = solver.update(weight_grad, opt_state, weights)
                    weights = optax.apply_updates(weights, updates)
                else:                
                    # Basic GD
                    weights -= params.learning_rate * weight_grad 
            # if i > 10:
            #     break
            epoch_iterations.append(iterations)
        epoch_iterations = jnp.array(epoch_iterations).flatten()
        mean = jnp.mean(epoch_iterations)
        all_mean_iterations.append(mean)
        token, all_mean_iterations = gather_batch(token, all_mean_iterations)
        all_mean_iterations = all_mean_iterations.tolist()
        
        if split_rank != 0:
            jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points and a mean threshold of {}", rank, mean, epoch_iterations.shape[0], jnp.mean(empty_neuron_states.thresholds))
        
        # Inference on the validation set
        val_accuracy, val_mean, _ = batch_predict(params, key, token, weights, empty_neuron_states, dataset="val", save=False, debug=False)
        # val_accuracy, val_mean = 0, 0
        epoch_accuracy = 0.0
        if split_rank == last_rank:
            # Store loss values
            mean_loss = jnp.mean(jnp.array(epoch_loss))
            all_loss.append(mean_loss)
            token, mean_loss = gather_batch(token, mean_loss)

            # Store training and validation accuracies
            epoch_accuracy = epoch_correct / epoch_total
            all_epoch_accuracies.append(epoch_accuracy)
            all_validation_accuracies.append(val_accuracy)
            token, all_epoch_accuracies = gather_batch(token, all_epoch_accuracies)
            token, all_validation_accuracies = gather_batch(token, all_validation_accuracies)
            all_epoch_accuracies, all_validation_accuracies = all_epoch_accuracies.tolist(), all_validation_accuracies.tolist()
            if rank == size-1:
                jax.debug.print("Epoch {} , Training Accuracy: {:.2f}%, Validation Accuracy: {:.2f}%, mean loss: {}, mean val iterations: {}", epoch, all_epoch_accuracies[-1] * 100, val_accuracy * 100, mean_loss, val_mean)
                jax.debug.print("----------------------------\n")
        epoch_accuracy, token = bcast(epoch_accuracy, root=size-1, comm=comm, token=token)
        if epoch_accuracy >= 0.9999:
            break
    # Inference on the test set
    test_accuracy, test_mean, _ = batch_predict(params, key, token, weights, empty_neuron_states, dataset="test", save=False, debug=False)
    # test_accuracy = 0
    
    # Gather the weights and iteration values at the last layer
    layer_weights_sizes = []
    for layer in network_layers:
        layer_weights_sizes.append(layer.weights_shape)
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_iter_th(params, layer_weights_sizes, weights, jnp.array(all_mean_iterations), empty_neuron_states.thresholds, token)
    
    jax.block_until_ready(token)

    # Synchronize all MPI processes again
    token = mpi4jax.barrier(comm=comm, token=token)
    end_time = time.time()
    
    MAX_LEN = 256
    result_path = jnp.zeros(MAX_LEN, dtype=jnp.uint8)
    if rank == last_rank * process_per_layer:
        # Execution time
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.6f} seconds")
        result_path_str = store_training_data(params, 
                            "train",
                            all_epoch_accuracies, 
                            all_validation_accuracies, 
                            test_accuracy,
                            execution_time,
                            all_iteration_mean,
                            weights_dict,
                            all_loss, 
                            thresholds_dict,
                            opti)
        
        encoded = np.frombuffer(result_path_str.encode("utf-8"), dtype=np.uint8)
        if encoded.size > MAX_LEN:
            raise ValueError("result_path too long")
        padded = np.pad(encoded, (0, MAX_LEN - encoded.size), constant_values=0)
        result_path = jnp.array(padded)
    result_path, token = bcast(result_path, root=last_rank*process_per_layer, comm=comm, token=token)
    result_path = bytes(result_path).decode("utf-8").rstrip("\x00")
    token = mpi4jax.barrier(comm=comm, token=token)

    return result_path, token
    
# region SAVE DATA
def store_training_data(params, mode, all_epoch_accuracies, all_validation_accuracies, test_accuracy, execution_time, all_iteration_mean, weights_dict, all_loss, thresholds_dict, optiname): 
    filename_add_on = f"_{optiname}_"
       
    # Choose the saving folder
    if mode == "train":
        result_dir = os.path.join("network_results", params.dataset, "training")
        filename_header = f"{params.random_seed}" + f"_ep{params.num_epochs}" + f"_batch{params.batch_size}_"
    elif mode == "inference":
        result_dir = os.path.join("network_results", params.dataset, "inference")
        filename_header = f"{params.random_seed}" + f"_load{params.load_file}" + f"_batch{params.batch_size}_"
        all_iteration_mean = np.array(all_iteration_mean).flatten().tolist()
    else:
        print("Wrong mode for storing data choose 'train' or 'inference'. No data is saved")
        return          
    
    train_accuracy = float(all_epoch_accuracies[-1])
    val_accuracy = float(all_validation_accuracies[-1])   
    test_accuracy = float(test_accuracy)    

    jax.debug.print(
        "Final Training Accuracy: {train:.2f}%, Final Validation Accuracy: {val:.2f}%, Test Accuracy: {test:.2f}%",
        train=train_accuracy * 100 if train_accuracy != -1 else jnp.nan,
        val=val_accuracy * 100 if val_accuracy != -1 else jnp.nan,
        test=test_accuracy * 100 if test_accuracy != -1 else jnp.nan,
    )
    
    for acc in [train_accuracy, val_accuracy, test_accuracy]:
        if acc >= 0:
            accuracy = acc
    # Set up file path 
    filename = filename_header + "_".join(map(str, params.layer_sizes)) 
    filename += f"_acc{accuracy:.3f}" 
    if best:
        filename = "best_" + filename         

    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, filename) + filename_add_on
    
    if os.path.exists(result_path + ".json"):
        index = 1
        while True:
            new_result_path = result_path + f"({index})"
            if os.path.exists(new_result_path + ".json"):
                index+=1
            else:
                result_path = new_result_path
                break                

    # Store the results
    result_data = {
        "time": float(execution_time),
        "loadfile": params.load_file,
        "shuffle data": params.shuffle,
        "shuffle input": params.shuffle_input,
        "rerun": params.rerun,
        "processes": size,
        "firing number": params.firing_nb,
        "synchronization rate": params.sync_rate,
        "async layer": params.async_layer,
        "restrict": params.restrict,
        "threshold impact": params.threshold_impact,
        "threshold lr": params.threshold_lr,
        "test accuracy": test_accuracy,
        "layer_sizes": params.layer_sizes,
        "batch_size": params.batch_size,
        "learning rate": params.learning_rate,
        "training accuracy": np.array(all_epoch_accuracies).tolist(),
        "validation accuracy": np.array(all_validation_accuracies).tolist(),
        "iterations mean": np.array(all_iteration_mean).tolist(),
        "loss": [float(loss) for loss in all_loss],
        "thresholds": thresholds_dict,
        "weights": weights_dict
    }

    with open(result_path + ".json", "w") as f:
        json.dump(result_data, f, indent=4)

    print(f"Results saved to {result_path}")

    if mode == "train":
        epochs = [i + 1 for i in range(len(all_epoch_accuracies))]        
        # Plot accuracies and loss values
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(epochs, all_epoch_accuracies, 'o-', label='Training Accuracy')
        ax1.plot(epochs, all_validation_accuracies, 's-', label='Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f"Final Val Acc: {all_validation_accuracies[-1]:.4f} | Final Train Acc: {all_epoch_accuracies[-1]:.4f}")
        ax1.legend(loc='best')
        ax1.grid(True)

        # Secondary y-axis: loss
        ax2 = ax1.twinx()
        ax2.plot(epochs, all_loss, '^-', label='Training Loss', color='tab:red')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='best')
        
        # Save the plot as an image file
        plt.tight_layout()
        plt.savefig(result_path + ".png")
        plt.close()
        
        # Plot activation values
        plt.figure(figsize=(8, 5))
        for i, layer_values in enumerate(all_iteration_mean):
            plt.plot(epochs, layer_values, marker='o', label=f'Layer {i} (last: {layer_values[-1]:.1f})')

        plt.xlabel("Epoch")
        plt.ylabel("Average Activation Values")
        plt.title("Average Activation Values per Layer")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(result_path + "_activations.png") 
        plt.close()
    return result_path + ".json"
    
def accuracy(batch_number, outputs, y, iterations, print):
    # Get predictions (indices of max values)
    predictions = jnp.argmax(outputs, axis=-1)
    
    # Calculate accuracy for this batch
    valid_mask = y != -1
    valid_y = y[valid_mask]
    valid_predictions = predictions[valid_mask]

    batch_correct = jnp.sum(valid_predictions == valid_y)
    if print:
        jax.debug.print("Batch {}: Predictions: {}, True: {}, Iterations avg: {}, Correct: {}/{}, last network output: {}",
                batch_number, valid_predictions, valid_y, jnp.mean(iterations), batch_correct, valid_y.shape[0], outputs[-1])
    return valid_y, batch_correct


def pad_batch(batch_x, batch_y, batch_size):
    # Pad the x data with 0 and the y data with nan for the last batch
    current_size = batch_y.shape[0]
    if current_size < batch_size:
        pad_amount = batch_size - current_size
        pad_y = jnp.full((pad_amount,), -1.0, dtype=jnp.float32)
        pad_x = jnp.zeros((pad_amount,) + batch_x.shape[1:], dtype=batch_x.dtype)  
        # jax.debug.print("rank {}, has batch size: {} and pad batch size: {}", rank, current_size, pad_x.shape)
        batch_y = jnp.concatenate([batch_y, pad_y], axis=0)
        batch_x = jnp.concatenate([batch_x, pad_x], axis=0)
    
    return batch_x, batch_y

def preprocess_data(data_generator):
    # Preprocess the data
    preprocessed = data_generator
    
    return iter(preprocessed)

def gather_w_iter_th(params, layer_weights_sizes, weights, mean_iterations, thresholds, token):
    # Gather all the weights and iteration values at the last layer to store them
    leader_rank = split_rank * process_per_layer

    weights_dict = {}
    all_iteration_mean = []
    thresholds_dict = {}
    if split_rank != last_rank and rank == leader_rank:
        token = send(mean_iterations, dest=last_rank * process_per_layer, tag=5,comm=comm, token=token)
        if split_rank != 0:
            token = send(weights, dest=last_rank * process_per_layer, tag=5,comm=comm, token=token)
            token = send(thresholds, dest=last_rank * process_per_layer, tag=5,comm=comm, token=token)
    elif split_rank == last_rank and rank == leader_rank:
        for i in range(last_rank):
            # Storing mean iterations
            iter_mean, token = recv(mean_iterations, source=i * process_per_layer, tag=5, comm=comm, token=token)
            all_iteration_mean.append(iter_mean)
            if i == 0:
                continue
            
            # Storing the weights 
            w, token = recv(jnp.zeros(layer_weights_sizes[i]), source=i * process_per_layer, tag=5, comm=comm, token=token)   
            weights_dict[f"layer_{i}"] = w.tolist()
            
            # Storing the thresholds
            thr, token = recv(jnp.zeros(params.flat_layer_sizes[i]), source=i * process_per_layer, tag=5, comm=comm, token=token)
            thresholds_dict[f"thresholds_{i}"]= thr.tolist()
            
        all_iteration_mean.append(mean_iterations)  # Append the mean iterations of the last layer
        weights_dict[f"layer_{last_rank}"] = weights.tolist()
        all_iteration_mean = all_iteration_mean[1:] # Don't keep the value of the input layer
        print("all iteration mean: rank", rank, all_iteration_mean)

    return weights_dict, all_iteration_mean, thresholds_dict


# region Inference main
def batch_predict(params, key, token, network_layers, weights, empty_neuron_states, layer_computation, dataset:str="train", save=True, debug=True):    
    global training_generator
    global validation_generator
    global test_generator    
    
    token = mpi4jax.barrier(comm=comm, token=token)
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
    for i in range(total_batches):
        neuron_states = empty_neuron_states
        if split_rank == 0:                 
            token, batch_x, batch_y = split_batch(token, batch_iterator)
            print(f"batch {i}")
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
            # token, outputs, iterations, all_neuron_states = (predict_batched)(weights, neuron_states, token, max_nonzero, batch_x)
            # token, outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, token, jnp.array(batch_x))
            # break

            token, outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, token, jnp.array(batch_x), layer_computation)

            # Send label to the last layer
            token = send(batch_y, dest=last_rank * process_per_layer + rank, tag=10,comm=comm, token=token)
        else:
            # token, outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, token, jnp.zeros((batch_part, layer_sizes[0])))
            # batch_data = jnp.zeros((batch_part, params.max_nonzero, layer_sizes[0][0], layer_sizes[0][1], layer_sizes[0][2]))
            batch_data = jnp.zeros((batch_part, 1, 4))

            token, outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, token, batch_data, layer_computation)
            # token, outputs, iterations, all_neuron_states = (predict_batched)(weights, neuron_states, token, max_nonzero, jnp.zeros((batch_size, layer_sizes[0])))
            # jax.debug.print("Rank {} All neuron states values shape: {}, output shape : {}", rank, all_neuron_states.values.shape, outputs.shape)

            if split_rank == last_rank:
                y, token = recv(jnp.zeros((batch_part,)), source=rank - (last_rank * process_per_layer), tag=10, comm=comm, token=token)   
                
                valid_y, batch_correct = accuracy(i, outputs, y, iterations, print=False)                 
                
                epoch_correct += batch_correct
                epoch_total += valid_y.shape[0]
        epoch_iterations.append(iterations)
        # jax.debug.print("Rank {}, iterations: {}", rank, iterations)
        # break
    
    # print(f"Shape iterations before flattening: {jnp.array(epoch_iterations).shape}")
    epoch_iterations = jnp.array(epoch_iterations).flatten()
    mean = jnp.mean(epoch_iterations)
    # print(f"Rank {rank} finished epoch with mean {mean} with {epoch_iterations.shape} iterations")

    if split_rank != 0:
        token, mean = gather_batch(token, mean)
    # jax.debug.print("Rank {}, all iterations shape: {}", rank, (epoch_iterations.shape[0]))
    
    if rank != 0 and debug:
        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, epoch_iterations.shape[0]*process_per_layer)
    
    epoch_accuracy = -1.0
    if split_rank == last_rank:
        epoch_accuracy = epoch_correct / epoch_total
        token, epoch_accuracy = gather_batch(token, epoch_accuracy)
        if debug:
            jax.debug.print("Epoch Accuracy: {:.2f}%", epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")
    
    
    layer_weights_sizes = []
    for layer in network_layers:
        layer_weights_sizes.append(layer.weights_shape)
    print(f"rank {rank}: {layer_weights_sizes}")
    
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_iter_th(params, layer_weights_sizes, weights, mean, empty_neuron_states.thresholds, token)
    # jax.debug.print("rank {} all iterations mean: {}, shape: {}", rank, all_iteration_mean, (all_iteration_mean.shape))
    
    jax.block_until_ready(token)

    # Synchronize all MPI processes again
    token = mpi4jax.barrier(comm=comm, token=token)
    end_time = time.time()

    if rank == last_rank * process_per_layer:
        execution_time = end_time - start_time

        if debug:            
            print(f"Execution Time: {execution_time:.6f} seconds")
        if save:
            accuracies = {"train": [-1], "val": [-1], "test": [-1]}
            if dataset in accuracies:
                accuracies[dataset] = [epoch_accuracy]

            store_training_data(params, 
                                "inference",
                                accuracies["train"], 
                                accuracies["val"], 
                                accuracies["test"][0],
                                execution_time,
                                all_iteration_mean,
                                weights_dict,
                                [],
                                thresholds_dict,
                                "")
    return epoch_accuracy, mean, end_time - start_time

# region Main
def rerun_init(data_file_path, new_epoch_nb, th_lr=0, th_impact=0, async_layer=-1):
    with open(data_file_path, "r") as f:
        stored_data = json.load(f)

    load_file = stored_data["loadfile"]
    shuffle = stored_data["shuffle data"]
    shuffle_input = stored_data["shuffle input"]
    firing_nb = stored_data["firing number"]
    sync_rate = stored_data["synchronization rate"]
    layer_sizes = tuple(stored_data["layer_sizes"])
    batch_size = stored_data["batch_size"]
    learning_rate = stored_data["learning rate"]
    init_thresholds = stored_data["thresholds"]["thresholds_0"][0]
    threshold_dict = stored_data["thresholds"]
    restrict = stored_data["restrict"]
    threshold_impact = stored_data["threshold impact"]
    threshold_lr = stored_data["threshold lr"]
    weights_dict = stored_data["weights"]

    params = Params(
        random_seed=random_seed,
        layer_sizes=layer_sizes, 
        init_thresholds=init_thresholds, 
        num_epochs=new_epoch_nb, 
        learning_rate=learning_rate, 
        batch_size=batch_size,
        load_file=load_file,
        shuffle=shuffle,
        restrict=restrict,
        firing_nb=firing_nb,
        sync_rate=sync_rate,
        max_nonzero=max_nonzero,
        shuffle_input=shuffle_input,
        threshold_lr=th_lr,
        threshold_impact=th_impact,
        rerun=data_file_path,
        async_layer=async_layer
    )
    
    weights = jnp.array(weights_dict["layer_"+str(split_rank)])
    thresholds = jnp.zeros(layer_sizes[split_rank])
    if split_rank < last_rank:
        thresholds = jnp.array(threshold_dict["thresholds_"+str(split_rank)])
    return params, weights, thresholds

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

def wait_for_file(path, timeout=10, poll_interval=0.1):
    """Wait for a file to exist and be non-empty."""
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        time.sleep(poll_interval)
    raise TimeoutError(f"File {path} not found or still empty after {timeout} seconds.")


@partial(jax.jit, static_argnames=['params'])
def test_jit(params, val):
    jax.debug.print("Test rank {}", rank)
    return None

if __name__ == "__main__":
    STOP_flag = 0
    random_seed = 42
    key = jax.random.key(random_seed)
    
    dataset = 'mnist'
    dataset = 'shd'
    dataset = 'nmnist'
    
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
    all_layers.append(( (2, 34, 34), # (channel, height, width)
                        (3, (3,3), (1,1), (1,1)), # (out_channel, kernel_size, padding, stride)
                        (1, (5,5), (2,2), (1,1)), 
                        (64,), # Fully connected layer
                        (10,)))
    
    # all_layers.append(( (1, 5, 5), # (channel, height, width)
    #                     (2, (3,3), (1,1), (1,1)), # (out_channel, kernel_size, padding, stride)
    #                     (10,)))


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
    batch_size = 128
    shuffle = False
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
                            loader = torch_mnist_loader_manual
                        case "shd":
                            loader = torch_SHD_loader
                        case "nmnist":
                            loader = torch_nmnist_loader
                        case _:
                            raise ValueError(f"Unknown dataset: {dataset}")
                    # Load the data 
                    (training_generator, total_train_batches), (validation_generator, total_val_batches), (test_generator, total_test_batches), max_nonzero = loader(batch_size, shuffle=shuffle)
                    
                    batch_x, batch_y = next(iter(training_generator))
                    print(batch_x.shape, batch_y.shape)
                    if len(batch_x.shape) == 5 and batch_x.shape[2:] != layer_sizes[0]:
                        STOP_flag = 1
            
            STOP_flag, token = bcast(jnp.array([STOP_flag]), root=0, comm=comm)
            if STOP_flag:
                print(f"Error: make sure that the input layer has the same dimensions as the data.")
                sys.exit(1)
                
            # Broadcast total_batches to all other ranks
            (total_train_batches, total_val_batches, total_test_batches), token = bcast(jnp.array([total_train_batches, total_val_batches, total_test_batches]), root=0 , comm=comm, token=token)                
            max_nonzero, token = bcast(jnp.array([max_nonzero]), root=0 , comm=comm)
            max_nonzero = max_nonzero.tolist()[0]
            
            params = Params(
                dataset=dataset,
                random_seed=random_seed,
                layer_sizes=layer_sizes, 
                init_thresholds=init_thresholds, 
                num_epochs=20, 
                learning_rate=0.0001, 
                batch_size=batch_size,
                load_file=load_file,
                shuffle=shuffle,
                restrict=restrict,
                firing_nb=128,
                sync_rate=20000,
                max_nonzero=max_nonzero,
                shuffle_input=False,
                threshold_lr=0.0, 
                threshold_impact=0.0, # Beta sparse
                rerun="",
                async_layer=async_layer,
                max_kernel=max_kernel,
                flat_layer_sizes=()
            )
            folder = "" #"network_results/training/"
            # rerun = "42_ep20_batch36_784_128_64_10_acc0.967_adam_.json"
            # rerun = "42_ep20_batch36_784_128_64_10_acc0.973_adam_.json"
            # rerun = "42_ep1_batch36_784_128_64_10_acc0.799_adam_.json"
            # rerun = None
            if rerun is not None:
                new_epoch_number = 21 # Number of training epoch to run again
                th_lr, beta = 0.0, 0.0
                
                if async_layer == -1 or async_layer >= last_rank:
                    cont = False
                    continue
                else:
                    async_layer += 1
                
                # if i % 2:
                #     new_epoch_number = 1
                #     beta = 0.01
                
                params, weights, thresholds = rerun_init(folder+rerun, new_epoch_number, th_lr, beta, async_layer=async_layer)
                if len(layer_sizes) != len(params.layer_sizes):
                    print(f"Error: rerun file {rerun} has different layer sizes than the current network structure {layer_sizes}.")
                    sys.exit(1)
            
            key, subkey = jax.random.split(key) 
            network = Network(key, params, layer_sizes=layer_sizes)
            weights = network.init_weights()

            flat_layer_sizes = []
            for layer in network.layers:
                flat_layer_sizes.append(layer.values.shape)
            if rank ==0:
                print(f"rank {rank}: flat layer sizes:{flat_layer_sizes}")
            params = dataclasses.replace(params, flat_layer_sizes=tuple(flat_layer_sizes))
            empty_neuron_states = network.layers[split_rank]
            # print(rank, empty_neuron_states.values.shape)
                        
            if rank == 0:
                print(f"Number of training batches: {total_train_batches}, validation batches: {total_val_batches}, test batches: {total_test_batches}")
                print(params)
                        
            t = 100
            all_time = 0
            # for i in range(t):
            #     _, _, ex_time = batch_predict(params, key, token, weights, empty_neuron_states, "val", save=False, debug=True)
            #     all_time += ex_time
            # print("average execution time : {}", all_time/t)
            layer_computation = fc_layer_computation
            if empty_neuron_states.is_conv:
                layer_computation = conv_layer_computation
            
            print(f"rank {rank}, is conv: {empty_neuron_states.is_conv}, weights shape: {weights.shape}")
            # batch_predict(params, key, token, network.layers, weights, empty_neuron_states, layer_computation, 'test', save=True, debug=True)
            # break
            result_path, token = train(token, params, key, network.layers, weights, empty_neuron_states, layer_computation, "adam")
            # rerun = result_path
            # print(rerun)
            
            
            
            
            
            
            
            
            
            
            
            
#______________________________________________Random Tests (can be deleted)________________________________________________________________________________________________
# previous_layer = jnp.zeros((2, 34, 34))
# layer = (6, (3,3), (1,1), (1,1))
# out_chan, kernel, padding, stride = layer
# in_shape = previous_layer.shape
# h_out = (in_shape[1] + 2 * padding[0] - kernel[0]) // stride[0] + 1
# w_out = (in_shape[2] + 2 * padding[1] - kernel[1]) // stride[1] + 1
# values = jnp.zeros((out_chan, h_out, w_out))  # Initialize values for convolutional layer
# in_chan = previous_layer.shape[0]
# key, subkey = jax.random.split(key) 
# thresholds = jax.random.normal(subkey, values.shape) * params.init_thresholds
# empty_conv_neuron = Conv_Neuron(
#                     neuron_state = Neuron_states(
#                                     values=values, 
#                                     thresholds=thresholds, 
#                                     input_residuals=np.zeros(previous_layer.shape),
#                                     weight_residuals={"input order": jnp.full(previous_layer.shape, -1, dtype=int), 
#                                                     "input activity": jnp.full(previous_layer.shape, 0, dtype=int), 
#                                                     "layer activity": jnp.zeros((layer[0],), dtype=int), 
#                                                     "output activity": jnp.zeros(previous_layer.shape)},
#                                     last_sent_iteration=0,
#                                     weights_shape=(out_chan, in_chan, kernel[0], kernel[1])),
#                     kernel=kernel,
#                     padding=padding,
#                     stride=stride,
#                     previous_layer=previous_layer
#                     )   

# layer_sizes = (28*28, 128, 64, 32, 10)
# empty_neuron_states = Neuron_states(
#                     values=jnp.zeros((layer_sizes[split_rank])), 
#                     thresholds=thresholds, 
#                     input_residuals=np.zeros((layer_sizes[split_rank-1],)),
#                     weight_residuals={"input order": jnp.full((layer_sizes[split_rank-1],), -1, dtype=int), 
#                                     "input activity": jnp.full((layer_sizes[split_rank-1],), 0, dtype=int), 
#                                     "layer activity": jnp.zeros((layer_sizes[split_rank],), dtype=int), 
#                                     "output activity": jnp.zeros((layer_sizes[split_rank-1], layer_sizes[split_rank]))},
#                     last_sent_iteration=0,
#                     weights_shape=(layer_sizes[split_rank-1], layer_sizes[split_rank])
#                     )

# test_jit(params, network.layers)