import os
from tqdm import tqdm

from async_MLP import layer_computation
from mpi4py import MPI
# os.environ["JAX_TRACEBACK_FILTERING"] = "on"
os.environ.pop("JAX_TRACEBACK_FILTERING", None)

import jax
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
from dataset_helpers.dvs_helper import torch_DVSGesture_loader
from dataset_helpers.cnn_mnist import get_weights_for_rank

from other_helpers.helpers import Params, NeuronStates
from other_helpers.helpers import accuracy, store_training_data, rerun_init, store_data_to_json
from other_helpers.helpers import activation_func, keep_top_k, output_vector_to_event
from other_helpers.helpers import update_history, process_history, load_config_with_defaults, parse_unknown_args_and_overrides_config
from other_helpers.backpropagation import MLP_back_prop
from other_helpers.loss_functions import loss_bpp, mean_loss
from other_helpers.MPI_helpers import MPIConfig, combine_batch_avg, gather_batch, split_batch, l2_weight_regularization
from other_helpers.event_pooling import output_to_event_array_with_pooling, full_matrix_to_event_array_with_pooling

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

layer_idx = None           # Rank corresponding to the layer
process_per_layer = None    # Number of processes for each layer
last_layer = None            # Rank of last layer
batch_part = None           # The size of the batch on each process
mpi_config = None           # MPIConfig class for mpi helpers functions

training_generator = None
validation_generator = None
test_generator = None

TQDM_DISABLE = False
STORE_EACH_EPOCH = True

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
                 weight_res: jnp.ndarray,
                 kernel: tuple[int, int],
                 padding: tuple[int, int] | str, # Padding can be either a tuple for padding in each direction or a string, "SAME"=keep same size, "VALID"=No padding
                 stride: tuple[int, int],
                 previous_layer: jnp.ndarray,
                 is_conv: bool = True,
                 pooling: str = "",
                 pool_size: tuple[int, int] = (2, 2),
                 pool_stride: tuple[int, int] = (2, 2)):

        self.neuron_state = neuron_state
        self.weight_res = weight_res
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
                    self.previous_layer,
                    self.weight_res)
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
        neuron_state, previous_layer, weight_res = children
        return cls(neuron_state, weight_res, kernel, padding, stride, previous_layer, 
                   is_conv, pooling, pool_size, pool_stride)
    
    def replace(self, **updates):
        """
        Return a new ConvNeuronStates object with updated fields.
        Supports replacing fields at both the ConvNeuronStates and NeuronStates levels.
        """
        # If we need to update inner neuron_state fields, do that first
        neuron_state_updates = {
            k: v for k, v in updates.items() if hasattr(self.neuron_state, k)
        }

        new_neuron_state = (
            self.neuron_state.replace(**neuron_state_updates)
            if neuron_state_updates
            else self.neuron_state
        )

        # Then replace fields at the ConvNeuronStates level
        return ConvNeuronStates(
            neuron_state=updates.get("neuron_state", new_neuron_state),
            weight_res=updates.get("weight_res", self.weight_res),
            kernel=updates.get("kernel", self.kernel),
            padding=updates.get("padding", self.padding),
            stride=updates.get("stride", self.stride),
            previous_layer=updates.get("previous_layer", self.previous_layer),
            is_conv=updates.get("is_conv", self.is_conv),
            pooling=updates.get("pooling", self.pooling),
            pool_size=updates.get("pool_size", self.pool_size),
            pool_stride=updates.get("pool_stride", self.pool_stride),
        )

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
                        - for a fully connected layer each tuple contains a single integer 
                            representing the number of neurons in that layer
                        - for a convolutional layer each tuple contains (out_chan, kernel, padding, stride) 
                            representing the output channel, kernel size, padding and stride.
        
        This function computes the shapes of each layer's neuron states using the specified parameters
        and initializes the neuron states with the correct shapes and values 
        '''
        layers = []             # Holds the empty neuron states of each layer  
        conv_layer_sizes = []   # Holds the shapes of the layers before pooling is applied
        flat_layer_sizes = []   # Holds the shapes of the layers after pooling is applied
        previous_layer = jnp.zeros(0)  # placeholder for input shape tracking
        filename = f"_b{params.batch_size}"

        for i, layer in enumerate(layer_sizes):
            if len(layer) == 1: # Linear layer
                if i == 0: # Linear layer input
                    prev_size = 1
                else:
                    # Previous layer is Linear layer: Use the previous layer's shape
                    # Previous layer is Conv layer: Flatten and use the size of the vector 
                    prev_size = previous_layer if isinstance(previous_layer, int) else previous_layer.flatten().size
                    if rank == 0 and debug:
                        print(f"rank {rank}, Previous layer: {prev_size}")

                key, subkey = jax.random.split(key) 
                # thresholds = jax.random.uniform(subkey, layer) * params.init_thresholds + th_bias
                thresholds = jnp.full(layer, params.init_thresholds)
                empty_neuron_states = NeuronStates(
                                    values=jnp.zeros(layer),
                                    thresholds=thresholds,
                                    input_residuals=jnp.zeros((prev_size,)),
                                    output_residuals=jnp.zeros(layer),
                                    input_order=jnp.full((prev_size,), -1, dtype=int),
                                    input_activity=jnp.zeros((prev_size,), dtype=int),
                                    layer_activity=jnp.zeros((layer[0],), dtype=int),
                                    output_activity=jnp.zeros((prev_size, layer[0])),
                                    last_sent_iteration=-1,
                                    input_vector=jnp.zeros((prev_size,)),
                                    output_vector=jnp.zeros((layer[0],)),
                                    values_history=jnp.zeros((params.history_size, layer[0])),
                                    history_index=jnp.array(0, dtype=jnp.int32),
                                    weights_shape=(prev_size, layer[0]),
                                    is_conv=False,
                )
                layers.append(empty_neuron_states)
                previous_layer = layer[0]
                filename += f"_L{layer[0]}"
                flat_layer_sizes.append(layer)
                conv_layer_sizes.append(layer)
            else:   # Convolutional layer
                in_chan = previous_layer.shape[0]
                pool_size = (2, 2)
                pool_stride = (2, 2)
                pooling = ""
                if len(layer) > 4: # If pooling parameters are specified
                    pooling = layer[4]
                    if len(layer) > 5 : pool_size = layer[5]
                    if len(layer) > 6 : pool_stride = layer[6]
                    layer = layer[:4]
                
                if i == 0: # Input layer
                    previous_layer = jnp.zeros(1)
                    values = jnp.zeros(layer)
                    out_chan, kernel, padding, stride = 1, (0,0), (0,0), (0,0) # Values used as placeholders for the input layer
                    filename += "_C{}x{}x{}".format(*layer)
                    out_chan, h_out, w_out = layer
                else:
                    # Compute the shape of this layer
                    out_chan, kernel, padding, stride = layer
                    in_shape = previous_layer.shape
                    h_out = (in_shape[1] + 2 * padding[0] - kernel[0]) // stride[0] + 1
                    w_out = (in_shape[2] + 2 * padding[1] - kernel[1]) // stride[1] + 1

                    if rank == 0 and debug:
                        print(f"rank {rank}, previous layer shape: {in_shape}, out shape: {(out_chan, h_out, w_out)}, kernel: {kernel}, padding: {padding}, stride: {stride}")
                    values = jnp.zeros((out_chan, h_out, w_out))  # Initialize values for convolutional layer
                    filename += f"_C{out_chan}x{in_chan}x{kernel[0]}x{kernel[1]}"

                h_out_pool, w_out_pool = h_out, w_out
                if pooling != "": # Compute the shape after pooling if needed
                    h_out_pool = (h_out - pool_size[0]) // pool_stride[0] + 1
                    w_out_pool = (w_out - pool_size[1]) // pool_stride[1] + 1
                    if pooling == "max":
                        filename += f"_P{pool_size[0]}x{pool_size[1]}"
                    elif pooling == "avg":
                        filename += f"_AvgP{pool_size[0]}x{pool_size[1]}"

                key, subkey = jax.random.split(key) 
                # thresholds = jax.random.uniform(subkey, values.shape) * params.init_thresholds + th_bias
                thresholds = jnp.full(values.shape, params.init_thresholds)
                weights_shape = (out_chan, in_chan, kernel[0], kernel[1])
                neuron_state = NeuronStates(
                    values=values,
                    thresholds=thresholds,
                    input_residuals=jnp.zeros(previous_layer.shape),
                    output_residuals=jnp.zeros(values.shape),
                    input_order=jnp.full(previous_layer.shape, -1, dtype=int),
                    input_activity=jnp.zeros(previous_layer.shape, dtype=int),
                    layer_activity=jnp.zeros(values.shape),
                    output_activity=jnp.zeros_like(values),  # placeholder, shape matches values
                    last_sent_iteration=-1,
                    input_vector=jnp.zeros(previous_layer.shape),
                    output_vector=jnp.zeros(values.shape),
                    values_history=jnp.zeros((params.history_size, *values.shape)),
                    history_index=jnp.array(0, dtype=jnp.int32),
                    weights_shape=weights_shape,
                    is_conv=True
                )

                empty_conv_neuron = ConvNeuronStates(
                    neuron_state=neuron_state,
                    weight_res=jnp.zeros(weights_shape),
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
                flat_layer_sizes.append(previous_layer.shape)   # Shape after pooling
                conv_layer_sizes.append(values.shape)           # Shape before pooling, needed to gather the thresholds after computation
        return cls(params=params, key=key, layers=tuple(layers), flat_layer_sizes=tuple(flat_layer_sizes), conv_layer_sizes=tuple(conv_layer_sizes), filename=filename)

    def init_weights(self):
        '''
        Initialize the weights for each layer based on the layer sizes.
        
        Returns the weights correponding to the MPI layer_idx.
        ''' 
        weights = init_params(self.key, self.layers, self.params, self.filename)
        return weights
    
    def rerun(self, thresholds):
        """
        Returns the new empty neuron states with the updated thresholds specified in the arguments.
        """
        if thresholds is not None:
            # print(layer_idx, type(self.layers[layer_idx]))
            old_layer = self.layers[layer_idx]
            layers_list = list(self.layers)

            if isinstance(old_layer, ConvNeuronStates):
                new_neuron_state = old_layer.neuron_state.replace(thresholds=thresholds)
                layers_list[layer_idx] = old_layer.replace(neuron_state=new_neuron_state)
            else:
                layers_list[layer_idx] = old_layer.replace(thresholds=thresholds)

            return layers_list[layer_idx]
        return self.layers[layer_idx]
    
    def tree_flatten(self):
        # children are arrays or other pytree objects
        children = (self.params, self.layers, self.key)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params, layers, key = children
        return cls(params=params, layers=layers, key=key)

def init_params(key, layers, params, filename="", best=False):
    # Initialize weights for each layer
    keys = jax.random.split(key, len(layers))
    load_file = params.load_file

    if layer_idx != 0:
        if load_file:
            folder = f"tensor_data/CNN/{params.dataset}/"
            f = "tensor_data"+filename+".npz"
            return get_weights_for_rank(folder+f, layer_idx)

        # Random initialization of the weights       
        layer = layers[layer_idx]
        weights_shape = layer.weights_shape
    
        if layer.is_conv:
            out_ch, in_ch, kh, kw = weights_shape            
            fan_in = in_ch * kh * kw
        else:
            fan_in = weights_shape[0]
            fan_out = weights_shape[1]
        
            # bound = jnp.sqrt(6.0 / (fan_in+fan_out))
            # return jax.random.uniform(keys[layer_idx], weights_shape, jnp.float32, -bound, bound)
            # std = jnp.sqrt(2.0 / fan_in)    
            std = 1e-2 
            weights_linear_layer = std * jax.random.normal(keys[layer_idx], weights_shape)
            return weights_linear_layer
        
        # Kaiming He Uniform initialization
        bound = jnp.sqrt(6.0 / fan_in)
        weights_conv_layer = jax.random.uniform(keys[layer_idx], weights_shape, jnp.float32, -bound, bound)
        
        # Kaiming He Normal initialization
        std = jnp.sqrt(2.0 / fan_in)
        weights_conv_layer = std * jax.random.normal(keys[layer_idx], weights_shape)


        bound = jnp.sqrt(2.0 / fan_in)
        weights_conv_layer = jax.random.uniform(keys[layer_idx], weights_shape, jnp.float32, -bound, bound)
        return weights_conv_layer
    else:
        return jnp.zeros((1,1,1,1)) # Return an empty holder for the weights of the input layer
    
#region FC computation 
@partial(jax.jit, static_argnames=['params', 'grad',])
def fc_layer_computation(params, key, neuron_idx, layer_input, weights, neuron_states, iteration=0, grad=False):    
    c, x, y = neuron_idx

    # Compute the flattened neuron index
    C, H, W = 0, 0, 0
    flat_layer_size = params.flat_layer_sizes[layer_idx-1]
    if len(flat_layer_size) == 3: # Fail safe in case there is no hidden layer
        C, H, W = flat_layer_size
    neuron_idx = c * (H * W) + x * W + y 
    
    # Compute the new values of the neuron states
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

    if grad:
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
    else:
        new_input_residuals = neuron_states.input_residuals
        new_input_activity = neuron_states.input_activity
    
    @jit
    def last_layer_case(_): # No need for additional computation at the output layer
        new_values_history, new_history_index = neuron_states.values_history, neuron_states.history_index
        if params.history_size > 0:
            new_values_history, new_history_index = update_history(new_values_history, new_history_index, activations)

        return jnp.array(0), jnp.zeros((activations.shape[0], 4)), NeuronStates(  values=activations, 
                                                                    thresholds=neuron_states.thresholds, 
                                                                    input_residuals=new_input_residuals, 
                                                                    output_residuals=neuron_states.output_residuals, 
                                                                    input_order=neuron_states.input_order, 
                                                                    input_activity=new_input_activity,
                                                                    layer_activity=neuron_states.layer_activity,
                                                                    output_activity=neuron_states.output_activity,
                                                                    last_sent_iteration=neuron_states.last_sent_iteration,
                                                                    input_vector=neuron_states.input_vector,
                                                                    output_vector=neuron_states.output_vector,
                                                                    values_history=new_values_history,
                                                                    history_index=new_history_index,
                                                                    weights_shape=neuron_states.weights_shape,
                                                                    is_conv=neuron_states.is_conv)
    
    @jit
    def hidden_layer_case(_):
        fire = (iteration-neuron_states.last_sent_iteration) >= params.sync_rate # Fire if sync rate reached
        fire = jnp.logical_or(fire, neuron_idx < 0) # Fire if last input received

        # APPLY THE SYNC RATE  
        activated_output = jax.lax.cond(fire, 
                                        lambda args: activation_func(args[0], args[1]), 
                                        lambda _: jnp.zeros(activations.shape),
                                        (neuron_states.thresholds, activations))
        
        # APPLY THE FIRING NUMBER        
        f_nb = params.firing_nb
        if isinstance(f_nb, int):
            activated_output = keep_top_k(activated_output, f_nb) # Get the top k activations
        else:
            activated_output = keep_top_k(activated_output, f_nb[layer_idx]) # Get the top k activations
                
        # APPLY THE RESTRICTION
        reset = params.restrict
        if not isinstance(reset, int):
            reset = reset[layer_idx]
        penalty = jax.lax.cond(reset <= 0,
                               lambda _: activated_output, 
                               lambda _: activated_output*reset, None)
        
        if grad:
            new_output_residuals = neuron_states.output_residuals + activated_output 

            # Update the layer activity by adding the neurons that activated
            active_indexes = jnp.where(activated_output > 0, 1, 0)
            new_layer_activity = neuron_states.layer_activity + active_indexes

            last_neuron_idx = jnp.argmax(neuron_states.input_order) # Last neuron index in the input order
            new_neuron_idx = jax.lax.cond(neuron_idx < 0,
                        lambda _: last_neuron_idx, 
                        lambda _: neuron_idx,
                        None)
            
            new_input_order = neuron_states.input_order.at[new_neuron_idx].set(iteration) # Update the input activity by setting the input neuron to the iteration number        
            
            new_output_activity = neuron_states.output_activity.at[new_neuron_idx].add(active_indexes)
            
            # Added +1 so that we can differentiate between never activated (0) and activated at iteration 0 (1)
            new_input_vector = neuron_states.input_vector.at[neuron_idx].set(iteration+1)   # Set the input neuron to the iteration at which the input was received
            new_output_vector = jnp.where(activated_output > 0,                             # Set the output neuron to the last iteration at which it activated
                                        iteration+1,
                                        neuron_states.output_vector)
        else:
            new_output_residuals=neuron_states.output_residuals
            new_layer_activity = neuron_states.layer_activity
            new_input_order = neuron_states.input_order
            new_output_activity = neuron_states.output_activity
            new_input_vector = neuron_states.input_vector
            new_output_vector = neuron_states.output_vector

        
        new_last_sent_iteration = jax.lax.cond(fire, lambda _: iteration, lambda _: neuron_states.last_sent_iteration, None)
        new_neuron_states = NeuronStates(   values=activations - penalty, 
                                            thresholds=neuron_states.thresholds, 
                                            input_residuals=new_input_residuals, 
                                            output_residuals=new_output_residuals, 
                                            input_order=new_input_order, 
                                            input_activity=new_input_activity,
                                            layer_activity=new_layer_activity,
                                            output_activity=new_output_activity,
                                            last_sent_iteration=new_last_sent_iteration,
                                            input_vector=new_input_vector,
                                            output_vector=new_output_vector,
                                            values_history=neuron_states.values_history,
                                            history_index=neuron_states.history_index,
                                            weights_shape=neuron_states.weights_shape,
                                            is_conv=neuron_states.is_conv)
        
        nb_valid_elements = jnp.count_nonzero(activated_output)
        processed_output = output_vector_to_event(key, activated_output, params, params.layer_sizes[layer_idx][0]) 

        # jax.debug.print("rank {} iteration {} nb valid elements {}, activated output {}", rank, iteration, nb_valid_elements, processed_output[0])
        # Pad to CNN format
        shaped_activated_output = jnp.pad(processed_output, ((0, 0), (2, 0)), constant_values=-2) 
        return nb_valid_elements, shaped_activated_output, new_neuron_states
    
    cond = layer_idx == last_layer
    return jax.lax.cond(cond, last_layer_case, hidden_layer_case, None)


#region Conv computation
@partial(jax.jit, static_argnames=['params', 'grad',])
def conv_layer_computation(params, key, neuron_idx, layer_input, weights, neuron_states, iteration=0, grad=False):
    '''
    Apply the convolution for an incoming event in the event-driven manner described in "Optimizing event-based neural networks on digital neuromorphic architecture: a comprehensive design space exploration"
    This convolution only supports 'SAME' padding scheme with stride 1
    
    weights: (out_ch, in_ch, k_h, k_w)
    '''
    out_ch, in_ch, k_h, k_w = weights.shape
    H, W = neuron_states.values.shape[1], neuron_states.values.shape[2] # c, h, w
    c, x, y = neuron_idx
    @jit
    def regular_input(neuron_states):
        # jax.debug.print("rank {} has x: {}, y: {}", rank, x, y)

        # Step 1: Multiply the input value by the flipped kernel to obtain the partial output values
        partial_activations = jnp.dot(layer_input, jnp.flip(weights[:, c, :, :], axis=(1, 2))) # Shape (out_ch, k_h, k_w) 
        # jax.debug.print("activations: {}", activations)        
        
        # Step 2: Extract the current values from the padded value and threshold matrices
        pad_h, pad_w = neuron_states.padding
        event_pad_h, event_pad_w = k_h-1 - pad_h, k_w-1 - pad_w # Event convolution requires a different padding depending on the required padding 

        values_padded = jnp.pad(neuron_states.values, ((0, 0), (event_pad_h, event_pad_h), (event_pad_w, event_pad_w)), constant_values=-10000) # Pad the values in neuron states with very negative value 
        thresholds_padded = jnp.pad(neuron_states.thresholds, ((0, 0), (event_pad_h, event_pad_h), (event_pad_w, event_pad_w))) # Pad the thresholds 
        # jax.debug.print("rank {}, neuron idx {}, padding: {}, start indices: {}", rank, neuron_idx, neuron_states.padding, start_indices)

        start_indices = (0, x, y) # Start indices for slicing and updating on padded matrices, always the same because the padding takes care of the needed offset
        slice_shape = partial_activations.shape  # (C, k_h, k_w)

        # jax.debug.print("rank {}, neuron idx: {}, start indices: {}, slice shape: {}, values padded shape: {}", rank, neuron_idx, start_indices, slice_shape, values_padded.shape)
        
        current_values_sliced = jax.lax.dynamic_slice(values_padded, start_indices, slice_shape)
        thresholds_sliced = jax.lax.dynamic_slice(thresholds_padded, start_indices, slice_shape)

        padding_mask = jnp.where(current_values_sliced == -10000, 0.0, 1.0) # Mask to zero out the padded values
        # jax.debug.print("rank {}, partial activations {}, current values sliced {}, start indices {}, padding mask {}", rank, partial_activations, current_values_sliced, start_indices, padding_mask)
        
        # Step 3: Add the partial output values to the current values to get the complete output values
        activations = (current_values_sliced + partial_activations) * padding_mask
        updated_values_slice = activations
        
        # Step 4: Apply sync rate: Add 1 to the internal counter for sync rate, if counter exceeds it we fire
        activity_slice = jax.lax.dynamic_slice(neuron_states.output_activity, start_indices, slice_shape)
        ne_activity_slice = activity_slice + 1
        fire_mask = jnp.where(ne_activity_slice >= params.sync_rate, 1, 0)
        activations *= fire_mask  # Only fire where the sync rate is reached

        # Step 5: Compute ReLu on the updated slice if fire is True
        activated_output = activation_func(thresholds_sliced, activations)
        # jax.debug.print("rank {}, input: {}, activations: {}, updated slice: {}, activated output: {}", rank, layer_input, activations, updated_slice, activated_output)

        # Step 6: Apply the firing number        
        f_nb = params.firing_nb
        if isinstance(f_nb, int):
            activated_output = keep_top_k(activated_output, f_nb, max_kernel=params.max_kernel) # Get the top k activations
        else:
            activated_output = keep_top_k(activated_output, f_nb[layer_idx], max_kernel=params.max_kernel) # Get the top k activations
        
        # Step 7: Update the internal activity counter by resetting it for the neurons that have fired
        activation_mask = jnp.where(activated_output > 0, 0, 1) # Reset the activity counter where a neuron has fired
        new_activity_slice = ne_activity_slice * activation_mask

        new_output_activity = jax.lax.dynamic_update_slice( # Write back to the internal counter
            neuron_states.output_activity, 
            new_activity_slice, 
            start_indices
        )
        # jax.debug.print("rank {}, activity_slice {}, new activity_slice {}, fire_mask {}, final activity slice {}", 
        #                 rank, activity_slice, ne_activity_slice, fire_mask, new_activity_slice)
        
        reset = params.restrict
        if not isinstance(reset, int):
            reset = reset[layer_idx]
        # Step 8: Apply the restriction
        penalty = jax.lax.cond( reset <= 0, 
                                lambda _: activated_output, 
                                lambda _: activated_output*reset, None)
        
        # Step 9: Compute remaining values
        remaining_value = updated_values_slice - penalty
        # jax.debug.print("rank {}, updated values slice {}, remaining value {}, activated output {}", rank, updated_values_slice, remaining_value, activated_output)

        values_padded = jax.lax.dynamic_update_slice(values_padded, remaining_value, start_indices)
        
        new_values = neuron_states.values.at[:,:,:].set(
                                                values_padded[:, event_pad_h:H+event_pad_h, event_pad_w:W+event_pad_w]
                                                )
        # jax.debug.print("rank {}, old values {}, values padded {}, new values {}", rank, neuron_states.values, values_padded, new_values)        

        # Step 10: Apply pooling and compute the output events 
        nb_valid_elements, out_events, unpooled_coords, unpooled_vals = output_to_event_array_with_pooling(activated_output, 
                                                                       start_indices, 
                                                                       new_values.shape,
                                                                       (event_pad_h, event_pad_w),
                                                                       neuron_states.pooling,
                                                                       neuron_states.pool_size,
                                                                       neuron_states.pool_stride,
                                                                       rank)
        # jax.debug.print("rank {}, neuron_idx {}, activated output \n{}, nb valid elements {}, out events {}", 
        #                  rank, neuron_idx, activated_output, nb_valid_elements, out_events)        

        # jax.debug.print("rank {} unpooled coords {}, unpooled vals {}  unpooled-x {}, unpooled-y {}", 
        #                 rank, unpooled_coords, unpooled_vals, unpooled_coords[:, 1]-x+event_pad_h, unpooled_coords[:, 2]-y+event_pad_w)
        # jax.debug.print("___________________________________________________________________________")

        if grad:
            # Step 11: Update the neuron state
            valid_els = jnp.where(unpooled_vals != 0, 1, 0)
            new_weight_res = neuron_states.weight_res.at[   unpooled_coords[:, 0], 
                                                            c,
                                                            unpooled_coords[:, 1]-x+event_pad_h, 
                                                            unpooled_coords[:, 2]-y+event_pad_w
                                                        ].add(valid_els)
            
            new_layer_activity = neuron_states.layer_activity.at[   unpooled_coords[:, 0], 
                                                                    unpooled_coords[:, 1], 
                                                                    unpooled_coords[:, 2]
                                                                ].add(jnp.where(unpooled_vals != 0, 1, 0))
            # new_layer_activity = neuron_states.layer_activity.at[   unpooled_coords[:, 0], 
            #                                                         unpooled_coords[:, 1], 
            #                                                         unpooled_coords[:, 2]
            #                                                     ].add(unpooled_vals[:])
            # jax.debug.print("rank {} unpooled coords {}, unpooled vals {}, new layer activity {}", rank, unpooled_coords, unpooled_vals, new_layer_activity)
            # jax.lax.cond(jnp.all(neuron_states.layer_activity == 0), 
            #             lambda _: None,
            #             lambda _: jax.debug.print("rank {} unpooled coords {}, unpooled vals {}, new layer activity {}", rank, unpooled_coords, unpooled_vals, new_layer_activity),
            #             None)

            input_act = neuron_states.input_activity
            new_input_activity = jax.lax.cond(nb_valid_elements > 0, lambda _: input_act.at[neuron_idx].add(1), lambda _: input_act, None)
            new_input_residuals = neuron_states.input_residuals.at[tuple(neuron_idx)].add(layer_input)
        else:
            new_input_residuals = neuron_states.input_residuals 
            new_input_activity = neuron_states.input_activity
            new_layer_activity = neuron_states.layer_activity
            new_weight_res = neuron_states.weight_res

        new_neuron_states = ConvNeuronStates(neuron_state=
                                            NeuronStates(values=new_values,
                                                thresholds=neuron_states.thresholds,
                                                input_residuals=new_input_residuals,
                                                output_residuals=neuron_states.output_residuals,
                                                input_order=neuron_states.input_order,
                                                input_activity=new_input_activity,
                                                layer_activity=new_layer_activity,
                                                output_activity=new_output_activity,
                                                last_sent_iteration=neuron_states.last_sent_iteration,
                                                input_vector=neuron_states.input_vector,
                                                output_vector=neuron_states.output_vector,
                                                values_history=neuron_states.values_history,
                                                history_index=neuron_states.history_index,
                                                weights_shape=neuron_states.weights_shape,
                                                is_conv=neuron_states.is_conv,
                                            ),
                                            weight_res=new_weight_res,
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
        # jax.debug.print("rank {} iteration {} nb valid elements {}, activated output {}", rank, iteration, nb_valid_elements, out_events[0])

        return nb_valid_elements, out_events, new_neuron_states

    @jit
    def last_input(neuron_states):
        if params.sync_rate == 1:
            C, H, W = params.flat_layer_sizes[layer_idx]
            return jnp.array(0), jnp.zeros((C*H*W, 4)), neuron_states

        # For full sync case, fire all neurons that are above the threshold  
        neuron_val = neuron_states.values
        activated_output = activation_func(neuron_states.thresholds, neuron_val)  
        
        # Step 4: Compute remaining values and update the neuron state
        remaining_value = neuron_val - activated_output
        nb_valid_elements, out_events, unpooled = full_matrix_to_event_array_with_pooling(activated_output, activated_output.shape, 
                                                                                          neuron_states.pooling, neuron_states.pool_size, 
                                                                                          neuron_states.pool_stride, rank)
        # jax.debug.print("valid el {}, out events {}", nb_valid_elements, jnp.count_nonzero(activated_output))
        # jax.debug.print("{}", iteration)
        # jax.debug.print("out shape {}", out_events.shape)
        
        # Add unpooled values to layer activity
        mask = unpooled != 0
        new_layer_activity = jnp.where(
            mask,
            neuron_states.layer_activity + unpooled,
            neuron_states.layer_activity
        )      
        # new_layer_activity = neuron_states.layer_activity
        # jax.debug.print("rank {} unpooled {}, mask {}, old layer activity {}, new layer activity {}, iteration {}, neuron idx {}", rank, jnp.count_nonzero(unpooled), jnp.count_nonzero(mask), jnp.count_nonzero(neuron_states.layer_activity), jnp.count_nonzero(new_layer_activity), iteration, neuron_idx)

        # jax.debug.print("rank {}, valid el {}, out events {}, unpooled {} NEW LAYER ACTIVITY {}", 
        #                 rank, nb_valid_elements, out_events.shape, unpooled, new_layer_activity)
        
        new_neuron_states = ConvNeuronStates(neuron_state=
                                            NeuronStates(values=remaining_value,
                                                thresholds=neuron_states.thresholds,
                                                input_residuals=neuron_states.input_residuals,
                                                output_residuals=neuron_states.output_residuals,
                                                input_order=neuron_states.input_order,
                                                input_activity=jnp.ones(neuron_states.input_activity.shape, dtype=int),
                                                layer_activity=new_layer_activity,
                                                output_activity=neuron_states.output_activity,
                                                last_sent_iteration=neuron_states.last_sent_iteration,
                                                input_vector=neuron_states.input_vector,
                                                output_vector=neuron_states.output_vector,
                                                values_history=neuron_states.values_history,
                                                history_index=neuron_states.history_index,
                                                weights_shape=neuron_states.weights_shape,
                                                is_conv=neuron_states.is_conv,
                                            ),
                                            weight_res=neuron_states.weight_res,
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
@partial(jax.jit, static_argnames=['params', 'layer_computation', 'grad',])
def conv_predict(params, key, weights, empty_neuron_states, layer_computation, batch_data: jnp.ndarray, grad=False):
    '''
    CNN inference, each layer sends each event separately in the format: (c, x, y, value)
    -1 means end of data from previous layer
    -2 means placeholder data in the input layer 
    '''
    rcv_size = 4
    @jit
    def input_layer(args):
        neuron_states, x = args # x binned is shape (timesteps, channel, height, width)
                                # x not binned is shape (max_nonzero, 4) (x, y, t, c)
        # jax.debug.print("Rank {}, input layer shape: {}", rank, x.shape)
        
        x_p = x
        # x_p = jnp.ones((50, rcv_size))
        @jit
        def send_input(i, carry):
            timestep = carry
            data = x_p[i]
            @jit
            def send_data(t):
                combined = data
                send(combined, dest=rank+process_per_layer, tag=0, comm=comm)
                return t+1
            
            timestep = jax.lax.cond(
                jnp.all(data != -2),
                send_data,
                lambda _: timestep,
                operand=timestep
            )
            return timestep

        # Initial carry: (timestep=0)
        iteration = jax.lax.fori_loop(0, x_p.shape[0], send_input, (0))
        #________________________________________________________________________________
        # x_p = jnp.array(x)
        # def send_input(i, carry):
        #     count = carry
        #     data = x_p[i]
        #     send(data, dest=rank+process_per_layer, tag=0, comm=comm)
        #     return i

        # def first_not_minus2(row):
        #     return (row != -2)
        # mask = jax.vmap(first_not_minus2)(x_p)
        # loop_iterations = (jnp.count_nonzero(mask)/2).astype(int)

        # iteration = jax.lax.fori_loop(0, loop_iterations, send_input, (0))

        # Send end signal
        send(jnp.array([-1.0, -1.0, -1.0, -1.0]), dest=rank+process_per_layer, tag=0, comm=comm)

        return neuron_states, iteration
    @jit
    def other_layers(args):
        neuron_states, _ = args
        def cond(state): # end of input has been reached -> break the while loop
            _, neuron_idx, _, _= state
            return jnp.all(neuron_idx != -1)
        @jit
        def forward_pass(state):
            neuron_states, _, timestep, iteration = state
            @jit
            def hidden_layers(args): # Send activation to the next layers
                loop_iterations, activated_output = args
                @jit
                def send_activation(i, _):
                    combined = activated_output[i]
                    
                    jax.lax.cond(
                        (combined[3] != 0), # Only send relevant data #TODO Check if this conditional is still needed
                        lambda _: send(combined, dest=rank+process_per_layer, tag=0, comm=comm),
                        lambda _: [],
                        operand=None
                    )
                    return None

                jax.lax.fori_loop(0, loop_iterations, send_activation, None)
                return None
            
            # Receive neuron values from previous layers and compute the activations
            input_data = recv(jnp.zeros(rcv_size), source=rank-process_per_layer, tag=0, comm=comm)
            # Unpack
            neuron_idx = input_data[:3].astype(jnp.int32) # channel, x, y
            layer_input = input_data[3] # value
            # if rank != 1:
            #     jax.debug.print("rank {} received neuron idx {} and value {} at iteration {}", rank, neuron_idx, layer_input, iteration)
            
            loop_iterations, activated_output, new_neuron_states = layer_computation(params, key, neuron_idx, layer_input, weights, neuron_states, iteration, grad)
            # jax.debug.print("rank {}, loop iterations: {}, activated output shape: {}", rank , loop_iterations, activated_output.shape)
            # loop_iterations, activated_output = 1, jnp.ones((15, rcv_size))
            # if rank == 3:
            #     loop_iterations = 4
            # if rank == 4:
            #     # loop_iterations = 2
            #     loop_iterations = jax.lax.cond(iteration%4==0, lambda _: 1, lambda _: 0, None)

            neuron_states = new_neuron_states
            # if rank == 9:
            #     jax.debug.print("rank {} receving neuron idx {} and value {} at iteration {} and sending {} events from {}", rank, neuron_idx, layer_input, iteration, loop_iterations, activated_output)
            # jax.lax.cond(loop_iterations != 0, lambda _: jax.debug.print("rank {} sending {} iterations \n{}", rank, loop_iterations, activated_output),
            #                 lambda _: None, None)
            jax.lax.cond(layer_idx == last_layer, lambda _: None, hidden_layers, (loop_iterations, activated_output)) # Don't send if we reach the last layer
            return neuron_states, neuron_idx, timestep, iteration+1
        
        neuron_idx, timestep, iteration =  jnp.zeros(3).astype(jnp.int32), 0, 0
        initial_state = (neuron_states, neuron_idx, timestep, iteration)
        
        # Loop until the rank receives a -1 timestep
        neuron_states, neuron_idx, timestep, iteration = jax.lax.while_loop(cond, forward_pass, initial_state)
        
        # Send -1 to the next rank when all incoming data has been processed
        jax.lax.cond(
            layer_idx != last_layer,
            lambda _: send(jnp.array([-1.0, -1.0, -1.0, -1.0]), dest=rank + process_per_layer, tag=0, comm=comm),
            lambda _: [],
            operand=None
        )
        return neuron_states, iteration-1
    
    # Loop over batches, accumulate output values and return them
    @jit
    def loop_over_batches(_, x):
        neuron_states = empty_neuron_states  
        new_neuron_states, iterations = jax.lax.cond(layer_idx==0, input_layer, other_layers, (neuron_states, x))
        
        return None, (new_neuron_states.values, iterations, new_neuron_states)
    _, (all_outputs, all_iterations, all_neuron_states) = jax.lax.scan(loop_over_batches, None, batch_data)    
    
    # Synchronize all ranks before starting the backward pass
    mpi4jax.barrier(comm=comm)
    # jax.debug.print("Rank {} finished forward pass", rank)
    return all_outputs, all_iterations, all_neuron_states


#region Training helpers
@partial(jax.jit, static_argnames=['params', 'layer_computation', 'conv_layer_sizes'])
def predict_bwd(params, key, conv_layer_sizes, weights, empty_neuron_states, layer_computation, batch_data):
    '''
    B: batch_size
    '''
    all_outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, batch_data, grad=True)
    w_sum = l2_weight_regularization(mpi_config, weights)

    # Receive the gradients from the later layers
    next_grad = recv(jnp.zeros((batch_part,) + params.flat_layer_sizes[layer_idx]), source=rank + process_per_layer, tag=2, comm=comm) # Shape: (B, 128)

    # Compute input's gradient and weight gradient
    weight_grad, th_grad, weight_res = MLP_back_prop(params, all_neuron_states, next_grad, layer_idx)
    weight_grad += 2 * params.w_reg * weights

    if layer_idx > 1:
        # Send gradient to the previous layer
        send_grad = jnp.dot(next_grad, weights.T) # Shape: (B, 128) @ (128, 784) = (B, 784)
        send_grad *= (~jnp.all(weight_res == 0, axis=2)) 
        send(send_grad, dest=rank-process_per_layer, tag=2, comm=comm)
    
    # Sparsity loss gradients 
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
    
    scaling = jax.lax.cond(params.sparsity_impact[layer_idx] > 0,
                           lambda _: params.sparsity_impact[layer_idx] / (all_iterations * batch_part * process_per_layer) ,
                           lambda _: 0.0,
                           None)
    
    input_activity = jnp.sum(all_neuron_states.input_activity, axis=0) # Shape (784)
    layer_activity = jnp.sum(all_neuron_states.layer_activity, axis=0) # Shape (128)
    
    layer_activity = gather_batch(layer_activity, mpi_config, average=False) # Gather the weight gradients from all ranks in the same layer
    input_activity = gather_batch(input_activity, mpi_config, average=False)
    
    sparsity_residuals = scaling * layer_activity # Shape: (128,)
    # jax.debug.print("Rank {}, scaling mean: {}, sparsity_residuals mean: {}, sparsity_residuals sum: {}", rank, scaling, jnp.mean(sparsity_residuals), jnp.sum(sparsity_residuals))
    
    th_sparsity_grad = -sparsity_residuals
    weight_sparsity_grad = jnp.outer(input_activity, sparsity_residuals) # Shape: (784, 128)
    # jax.debug.print("Rank {}, th_sparsity_grad: {}, weight_sparsity_grad: {}", rank, jnp.mean(th_sparsity_grad), np.mean(weight_sparsity_grad))
    
    return all_outputs, iterations, all_neuron_states, (weight_grad, th_grad, weight_sparsity_grad, th_sparsity_grad) 

@partial(jax.jit, static_argnames=['params', 'layer_computation', 'conv_layer_sizes'])
def conv_predict_bwd(params, key, conv_layer_sizes, weights, empty_neuron_states, layer_computation, batch_data):
    all_outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, batch_data, grad=True)
    w_sum = l2_weight_regularization(mpi_config, weights)

    # Receive the gradients from the later layers
    out_layer_shape = params.flat_layer_sizes[layer_idx]
    next_grad_1 = recv(jnp.zeros((batch_part,) + out_layer_shape), source=rank + process_per_layer, tag=2, comm=comm) # Shape: (B, 5, 28, 28) // (B, 5, 14, 14) with pooling

    if empty_neuron_states.pooling != "":
        # Upsampling next_grad: from (B, 5, 14, 14) to (B, 5, 28, 28) by copying each element sh*sw number of times
        sh, sw = empty_neuron_states.pool_size
        next_grad = jnp.repeat(jnp.repeat(next_grad_1, sh, axis=2), sw, axis=3)

        # TODO: For non symetric pooling => Pad to the correct size 17->8 : 8->17
        # Commented implementation only works when the convolutions preserve the size of the feature maps
        # target_size_h, target_size_w = params.flat_layer_sizes[layer_idx-1][1], params.flat_layer_sizes[layer_idx-1][2]
        # pad_h = target_size_h - next_grad.shape[2]
        # pad_w = target_size_w - next_grad.shape[3]
        # next_grad = jnp.pad(next_grad, ((0, 0), (0, 0), (0, max(0, pad_h)), (0, max(0, pad_w))))
        
        # jax.debug.print("rank {} target next grad shape: {} {} ", rank, target_size_h, target_size_w)
        # jax.debug.print("rank {} upsampled weight grad: {}", rank, next_grad.shape)
    else:
        next_grad = next_grad_1
    
    # Mask the cells that never fired 
    activity_mask = jnp.where(all_neuron_states.layer_activity > 0, 1.0, 0.0)
    next_grad = activity_mask * next_grad
    # jax.lax.cond(   jnp.all(jnp.isfinite(next_grad)),
    #                 lambda _: None,
    #                 lambda _: jax.debug.print("NaN or Inf detected in next grad {}!", next_grad),
    #                 operand=None,)    
    # next_grad = all_neuron_states.layer_activity * next_grad 

    pad_x, pad_y = params.layer_sizes[layer_idx][2]
    strides = params.layer_sizes[layer_idx][3]
    @jit
    def grad_w(x, dy):
        # x: (3, 28, 28)
        # dy: (5, 28, 28)
        lhs = x[:, None, :, :]       # (3, 1, 28, 28)
        rhs = dy[:, None, :, :]      # (5, 1, 28, 28)
        
        grad = jax.lax.conv_general_dilated(
            lhs, rhs,
            window_strides=strides,
            padding=((pad_x, pad_x), (pad_y, pad_y)),
            dimension_numbers=('NCHW', 'OIHW', 'CNHW') # "CNHW" = (5, 3, 3, 3) -- "NCHW" = (3, 5, 3, 3)
        )
        return grad # (5, 3, 3, 3)    
        
    input_residuals = all_neuron_states.input_residuals # Shape: (B, 3, 28, 28)
    weight_grad = jax.vmap(grad_w)(input_residuals, next_grad) # Shape: (5, 3, 3, 3) #TODO Use batches in convolution directly instead of vmap?

    # weight_grad = weight_grad * weight_res
    weight_grad += 2 * params.w_reg * weights
    
    @jit
    def grad_x(dY, W):
        # dY: (5, 28, 28)
        # W: (out, in, k_h, k_w)
        # Full convolution in the backward pass, source: https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c

        # Flip kernel spatially
        W_flipped = jnp.flip(W, axis=(2, 3)) 
        rhs = W_flipped.transpose(1, 0, 2, 3) # shape (in, out, k_h, k_w)
        
        # Compute full convolution padding (kernel_size - 1 - forward padding)
        k_H, k_W = W.shape[2], W.shape[3]
        pad_h = k_H - 1 - pad_x
        pad_w = k_W - 1 - pad_y
        manual_padding = ((pad_h, pad_h), (pad_w, pad_w))
                
        return jax.lax.conv_general_dilated(
            lhs=dY,                 # gradient from next layer
            rhs=rhs,                # flipped weights
            window_strides=(1,1),
            padding=manual_padding,        
            lhs_dilation=strides,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )

    # Compute threshold gradients
    layer_activity = jnp.where(all_neuron_states.layer_activity > 0, 1, 0)
    # jax.debug.print("RANK {} has next grad shape: {} layer_activity shape: {}", rank, next_grad.shape, layer_activity.shape)
    
    th_grad = -jnp.mean(next_grad * layer_activity, axis=0)
    thresholds = all_neuron_states.thresholds[0] # The whole batch has the same thresholds
    th_grad = th_grad * thresholds * (thresholds - 1)
    # jax.debug.print("th grad {} {} {}", all_neuron_states.values.shape, jnp.count_nonzero(th_grad), (th_grad.shape))
    # th_grad = jnp.zeros(all_neuron_states.values.shape)

    if layer_idx > 1:
        # Send gradient to the previous layer
        send_grad = (grad_x)(next_grad, weights) #TODO Apply the masking here like for linear backprop?
        send(send_grad, dest=rank-process_per_layer, tag=2,comm=comm)

     # Sparsity loss gradients 
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
    
    scaling = jax.lax.cond(params.sparsity_impact[layer_idx] > 0,
                           lambda _: params.sparsity_impact[layer_idx] / (all_iterations * batch_part * process_per_layer) ,
                           lambda _: 0.0,
                           None)
    
    input_activity = jnp.sum(all_neuron_states.input_activity, axis=0) # Shape (1, 28, 28)
    layer_activity = jnp.sum(all_neuron_states.layer_activity, axis=0) # Shape (3, 28, 28)
    
    layer_activity = gather_batch(layer_activity, mpi_config, average=False) 
    input_activity = gather_batch(input_activity, mpi_config, average=False)
    
    sparsity_residuals = scaling * layer_activity # Shape: (128,)
    # jax.debug.print("Rank {}, scaling mean: {}, sparsity_residuals: {}, " \
    # "layer_activity {} input_activity: {}, ", rank, scaling, jnp.mean(sparsity_residuals), jnp.mean(layer_activity), jnp.mean(input_activity))
    
    th_sparsity_grad = -sparsity_residuals
    weight_sparsity_grad = grad_w(input_activity.astype(jnp.float32), sparsity_residuals) # Shape: (784, 128)
    # jax.debug.print("rank {}, sparsity weight grad {} {}", rank, weight_sparsity_grad.shape, weights.shape)
    # weight_sparsity_grad = jnp.zeros_like(weights)

    return all_outputs, iterations, all_neuron_states, (weight_grad, th_grad, weight_sparsity_grad, th_sparsity_grad) 

# Define the loss function
@partial(jax.jit, static_argnames=['params', 'layer_computation',])
def loss_fn(params, key, weights, empty_neuron_states, layer_computation, target, batch_data):
    all_outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, batch_data, grad=True)
    w_sum = l2_weight_regularization(mpi_config, weights)

    # Compute Loss and loss gradient
    loss, loss_grad = jax.value_and_grad(mean_loss)(all_outputs, target)
    loss_grad /= process_per_layer # Shape (B, 10)
    loss += params.w_reg * w_sum

    # Compute output gradient and weight gradient
    out_grad, weight_grad = jax.vmap(loss_bpp, in_axes=(None, 0, 0))(weights, all_neuron_states, loss_grad) # Shape (B, 128), (B, 128, 10)
    mean_weight_grad = jnp.mean(weight_grad, axis=0) # Shape: (128, 10)
    mean_weight_grad += 2 * params.w_reg * weights
    mean_weight_grad = jnp.expand_dims(mean_weight_grad, axis=0)  # Shape: (1, 128, 10)

    # Send gradient to previous layers                
    send(out_grad, dest=rank-process_per_layer, tag=2,comm=comm)
    
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)

    total_loss = loss + sparsity_L 

    acc_history, avg_rank = None, None
    if params.history_size > 0:
        # One-hot target → scalar class index
        target_labels = jnp.argmax(target, axis=-1)
        acc_history, avg_rank = process_history(all_neuron_states.values_history, all_neuron_states.history_index, target_labels)

    return (loss, all_outputs, iterations, total_loss, (acc_history, avg_rank)), (mean_weight_grad, loss_grad)

def sparsity_loss(params, all_neuron_states, iterations):
    '''
    Compute the sparsity loss based on the input residuals and the number of iterations
    '''
    if params.sparsity_impact[layer_idx] <= 0.0:
        return 0, 1, 0
    
    # Gather all the activations at the last layer to compute the sparsity loss
    leader_rank = layer_idx * process_per_layer
    # jax.debug.print("Rank {}, activations shape: {}, iterations shape: {}, last layer neuron values: {}", rank, (activations.shape), jnp.mean(iterations.shape), jnp.sum(all_neuron_states.values))

    activations = gather_batch(all_neuron_states.input_residuals, mpi_config, average=False) # Gather the weight gradients from all ranks in the same layer
    iterations = gather_batch(iterations, mpi_config, average=True) # Gather the iterations from all ranks in the same layer
    # jax.debug.print("Rank {}, activations shape: {}, iterations shape: {}, last layer neuron values: {}", rank, (activations.shape), jnp.mean(iterations.shape), jnp.sum(all_neuron_states.values))

    all_iterations = 0.0
    all_activations = 0.0
    sparsity_L = 0.0
    if layer_idx != last_layer and rank == leader_rank:
        # jax.debug.print("Rank {}, sending activations {} and iterations {} to the last rank", rank, jnp.sum(activations), jnp.mean(iterations))
        send(jnp.sum(activations), dest=last_layer * process_per_layer, tag=6,comm=comm)
        if rank == 0:
            send(jnp.mean(iterations), dest=last_layer * process_per_layer, tag=6,comm=comm)
    elif layer_idx == last_layer and rank == leader_rank:
        for i in range(last_layer):
            # Storing the thresholds
            act_sum = recv(jnp.zeros(1), source=i * process_per_layer, tag=6, comm=comm)
            all_activations = all_activations + act_sum[0] # Sum of all activations in the hidden layers
            
            if i == 0: # Get iterations of input data
                iter_mean = recv(jnp.zeros(1), source=i * process_per_layer, tag=6, comm=comm)
                all_iterations = iter_mean[0]
        all_activations += jnp.sum(activations) # Adding the activations of the last layer

        sparsity_L = all_activations /  (all_iterations * batch_part * process_per_layer)
        # jax.debug.print("Rank {}, sparsity L: {}, all iterations: {}, all activations: {}", rank, sparsity_L, all_iterations, all_activations)
    all_iterations = bcast(all_iterations, root=last_layer*process_per_layer, comm=comm)

    return all_activations, all_iterations, sparsity_L

# region TRAINING
def train(params: Params, key, total_batches, network, weights, empty_neuron_states, layer_computation, opti, trial=None):     
    """
    MPI SEND/RECEIVE tag list:
    tag 0:  forward computation, data format: (previous_layer_neuron_index, neuron_value)
            end of input is encoded with the index -1
    tag 2: backward computation, last layer gradient shape: (layer_sizes[-1], 1)
    tag 3: weight residuals, shape: (layer_sizes[rank], layer_sizes[rank+1])
    tag 4: communication between processes to split the data
    tag 5: weights for storing
    tag 6: activations for sparsity loss
    tag 7: compute weight regularization
    tag 10: data labels(y)
    tag 20: communications for gathering, sharing and averaging data across ranks in the same layer
    """
    global training_generator
    global validation_generator
    global test_generator
        
    # Initialize the lists for storing the intermediate values
    if layer_idx == last_layer:
        all_epoch_accuracies = []
        all_validation_accuracies = []
        all_loss = []
        all_history = []
    all_mean_iterations = []
    
    # Initialize the optimizer
    if rank == 0:
        print(f"{opti} optimizer selected")
    if opti == "adam":
        solver = optax.adam(learning_rate=params.learning_rate)
    elif opti == "adamw":        
        solver = optax.adam(learning_rate=params.learning_rate)
    elif opti == "sgd":
        solver = optax.sgd(learning_rate=params.learning_rate)
    elif opti == "rmsprop":
        solver = optax.rmsprop(learning_rate=params.learning_rate, decay=0.9, eps=1e-8)
        print("amsgrad optimizer selected")
        solver = optax.amsgrad(learning_rate=params.learning_rate)
    elif opti == "lion":
        solver = optax.lion(learning_rate=params.learning_rate)
    else: 
        solver = None
    if solver is not None:
        opt_state = solver.init(weights)
    
    th_solver = optax.adam(learning_rate=params.threshold_lr)
    th_opt_state = th_solver.init(empty_neuron_states.thresholds)
    
    # Synchronize all ranks and start timer
    mpi4jax.barrier(comm=comm)
    start_time = time.time()

    for epoch in tqdm(range(params.num_epochs), disable=TQDM_DISABLE):
        key, subkey = jax.random.split(key) 

        if layer_idx == last_layer:
            epoch_correct = 0
            epoch_total = 0
            epoch_loss = []
            
        epoch_iterations = []
        if layer_idx == 0:
            batch_iterator = None
            if rank == 0:
                batch_iterator = iter(training_generator) # Make the dataloader iterable
            
        for i in tqdm(range(total_batches[0]), disable=TQDM_DISABLE):
            neuron_states = empty_neuron_states
            if layer_idx == 0: # Input layer
                batch_x, batch_y = split_batch(params, batch_iterator, mpi_config, 4)

                send(batch_y, dest=last_layer * process_per_layer + rank, tag=10,comm=comm) # Send the labels to the output layer

                # Run the forward pass
                outputs, iterations, all_neuron_states = (conv_predict)(params, subkey, weights, neuron_states, layer_computation, jnp.array(batch_x))
                all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
            else:
                if layer_idx==last_layer: # Output layer
                    # Receive the labels from the input layer
                    y = recv(jnp.zeros((batch_part,)), source=rank - (last_layer * process_per_layer), tag=10, comm=comm)  # Source rank opposite operation: rank - (last_layer * process_per_layer)
                    y_encoded = jnp.array(one_hot_encode(y, num_classes=params.layer_sizes[-1][0]))

                    # Run the forward and backward pass for the output layer
                    (loss, outputs, iterations, total_loss, history), gradients = (loss_fn)(params, subkey, weights, neuron_states, layer_computation, y_encoded, jnp.zeros((batch_part, 1, 4)))

                    weight_grad = gradients[0]
                    # weight_grad = gather_batch(weight_grad, mpi_config, average=True)
                    weight_grad = combine_batch_avg(weight_grad, mpi_config) # Gather the weight gradients from all ranks in the same layer
                
                    # Store the accuracy, loss and history                    
                    valid_y, batch_correct = accuracy(i, outputs, y, iterations, print=False)                 
                    # print(f"Batch {i}, Accuracy: {batch_correct}/{valid_y.shape[0]} ")
                    epoch_correct += batch_correct
                    epoch_total += valid_y.shape[0]
                    
                    epoch_loss.append(loss)
                    if params.history_size > 0:
                        all_history.append(history)
                else:
                    # Select the correct backward pass function
                    bwd_fn = predict_bwd
                    if empty_neuron_states.is_conv:
                        bwd_fn = conv_predict_bwd

                    # Run the forward and backward pass for the hidden layers
                    outputs, iterations, all_neuron_states, grads = (bwd_fn)(params, subkey, network.conv_layer_sizes, weights, neuron_states, layer_computation, jnp.zeros((batch_part, 1, 4)))
                    weight_grad, threshold_grad, weight_sparsity_grad, threshold_sparsity_grad = grads
                    
                    threshold_grad = gather_batch(threshold_grad, mpi_config, average=True) # Gather the thresholds' gradients from all ranks in the same layer

                    # weight_grad = gather_batch(weight_grad, mpi_config, average=True)
                    weight_grad = combine_batch_avg(weight_grad, mpi_config) # Gather the weight gradients from all ranks in the same layer
                    
                    # Add sparsity loss' impact to the gradient if relevant
                    if params.sparsity_impact[layer_idx] > 0:
                        weight_grad = weight_grad + weight_sparsity_grad
                        threshold_grad = threshold_grad + threshold_sparsity_grad
                    
                    # Update thresholds                    
                    if params.threshold_lr != 0:
                        th_updates, th_opt_state = solver.update(threshold_grad, th_opt_state, empty_neuron_states.thresholds)
                        empty_neuron_states.thresholds = jax.nn.sigmoid(optax.apply_updates(empty_neuron_states.thresholds, th_updates)) # TODO Check if this still works
      
                # Update weights
                if solver is not None:
                    # Optax optimizer
                    updates, opt_state = solver.update(weight_grad, opt_state, weights)
                    weights = optax.apply_updates(weights, updates)
                else:                
                    # Basic GD
                    weights -= params.learning_rate * weight_grad 
            # if i > 3: # Run a few epochs for testing
            #     break
            # return
            epoch_iterations.append(iterations[iterations > 1])
        
        # Compute the average iterations for each layer
        epoch_iterations = jnp.concatenate(epoch_iterations)
        mean = jnp.mean(epoch_iterations)
        all_mean_iterations.append(mean)
        all_mean_iterations = gather_batch(all_mean_iterations, mpi_config)
        all_mean_iterations = all_mean_iterations.tolist()
        
        if layer_idx != 0:
            jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points and a mean threshold of {}", rank, mean, epoch_iterations.shape[0], jnp.mean(empty_neuron_states.thresholds))
        
        # Inference on the validation set
        val_accuracy, val_mean, _ = batch_predict(params, key, total_batches, network, weights, empty_neuron_states, layer_computation, dataset="val", save=False, debug=False)

        epoch_accuracy = 0.0
        if layer_idx == last_layer:
            # Store loss values
            mean_loss = jnp.mean(jnp.array(epoch_loss))
            all_loss.append(mean_loss)
            mean_loss = gather_batch(mean_loss, mpi_config)

            # Store training and validation accuracies
            epoch_accuracy = epoch_correct / epoch_total
            all_epoch_accuracies.append(epoch_accuracy)
            all_validation_accuracies.append(val_accuracy)
            all_epoch_accuracies = gather_batch(all_epoch_accuracies, mpi_config)
            all_validation_accuracies = gather_batch(all_validation_accuracies, mpi_config)
            all_epoch_accuracies, all_validation_accuracies = all_epoch_accuracies.tolist(), all_validation_accuracies.tolist()
            if rank == size-1:
                jax.debug.print("Epoch {} , Training Accuracy: {:.2f}%, Validation Accuracy: {:.2f}%, mean loss: {}, mean val iterations: {}", epoch, all_epoch_accuracies[-1] * 100, val_accuracy * 100, mean_loss, val_mean)
                jax.debug.print("----------------------------\n")
        epoch_accuracy = bcast(epoch_accuracy, root=size-1, comm=comm)
        if epoch_accuracy >= 0.9999:
            break
        if STORE_EACH_EPOCH: 
            # Gather the weights and iteration values at the last layer
            layer_weights_sizes = [] # Retrieve the shapes of all the weights 
            for layer in network.layers:
                layer_weights_sizes.append(layer.weights_shape)
            weights_dict, all_iteration_mean, thresholds_dict = gather_w_iter_th(network, layer_weights_sizes, weights, jnp.array(all_mean_iterations), empty_neuron_states.thresholds)

            if rank == last_layer * process_per_layer:
                result_path_str = store_training_data(
                                size,
                                network, 
                                "train",
                                all_epoch_accuracies, 
                                all_validation_accuracies, 
                                -1.0,
                                time.time() - start_time,
                                all_iteration_mean,
                                weights_dict,
                                all_loss, 
                                thresholds_dict,
                                opti,
                                "CNN_temp",
                                all_history,
                                total_batches[0])

    # Inference on the test set
    test_accuracy, test_mean, _ = batch_predict(params, key, total_batches, network, weights, empty_neuron_states, layer_computation, dataset="test", save=False, debug=False)
    
    # Gather the weights and iteration values at the last layer
    layer_weights_sizes = [] # Retrieve the shapes of all the weights 
    for layer in network.layers:
        layer_weights_sizes.append(layer.weights_shape)
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_iter_th(network, layer_weights_sizes, weights, jnp.array(all_mean_iterations), empty_neuron_states.thresholds)
    
    # Synchronize all MPI processes again
    mpi4jax.barrier(comm=comm)
    end_time = time.time()
    
    # Compute processing time and store all the results in a json file
    MAX_LEN = 256
    result_path = jnp.zeros(MAX_LEN, dtype=jnp.uint8)
    if rank == last_layer * process_per_layer:
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
                            "CNN",
                            all_history,
                            total_batches[0])
        
        encoded = np.frombuffer(result_path_str.encode("utf-8"), dtype=np.uint8)
        if encoded.size > MAX_LEN:
            raise ValueError("result_path too long")
        padded = np.pad(encoded, (0, MAX_LEN - encoded.size), constant_values=0)
        result_path = jnp.array(padded)
    result_path = bcast(result_path, root=last_layer*process_per_layer, comm=comm)
    result_path = bytes(result_path).decode("utf-8").rstrip("\x00")

    return result_path

def gather_w_iter_th(network, layer_weights_sizes, weights, mean_iterations, thresholds):
    # Gather all the weights and iteration values at the last layer to store them
    leader_rank = layer_idx * process_per_layer

    weights_dict = {}
    all_iteration_mean = []
    thresholds_dict = {}
    if layer_idx != last_layer and rank == leader_rank:
        send(mean_iterations, dest=last_layer * process_per_layer, tag=5,comm=comm)
        if layer_idx != 0:
            send(weights, dest=last_layer * process_per_layer, tag=5,comm=comm)
            send(thresholds, dest=last_layer * process_per_layer, tag=5,comm=comm)
    elif layer_idx == last_layer and rank == leader_rank:
        for i in range(last_layer):
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
        weights_dict[f"layer_{last_layer}"] = weights.tolist()
        all_iteration_mean = all_iteration_mean[1:] # Don't keep the value of the input layer
        print("all iteration mean: rank", rank, all_iteration_mean)

    return weights_dict, all_iteration_mean, thresholds_dict


# region Inference loop
def batch_predict(params, key, total_batches, network, weights, empty_neuron_states, layer_computation, dataset:str="train", save=True, debug=True, readInputJson=False):    
    global training_generator
    global validation_generator
    global test_generator    
    
    mpi4jax.barrier(comm=comm)
    start_time = time.time()
    
    if dataset == "train":
        total_batches = total_batches[0]
        if layer_idx == 0:
            batch_iterator = None
            if rank == 0:
                print(f"Inference on the training set...")
                batch_iterator = iter(training_generator)
    elif dataset == "val":
        total_batches = total_batches[1]
        if layer_idx == 0:
            batch_iterator = None
            if rank == 0:
                print(f"Inference on the validation set...")
                batch_iterator = iter(validation_generator)
    elif dataset == "test":
        total_batches = total_batches[2]
        if layer_idx == 0:
            batch_iterator = None
            if rank == 0:
                print(f"Inference on the test set...")
                batch_iterator = iter(test_generator)
    else:
        print("INVALID DATASET")
        return
    if total_batches == 0:
        return -0.01, -1.0, -1.0 # arbitrary code for empty dataset

    if layer_idx == last_layer:
        epoch_correct = 0
        epoch_total = 0
        all_history = []

    epoch_iterations = []
    for i in tqdm(range(total_batches), disable=TQDM_DISABLE):
        if layer_idx == 0:         
            # readInputJson = True        
            if readInputJson: # Test with stored input
                folder_add = "14"
                with open(f'pretrained_data/CNN/pretrained_data{folder_add}/{len(params.layer_sizes)}hidden_single_input.json') as f:
                    batch_x = np.array(json.load(f)).squeeze() 
                    batch_x = jnp.expand_dims(batch_x[0], axis=0)
                # with open(f'{len(params.layer_sizes)}hidden_single_input.json') as f:
                #     batch_x = np.array(json.load(f)).squeeze() 
                #     batch_x = jnp.expand_dims(batch_x, axis=0)
                with open(f'pretrained_data/CNN/pretrained_data{folder_add}/{len(params.layer_sizes)}hidden_single_output.json') as f:
                    batch_y = np.array(json.load(f)["labels"]).squeeze()
                    batch_y = jnp.expand_dims(batch_y[0], axis=0)
            else:
                batch_x, batch_y = split_batch(params, batch_iterator, mpi_config, 4)
            # store_data_to_json(f"{len(params.layer_sizes)}hidden_single_input.json", batch_x.tolist()) # Store for hardware usage
            
            # batch_x, batch_y = jnp.array([(0.0,1.0,1.0, 1.0), (0.0,2.0,2.0, 2.0), (0.0, 1.0, 0.0, 3.0), (0.0, 4.0, 4.0, 4.0), (0.0, 3.0, 3.0, 5.0), (-2, -2, -2 ,-2)]), jnp.array([1])
            # batch_x = jnp.expand_dims(batch_x, axis=0)
            # print("batch x shape, batch y shape", batch_x.shape, batch_y.shape)

            outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, jnp.array(batch_x))

            # Send label to the last layer
            send(batch_y, dest=last_layer * process_per_layer + rank, tag=10,comm=comm)
        else:
            # outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, jnp.zeros((batch_part, layer_sizes[0])))
            batch_data = jnp.zeros((batch_part, 1, 4))

            outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, batch_data)
            # jax.debug.print("Rank {} All neuron states values shape: {}, output shape : {}", rank, all_neuron_states.values.shape, outputs)

            if layer_idx == last_layer:
                # jax.debug.print("Rank {} All neuron states values shape: {}, output shape : {}", rank, all_neuron_states.values.shape, outputs)

                y = recv(jnp.zeros((batch_part,)), source=rank - (last_layer * process_per_layer), tag=10, comm=comm)   
                
                valid_y, batch_correct = accuracy(i, outputs, y, iterations, print=False)                 
                
                epoch_correct += batch_correct
                epoch_total += valid_y.shape[0]
                # store_data_to_json(f"{len(params.layer_sizes)}hidden_single_output.json", outputs.tolist(), y.tolist())

                if params.history_size > 0:
                    # One-hot target → scalar class index
                    history = process_history(all_neuron_states.values_history, all_neuron_states.history_index, y)
                    all_history.append(history)
            # store_data_to_json(f"{len(params.layer_sizes)}hidden_intermediates_layer{rank}.json", outputs.tolist())

        # store_data_to_json(f"{len(params.layer_sizes)}hidden_iterations_layer{rank}.json", iterations.tolist())
        epoch_iterations.append(iterations[iterations > 1])
        # jax.debug.print("Rank {}, iterations: {}", rank, iterations)
        # if i >= 0:
        #     break
    
    # print(f"Shape iterations before flattening: {jnp.array(epoch_iterations).shape}")
    epoch_iterations = jnp.concatenate(epoch_iterations)
    mean = jnp.mean(epoch_iterations)
    # print(f"Rank {rank} finished epoch with mean {mean} with {epoch_iterations.shape} iterations")

    if layer_idx != 0:
        mean = gather_batch(mean, mpi_config)
    # jax.debug.print("Rank {}, all iterations shape: {}", rank, (epoch_iterations.shape[0]))
    
    if rank != 0 and debug:
        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, epoch_iterations.shape[0]*process_per_layer)
    
    epoch_accuracy = -1.0
    if layer_idx == last_layer:
        epoch_accuracy = epoch_correct / epoch_total
        epoch_accuracy = gather_batch(epoch_accuracy, mpi_config)
        if debug:
            jax.debug.print("Epoch Accuracy: {:.2f}%", epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")
    
    
    layer_weights_sizes = []
    for layer in network.layers:
        layer_weights_sizes.append(layer.weights_shape)
    print(f"rank {rank}: {layer_weights_sizes}")
    
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_iter_th(network, layer_weights_sizes, weights, mean, empty_neuron_states.thresholds)

    # Synchronize all MPI processes again
    mpi4jax.barrier(comm=comm)
    end_time = time.time()

    if rank == last_layer * process_per_layer:
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
                                "CNN",
                                all_history,
                                total_batches)
    return epoch_accuracy, mean, end_time - start_time

def get_layer_idx(batch_size, layer_sizes):
    '''
    Define for each MPI rank:
    - layer_idx:            Which layer it belongs to
    - process_per_layer:    How many MPI processes there are per layer
    - last_layer:           The index of the last layer
    - batch_part:           The size of the batch each rank has to process        
    '''
    global layer_idx 
    global process_per_layer
    global last_layer
    global batch_part
    global mpi_config

    last_layer = len(layer_sizes)-1
    process_per_layer = size // (last_layer+1)
    layer_idx = rank // process_per_layer
    batch_part = batch_size // process_per_layer

    mpi_config = MPIConfig(
        rank=rank,
        layer_idx=layer_idx,
        last_layer=last_layer,
        process_per_layer=process_per_layer,
        batch_part=batch_part,
        comm=comm
    )
    print(f"Rank {rank}, layer idx: {layer_idx}, batch part: {batch_part}, process per layer: {process_per_layer}, last rank: {last_layer}")

# region MAIN
def main(random_seed, key, rank_, size_, comm_, trial=None, trial_params=None, config_path=None, data_dir=""):  
    global training_generator
    global validation_generator
    global test_generator 
    global rank
    global size
    global comm
    global TQDM_DISABLE

    rank, size, comm = rank_, size_, comm_
    if rank != 0:
        TQDM_DISABLE = True
    
    # Load configuration from file or use defaults
    config = load_config_with_defaults(config_path)
    config = parse_unknown_args_and_overrides_config(unknown, config)

    # Extract configuration parameters
    dataset = config['dataset']
    layer_sizes = tuple(config['layer_sizes'])
    batch_size = config['batch_size']

    load_file = config['load_file']
    best = config['best']
    rerun = config['rerun']
    
    # Get the size of the biggest kernel (Partially used for getting top k elements but not mandatory anymore)
    max_kernel = 0
    for layer in layer_sizes:
        if len(layer) >= 4:
            k= layer[1]
            k_prod = (k[0]+k[0]-1) * (k[1]+k[1]-1)
            if k_prod > max_kernel:
                max_kernel = k_prod
    
    if size % len(layer_sizes) != 0:
        print(f"Error: layer_sizes ({len(layer_sizes)}) must match number of MPI ranks ({size})")
        sys.exit(1)

    get_layer_idx(batch_size, layer_sizes) # Compute the layer index for training/inference with multiple processes per batch
    
    if batch_size % process_per_layer != 0:
        print(f"Error: one batch ({batch_size}) must be divisible by the number of processes per layer ({process_per_layer})")
        sys.exit(1)

    for s in [2]: # Loop for multiple experiments 
        # Initialize parameters (input data for rank 0 and weights for other ranks)
        total_train_batches, total_val_batches, total_test_batches, max_nonzero = 0, 0, 0, 0
        if rank == 0:
            downsample = False
            # Load the data 
            match dataset:
                case "mnist":
                    loader = partial(mnist_loader_manual)
                    if layer_sizes[0][1] == 14:
                        downsample = True
                case "shd":
                    loader = torch_SHD_loader
                case "nmnist":
                    loader = torch_nmnist_loader
                case "dvs":
                    loader = partial(torch_DVSGesture_loader)
                    if layer_sizes[0][1] == 64:
                        downsample = True
                case _:
                    raise ValueError(f"Unknown dataset: {dataset}")
                
            if downsample:
                print("Downsampling the dataset...")
            # Load the data 
            train_data, val_data, test_data, max_nonzero = loader(  batch_size=batch_size, 
                                                                    shuffle=False,
                                                                    CNN_preprocess=True,
                                                                    downsample=downsample,
                                                                    data_dir=data_dir)
            training_generator, total_train_batches = train_data
            validation_generator, total_val_batches = val_data
            test_generator, total_test_batches =  test_data
            
        # Broadcast total_batches to all other ranks
        total_train_batches, total_val_batches, total_test_batches = bcast(jnp.array([total_train_batches, total_val_batches, total_test_batches]), root=0 , comm=comm)                
        max_nonzero = bcast(jnp.array([max_nonzero]), root=0, comm=comm)
        max_nonzero = max_nonzero.tolist()[0]
        
        params = Params(
            dataset=dataset,
            random_seed=random_seed,
            layer_sizes=layer_sizes, 
            init_thresholds=config['init_thresholds'], 
            num_epochs=config['num_epochs'], 
            learning_rate=config['learning_rate'], 
            batch_size=batch_size,
            load_file=load_file,
            shuffle_activations=config['shuffle_activations'],      # Whether shuffle the activations in the hidden layer or not
            restrict=config['restrict'],              # Reset rate for each layer
            firing_nb=config['firing_nb'],                 # How many top values do we allow to fire for each layer
            sync_rate=config['sync_rate'],                    # How many input values do we need to receive before firing
            max_nonzero=max_nonzero,        # Maximum size of the input data (Computed from the dataloader, do not change it here)
            shuffle_input=config['shuffle_input'],            # Whether shuffle the input values or not 
            threshold_lr=config['threshold_lr'],               # Threshold learning rate
            sparsity_impact=tuple(config['sparsity_impact']), # Beta sparse (Sparsity loss's impact)
            w_reg=config['w_reg'],                      # Weight regularization impact
            rerun="",
            top_weights=config['top_weights'],
            max_kernel=max_kernel,
            flat_layer_sizes=(),            # Each layer's shape
            history_size=config['history_size']                  # How many output states should we keep for plotting output history
        )

        # Build the network using the above parameters and initialize the weights
        key, subkey = jax.random.split(key) 
        network = Network.build(params, key, layer_sizes=layer_sizes, 
                                flat_layer_sizes=(), conv_layer_sizes=(), 
                                th_bias=0.0)
        weights = network.init_weights()
        empty_neuron_states = network.layers[layer_idx]

        if rerun is not None:
            override_list = config.get('override_params', None)
            params, weights, thresholds = rerun_init(
                rerun, 
                mpi_config, 
                params, 
                override_params=override_list
            )

            if layer_idx > 0:
                empty_neuron_states = network.rerun(thresholds)

        params = dataclasses.replace(params, flat_layer_sizes=network.flat_layer_sizes)

        if rank == 0:
            print(f"Number of training batches: {total_train_batches}, validation batches: {total_val_batches}, test batches: {total_test_batches}")
            print(params)
        
        # Select the correct layer computation function
        layer_computation = fc_layer_computation
        if empty_neuron_states.is_conv:
            layer_computation = conv_layer_computation
        
        total_batches = (total_train_batches, total_val_batches, total_test_batches)

        mode = config['mode']
        if mode == 'inference':
            # To only run inference
            batch_predict(params, key, total_batches, network, weights, empty_neuron_states, layer_computation, 'test', save=True, debug=True)
        elif mode == 'training':
            # To run the full training pipeline
            result_path = train(params, key, total_batches, network, weights, empty_neuron_states, layer_computation, "adam")
        else:
            print(f"Unknown mode in config file, choose either 'training' or 'inference', got {mode}")
            sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train async neural network')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to YAML configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--data_dir', type=str, default="",
                       help='Directory for storing and reading the datasets (default: current directory/data/)')
    args, unknown = parser.parse_known_args()
    
    random_seed = args.seed
    key = jax.random.key(random_seed)
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()      # Real rank
    size = comm.Get_size()

    main(random_seed, key, rank, size, comm, config_path=args.config, data_dir=args.data_dir)
# JAX_PLATFORMS=cpu mpirun -n 5 python async_CNN.py --config "CNN_config.yaml"