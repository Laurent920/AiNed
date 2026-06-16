import os
from tqdm import tqdm

from mpi4py import MPI
# os.environ["JAX_TRACEBACK_FILTERING"] = "on"
os.environ.pop("JAX_TRACEBACK_FILTERING", None)

import jax
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import jax.numpy as jnp
from jax import jit
import optax

import dataclasses
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
from dataset_helpers.cifar10_helper import cifar10_loader_manual
from dataset_helpers.iris_species_helper import torch_iris_loader
from dataset_helpers.network_helper import one_hot_encode
from dataset_helpers.nmnist_helper import torch_nmnist_loader
from dataset_helpers.shd_helper import torch_SHD_loader
from dataset_helpers.dvs_helper import torch_DVSGesture_loader
from dataset_helpers.ncars_helper import torch_NCARS_loader
from dataset_helpers.cnn_pytorch import get_weights_for_rank

from other_helpers.helpers import BaseParams, NeuronStates
from other_helpers.helpers import accuracy, prepare_result_payload, rerun_init, store_data_to_json, store_result_artifacts
from other_helpers.helpers import activation_func, keep_top_k, output_vector_to_event
from other_helpers.helpers import update_history, process_history, load_config_with_defaults, parse_unknown_args_and_overrides_config
from forward_backward_pass.backpropagation import MLP_back_prop
from forward_backward_pass.loss_functions import loss_bpp, loss_func
from other_helpers.event_pooling import output_to_event_array_with_pooling, full_matrix_to_event_array_with_pooling, pool_output_size

from other_helpers.MPI_helpers import MPIConfig, combine_batch_avg, gather_batch, split_batch, l2_weight_regularization
from other_helpers.init_weights import init_params
from forward_backward_pass.inference import predict, layer_computation as fc_layer_computation

jax.config.update("jax_debug_nans", True)

TQDM_DISABLE = False
STORE_EACH_EPOCH = False
BUFFER_SIZE = 0
END_SIGNAL = jnp.array([-1.0, -1.0, -1.0, -1.0], dtype=jnp.float32)

# Initialize empty global MPI variables
comm = None
rank = None      
size = None

layer_idx = None           # Rank corresponding to the layer
process_per_layer = None    # Number of processes for each layer
last_layer_idx = None            # Rank of last layer
batch_part_size = None           # The size of the batch on each process
mpi_config = None           # MPIConfig class for mpi helpers functions
processes_per_layer_global = None

training_generator = None
validation_generator = None
test_generator = None

#region ConvNeuronStates
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
        # Fields that belong to ConvNeuronStates specifically
        _own_fields = {"neuron_state", "weight_res", "kernel", "padding", 
                    "stride", "previous_layer", "is_conv", "pooling", 
                    "pool_size", "pool_stride"}

        neuron_state_updates = {k: v for k, v in updates.items() if k not in _own_fields}
        own_updates = {k: v for k, v in updates.items() if k in _own_fields}

        new_neuron_state = (
            self.neuron_state.replace(**neuron_state_updates)
            if neuron_state_updates
            else self.neuron_state
        )

        return ConvNeuronStates(
            neuron_state=own_updates.get("neuron_state", new_neuron_state),
            weight_res=own_updates.get("weight_res", self.weight_res),
            kernel=own_updates.get("kernel", self.kernel),
            padding=own_updates.get("padding", self.padding),
            stride=own_updates.get("stride", self.stride),
            previous_layer=own_updates.get("previous_layer", self.previous_layer),
            is_conv=own_updates.get("is_conv", self.is_conv),
            pooling=own_updates.get("pooling", self.pooling),
            pool_size=own_updates.get("pool_size", self.pool_size),
            pool_stride=own_updates.get("pool_stride", self.pool_stride),)

@dataclasses.dataclass(frozen=True)
class Params(BaseParams):
    max_kernel: int | None = None
    flat_layer_sizes: tuple[int, ...] | None = None

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
                sr = params.sync_rate
                sr = sr if isinstance(sr, int) else sr[layer_idx]
                sync_rate_vector = jnp.full(shape=layer, fill_value=sr)

                empty_neuron_states = NeuronStates(
                                    values=jnp.zeros(layer),
                                    bias=jnp.zeros(layer),
                                    thresholds=thresholds,
                                    input_residuals=jnp.zeros((prev_size,)),
                                    input_order=jnp.full((prev_size,), -1, dtype=int),
                                    input_activity=jnp.zeros((prev_size,), dtype=int),
                                    layer_activity=jnp.zeros((layer[0],), dtype=int),
                                    output_activity=jnp.zeros((prev_size, layer[0])),
                                    last_sent_iteration=jnp.full(shape=layer, fill_value=-1),
                                    input_vector=jnp.zeros((prev_size,)),
                                    output_vector=jnp.zeros((layer[0],)),
                                    sync_rate_vector=sync_rate_vector,
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
                    ep_h, ep_w = kernel[0] - 1 - padding[0], kernel[1] - 1 - padding[1]
                    values = jnp.full((out_chan, h_out + 2*ep_h, w_out + 2*ep_w), -10000.0)
                    values = values.at[:, ep_h:h_out+ep_h, ep_w:w_out+ep_w].set(0.0)
                    filename += f"_C{out_chan}x{in_chan}x{kernel[0]}x{kernel[1]}"

                h_out_pool, w_out_pool = h_out, w_out
                if pooling != "": # Compute the shape after pooling if needed (ceil mode)
                    h_out_pool = pool_output_size(h_out, pool_size[0], pool_stride[0])
                    w_out_pool = pool_output_size(w_out, pool_size[1], pool_stride[1])
                    if pooling == "max":
                        filename += f"_P{pool_size[0]}x{pool_size[1]}"
                    elif pooling == "avg":
                        filename += f"_AvgP{pool_size[0]}x{pool_size[1]}"

                key, subkey = jax.random.split(key)
                unpadded_shape = (out_chan, h_out, w_out)
                # thresholds = jax.random.uniform(subkey, values.shape) * params.init_thresholds + th_bias
                thresholds = jnp.full(values.shape, params.init_thresholds)
                weights_shape = (out_chan, in_chan, kernel[0], kernel[1])
                neuron_state = NeuronStates(
                    values=values,
                    bias=jnp.zeros(unpadded_shape),
                    thresholds=thresholds,
                    input_residuals=jnp.zeros(previous_layer.shape),
                    input_order=jnp.full(previous_layer.shape, -1, dtype=int),
                    input_activity=jnp.zeros(previous_layer.shape, dtype=int),
                    layer_activity=jnp.zeros(unpadded_shape),
                    output_activity=jnp.zeros_like(values),
                    last_sent_iteration=-1,
                    input_vector=jnp.zeros(previous_layer.shape),
                    output_vector=jnp.zeros(unpadded_shape),
                    values_history=jnp.zeros((params.history_size, *unpadded_shape)),
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
                conv_layer_sizes.append(unpadded_shape)          # Unpadded shape before pooling, used for gradient computation
        return cls(params=params, key=key, layers=tuple(layers), flat_layer_sizes=tuple(flat_layer_sizes), conv_layer_sizes=tuple(conv_layer_sizes), filename=filename)

    def init_weights(self):
        '''
        Initialize the weights for each layer based on the layer sizes.
        
        Returns the weights correponding to the MPI layer_idx.
        ''' 
        weights = init_params(self.key, self.layers, self.params, layer_idx, self.filename)
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

#region Conv computation
@partial(jax.jit, static_argnames=['params', 'mpi_config', 'grad',])
def conv_layer_computation(params, mpi_config, key, neuron_idx, layer_input, weights, neuron_states, iteration=0, grad=False):
    '''
    Apply the convolution for an incoming event in the event-driven manner described in "Optimizing event-based neural networks on digital neuromorphic architecture: a comprehensive design space exploration"
    This convolution only supports 'SAME' padding scheme with stride 1
    
    weights: (out_ch, in_ch, k_h, k_w)
    '''
    out_ch, in_ch, k_h, k_w = weights.shape
    c, x, y = neuron_idx
    pad_value = jnp.asarray(-10000.0, dtype=neuron_states.values.dtype)
    pad_h, pad_w = neuron_states.padding
    event_pad_h, event_pad_w = k_h - 1 - pad_h, k_w - 1 - pad_w
    H = neuron_states.values.shape[1] - 2 * event_pad_h
    W = neuron_states.values.shape[2] - 2 * event_pad_w

    sr = params.sync_rate if isinstance(params.sync_rate, int) else params.sync_rate[layer_idx]
    if sr == 10000:
        if neuron_states.pooling != "":
            C_f, H_f, W_f = params.flat_layer_sizes[layer_idx]
            event_array_size = C_f * H_f * W_f
        else:
            event_array_size = out_ch * H * W
    else:
        event_array_size = out_ch * k_h * k_w

    @jit
    def regular_input(neuron_states):
        # jax.debug.print("rank {} has x: {}, y: {}", rank, x, y)

        # Step 1: Multiply the input value by the flipped kernel to obtain the partial output values
        # layer_input is scalar for event-driven conv; scalar multiply is cheaper than dot.
        partial_activations = layer_input * jnp.flip(weights[:, c, :, :], axis=(1, 2)) # Shape (out_ch, k_h, k_w)
        # jax.debug.print("activations: {}", activations)        
        
        # Step 2: values and thresholds are stored pre-padded — use directly, no allocation needed.
        # event_pad_h/w and H/W are captured from the outer scope (computed once, not per event).
        values_padded = neuron_states.values
        thresholds_padded = neuron_states.thresholds
        # jax.debug.print("rank {}, neuron idx {}, padding: {}, start indices: {}", rank, neuron_idx, neuron_states.padding, start_indices)

        start_indices = (0, x, y) # Start indices for slicing and updating on padded matrices, always the same because the padding takes care of the needed offset
        slice_shape = partial_activations.shape  # (C, k_h, k_w)

        # jax.debug.print("rank {}, neuron idx: {}, start indices: {}, slice shape: {}, values padded shape: {}", rank, neuron_idx, start_indices, slice_shape, values_padded.shape)
        
        current_values_sliced = jax.lax.dynamic_slice(values_padded, start_indices, slice_shape)
        thresholds_sliced = jax.lax.dynamic_slice(thresholds_padded, start_indices, slice_shape)

        padding_mask = jnp.where(current_values_sliced == pad_value, 0.0, 1.0) # Mask to zero out the padded values
        # jax.debug.print("rank {}, partial activations {}, current values sliced {}, start indices {}, padding mask {}", rank, partial_activations, current_values_sliced, start_indices, padding_mask)
        
        # Step 3: Add the partial output values to the current values to get the complete output values
        activations = (current_values_sliced + partial_activations) * padding_mask
        updated_values_slice = activations
        
        # Step 4: Apply sync rate: Add 1 to the internal counter for sync rate, if counter exceeds it we fire
        activity_slice = jax.lax.dynamic_slice(neuron_states.output_activity, start_indices, slice_shape)
        ne_activity_slice = activity_slice + 1
        sr = params.sync_rate
        sr = sr if isinstance(sr, int) else sr[layer_idx]
        activations = jnp.where(ne_activity_slice >= sr, activations, 0.0)  # Only fire where the sync rate is reached

        # Step 5: Compute ReLu on the updated slice if fire is True
        activated_output = activation_func(thresholds_sliced, activations)
        # jax.debug.print("rank {}, input: {}, activations: {}, updated slice: {}, activated output: {}", rank, layer_input, activations, updated_slice, activated_output)

        # Step 6: Apply the firing number        
        f_nb = params.firing_nb
        k = f_nb if isinstance(f_nb, int) else f_nb[layer_idx]
        activated_output = keep_top_k(activated_output, k, max_kernel=params.max_kernel) # Get the top k activations
        
        # Step 7: Update the internal activity counter by resetting it for the neurons that have fired
        activation_mask = jnp.where(activated_output > 0, 0.0, 1.0) # Reset the activity counter where a neuron has fired
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

        # Restore pad_value in border positions before writing back (border never fires, must stay marked)
        remaining_value = jnp.where(padding_mask == 0, pad_value, remaining_value)
        values_padded = jax.lax.dynamic_update_slice(values_padded, remaining_value, start_indices)
        new_values = values_padded  # stays pre-padded; no crop needed

        # Step 10: Apply pooling and compute the output events
        nb_valid_elements, out_events, unpooled_coords, unpooled_vals = output_to_event_array_with_pooling(activated_output,
                                                                       start_indices,
                                                                       (out_ch, H, W),
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

        new_neuron_states = neuron_states.replace(
            values=new_values,
            input_residuals=new_input_residuals,
            input_activity=new_input_activity,
            layer_activity=new_layer_activity,
            output_activity=new_output_activity,
            weight_res=new_weight_res,)
        # jax.debug.print("rank {}, values padded: {}, current slice: {}, updated slice: {}, activated output: {}, remaining: {}, neuron state: {}", rank, values_padded.shape, current_slice.shape, updated_slice.shape, activated_output.shape, remaining_value.shape, neuron_states.values.shape)
        # jax.debug.print("rank {} iteration {} nb valid elements {}, activated output {}", rank, iteration, nb_valid_elements, out_events[0])

        return nb_valid_elements, out_events, new_neuron_states

    @jit
    def last_input(neuron_states):
        if sr != 10000:
            return jnp.array(0), jnp.zeros((event_array_size, 4)), neuron_states

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
        
        new_neuron_states = neuron_states.replace(
            values=remaining_value,
            input_activity=jnp.ones(neuron_states.input_activity.shape, dtype=int),
            layer_activity=new_layer_activity,)

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
    def input_layer(args):
        neuron_states, x = args # x binned is shape (timesteps, channel, height, width)
                                # x not binned is shape (max_nonzero, 4) (x, y, t, c)
        # jax.debug.print("Rank {}, input layer shape: {}", rank, x.shape)
        
        x_p = x
        # x_p = jnp.ones((50, rcv_size))
        def send_input(i, carry):
            timestep = carry
            data = x_p[i]
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
        send(END_SIGNAL, dest=rank+process_per_layer, tag=0, comm=comm)

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
            
            loop_iterations, activated_output, new_neuron_states = layer_computation(params, mpi_config, key, neuron_idx, layer_input, weights, neuron_states, iteration, grad)
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
            jax.lax.cond(layer_idx == last_layer_idx, lambda _: None, hidden_layers, (loop_iterations, activated_output)) # Don't send if we reach the last layer
            return neuron_states, neuron_idx, timestep, iteration+1
        
        neuron_idx, timestep, iteration =  jnp.zeros(3).astype(jnp.int32), 0, 0
        initial_state = (neuron_states, neuron_idx, timestep, iteration)
        
        # Loop until the rank receives a -1 timestep
        neuron_states, neuron_idx, timestep, iteration = jax.lax.while_loop(cond, forward_pass, initial_state)
        
        # Send -1 to the next rank when all incoming data has been processed
        jax.lax.cond(
            layer_idx != last_layer_idx,
            lambda _: send(END_SIGNAL, dest=rank + process_per_layer, tag=0, comm=comm),
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
    next_grad = recv(jnp.zeros((batch_part_size,) + params.flat_layer_sizes[layer_idx]), source=rank + process_per_layer, tag=2, comm=comm) # Shape: (B, 128)

    # Compute input's gradient and weight gradient
    weight_grad, th_grad, weight_res, bias_grad = MLP_back_prop(params, all_neuron_states, next_grad, layer_idx)
    weight_grad += 2 * params.w_reg * weights

    if layer_idx > 1:
        cur_relu_mask = (all_neuron_states.output_vector > 0).astype(next_grad.dtype)

        # Send gradient to the previous layer
        send_grad = jnp.dot(next_grad * cur_relu_mask, weights.T) # Shape: (B, 128) @ (128, 784) = (B, 784)
        send(send_grad, dest=rank-process_per_layer, tag=2, comm=comm)
    
    # Sparsity loss gradients 
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
    
    scaling = jax.lax.cond(params.sparsity_impact[layer_idx] > 0,
                           lambda _: params.sparsity_impact[layer_idx] / (all_iterations * batch_part_size * process_per_layer) ,
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
    next_grad_1 = recv(jnp.zeros((batch_part_size,) + out_layer_shape), source=rank + process_per_layer, tag=2, comm=comm) # Shape: (B, 5, 28, 28) // (B, 5, 14, 14) with pooling

    if empty_neuron_states.pooling != "":
        ph, pw = empty_neuron_states.pool_size
        sh, sw = empty_neuron_states.pool_stride

        # Reconstruct a stable pre-pooling proxy from activity to route pooled gradients.
        # This avoids broadcasting the same pooled gradient to every element in the window.
        pre_pool_proxy = all_neuron_states.layer_activity.astype(next_grad_1.dtype)

        # Ceil-mode: pad the conv map so VALID windows produce the (ceil) pooled
        # shape that matches next_grad_1; the gradient is cropped back afterwards.
        Hc, Wc = pre_pool_proxy.shape[-2], pre_pool_proxy.shape[-1]
        Hp = pool_output_size(Hc, ph, sh)
        Wp = pool_output_size(Wc, pw, sw)
        pad_h = (Hp - 1) * sh + ph - Hc
        pad_w = (Wp - 1) * sw + pw - Wc
        pad_val = -jnp.inf if empty_neuron_states.pooling == "max" else 0.0
        pre_pool_proxy = jnp.pad(pre_pool_proxy, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), constant_values=pad_val)

        @jit
        def pool_only(x):
            if empty_neuron_states.pooling == "max":
                return jax.lax.reduce_window(
                    x,
                    init_value=-jnp.inf,
                    computation=jax.lax.max,
                    window_dimensions=(1, 1, ph, pw),
                    window_strides=(1, 1, sh, sw),
                    padding="VALID",
                )
            if empty_neuron_states.pooling == "avg":
                out = jax.lax.reduce_window(
                    x,
                    init_value=0.0,
                    computation=jax.lax.add,
                    window_dimensions=(1, 1, ph, pw),
                    window_strides=(1, 1, sh, sw),
                    padding="VALID",
                )
                return out / float(ph * pw)
            return x

        _, pullback = jax.vjp(pool_only, pre_pool_proxy)
        next_grad = pullback(next_grad_1)[0][..., :Hc, :Wc]  # crop ceil-mode padding
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
    if params.init_thresholds != 0:
        k_h, k_w = weights.shape[2], weights.shape[3]
        ep_h_b, ep_w_b = k_h - 1 - empty_neuron_states.padding[0], k_w - 1 - empty_neuron_states.padding[1]
        thresholds = all_neuron_states.thresholds[0]
        H_th = thresholds.shape[1] - 2 * ep_h_b
        W_th = thresholds.shape[2] - 2 * ep_w_b
        thresholds = thresholds[:, ep_h_b:H_th+ep_h_b, ep_w_b:W_th+ep_w_b]
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
                           lambda _: params.sparsity_impact[layer_idx] / (all_iterations * batch_part_size * process_per_layer) ,
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
    # jax.debug.print("Rank {} finished forward pass in loss_fn with {}", rank, all_outputs)

    # Compute Loss and loss gradient
    loss, loss_grad = jax.value_and_grad(loss_func)(all_outputs, target)
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
    if layer_idx != last_layer_idx and rank == leader_rank:
        # jax.debug.print("Rank {}, sending activations {} and iterations {} to the last rank", rank, jnp.sum(activations), jnp.mean(iterations))
        send(jnp.sum(activations), dest=last_layer_idx * process_per_layer, tag=6,comm=comm)
        if rank == 0:
            send(jnp.mean(iterations), dest=last_layer_idx * process_per_layer, tag=6,comm=comm)
    elif layer_idx == last_layer_idx and rank == leader_rank:
        for i in range(last_layer_idx):
            # Storing the thresholds
            act_sum = recv(jnp.zeros(1), source=i * process_per_layer, tag=6, comm=comm)
            all_activations = all_activations + act_sum[0] # Sum of all activations in the hidden layers
            
            if i == 0: # Get iterations of input data
                iter_mean = recv(jnp.zeros(1), source=i * process_per_layer, tag=6, comm=comm)
                all_iterations = iter_mean[0]
        all_activations += jnp.sum(activations) # Adding the activations of the last layer

        sparsity_L = all_activations /  (all_iterations * batch_part_size * process_per_layer)
        # jax.debug.print("Rank {}, sparsity L: {}, all iterations: {}, all activations: {}", rank, sparsity_L, all_iterations, all_activations)
    all_iterations = bcast(all_iterations, root=last_layer_idx*process_per_layer, comm=comm)

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
    all_epoch_accuracies = []
    all_validation_accuracies = []
    all_loss = []
    all_history = None
    if layer_idx == last_layer_idx:
        all_history = []
    all_mean_iterations = []
    
    # Initialize the optimizer
    if rank == 0:
        print(f"{opti} optimizer selected")
    if opti == "adam":
        solver = optax.adam(learning_rate=params.learning_rate)
    elif opti == "adamw":
        solver = optax.adamw(learning_rate=params.learning_rate)
    elif opti == "sgd":
        solver = optax.sgd(learning_rate=params.learning_rate)
    elif opti == "sgd_onecycle":
        # Warmup-cosine schedule matching PyTorch OneCycleLR (best config from pytorch search).
        # learning_rate is the peak LR; starts at lr/25, decays to ~0 after warmup.
        # Use w_reg: 0.0 in config to avoid double-counting with add_decayed_weights.
        total_steps = params.num_epochs * total_batches[0]
        warmup_steps = int(0.3 * total_steps)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=params.learning_rate / 25,
            peak_value=params.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=1e-5,
        )
        solver = optax.chain(
            optax.add_decayed_weights(5e-4),
            optax.sgd(learning_rate=schedule, momentum=0.9, nesterov=True),
        )
    elif opti == "rmsprop":
        solver = optax.rmsprop(learning_rate=params.learning_rate, decay=0.9, eps=1e-8)
    elif opti == "amsgrad":
        solver = optax.amsgrad(learning_rate=params.learning_rate)
    elif opti == "adagrad":
        solver = optax.adagrad(learning_rate=params.learning_rate)
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

        if layer_idx == last_layer_idx:
            epoch_correct = 0
            epoch_total = 0
            epoch_loss = []
            
        epoch_iter_sum = 0.0
        epoch_iter_count = 0
        if layer_idx == 0:
            batch_iterator = None
            if rank == 0:
                batch_iterator = iter(training_generator) # Make the dataloader iterable
            
        for i in tqdm(range(total_batches[0]), disable=TQDM_DISABLE):
            neuron_states = empty_neuron_states
            if layer_idx == 0: # Input layer
                batch_x, batch_y = split_batch(params, batch_iterator, mpi_config, 4)

                # Send labels to the output layer via plain mpi4py to avoid mpi4jax cache pollution
                comm.Send(np.ascontiguousarray(np.asarray(batch_y, dtype=np.float32)), dest=last_layer_idx * process_per_layer + rank, tag=10)

                # Run the forward pass
                outputs, iterations, all_neuron_states = (conv_predict)(params, subkey, weights, neuron_states, layer_computation, jnp.array(batch_x))
                all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
            else:
                if layer_idx==last_layer_idx: # Output layer
                    # Receive the labels from the input layer via plain mpi4py
                    y_buf = np.empty((batch_part_size,), dtype=np.float32)
                    comm.Recv(y_buf, source=rank - (last_layer_idx * process_per_layer), tag=10)
                    y = y_buf
                    y_encoded = jnp.array(one_hot_encode(y, num_classes=params.layer_sizes[-1][0]))

                    # Run the forward and backward pass for the output layer
                    (loss, outputs, iterations, total_loss, history), gradients = (loss_fn)(params, subkey, weights, neuron_states, layer_computation, y_encoded, jnp.zeros((batch_part_size, 1, 4)))
                    # print(f"loss: {loss}")
                    weight_grad = gradients[0]
                    # weight_grad = gather_batch(weight_grad, mpi_config, average=True)
                    weight_grad = combine_batch_avg(weight_grad, mpi_config) # Gather the weight gradients from all ranks in the same layer
                
                    # Store the accuracy, loss and history                    
                    valid_y, batch_correct = accuracy(i, outputs, y, iterations, print=False)                 
                    # print(f"Batch {i}, Accuracy: {batch_correct}/{valid_y.shape[0]} ")
                    epoch_correct += int(batch_correct)
                    epoch_total += valid_y.shape[0]

                    epoch_loss.append(float(loss))
                    if params.history_size > 0:
                        all_history.append(history)
                else:
                    # Select the correct backward pass function
                    bwd_fn = predict_bwd
                    if empty_neuron_states.is_conv:
                        bwd_fn = conv_predict_bwd

                    # Run the forward and backward pass for the hidden layers
                    outputs, iterations, all_neuron_states, grads = (bwd_fn)(params, subkey, network.conv_layer_sizes, weights, neuron_states, layer_computation, jnp.zeros((batch_part_size, 1, 4)))
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
            valid_mask = iterations > 1
            epoch_iter_sum += float(jnp.sum(jnp.where(valid_mask, iterations, 0.0)))
            epoch_iter_count += int(jnp.sum(valid_mask))

        # Compute the average iterations for each layer
        mean = epoch_iter_sum / epoch_iter_count if epoch_iter_count > 0 else 0.0
        mean = gather_batch(jnp.array(mean), mpi_config)
        all_mean_iterations.append(float(mean))

        if layer_idx != 0:
            jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points and a mean threshold of {}", rank, mean, epoch_iter_count, jnp.mean(empty_neuron_states.thresholds))
        
        # Inference on the validation set
        val_accuracy, val_mean = 0.0, 0.0
        if total_batches[1] != 0: 
            val_accuracy, val_mean, _ = batch_predict(params, key, total_batches, network, weights, empty_neuron_states, layer_computation, dataset="val", save=False, debug=False)

        epoch_accuracy = 0.0
        if layer_idx == last_layer_idx:
            # Store loss values
            mean_loss = jnp.mean(jnp.array(epoch_loss))
            mean_loss = gather_batch(mean_loss, mpi_config)
            all_loss.append(float(mean_loss))

            # Store training and validation accuracies
            epoch_accuracy = epoch_correct / epoch_total
            epoch_accuracy = gather_batch(epoch_accuracy, mpi_config)
            all_epoch_accuracies.append(float(epoch_accuracy))
            all_validation_accuracies.append(float(val_accuracy))
            if rank == size-1:
                jax.debug.print("Epoch {} , Training Accuracy: {:.2f}%, Validation Accuracy: {:.2f}%, mean loss: {}, mean val iterations: {}", epoch, all_epoch_accuracies[-1] * 100, val_accuracy * 100, mean_loss, val_mean)
                jax.debug.print("----------------------------\n")
        epoch_accuracy = bcast(epoch_accuracy, root=size-1, comm=comm)
        if epoch_accuracy >= 0.9999:
            break
        if STORE_EACH_EPOCH: 
            all_iteration_mean = gather_iteration_means(jnp.array(all_mean_iterations))
            result_path_str = store_training_data_distributed(
                size,
                network,
                "train",
                all_epoch_accuracies,
                all_validation_accuracies,
                -1.0,
                time.time() - start_time,
                all_iteration_mean,
                weights,
                empty_neuron_states.thresholds,
                all_loss,
                opti,
                "CNN_temp",
                all_history,
                total_batches[0],
            )

    # Inference on the test set
    test_accuracy, test_mean, _ = batch_predict(params, key, total_batches, network, weights, empty_neuron_states, layer_computation, dataset="test", save=False, debug=False)
    
    all_iteration_mean = gather_iteration_means(jnp.array(all_mean_iterations))
    
    # Synchronize all MPI processes again
    mpi4jax.barrier(comm=comm)
    end_time = time.time()
    
    execution_time = end_time - start_time
    if rank == last_layer_idx * process_per_layer:
        print(f"Execution Time: {execution_time:.6f} seconds")

    result_path = store_training_data_distributed(
        size,
        network,
        "train",
        all_epoch_accuracies,
        all_validation_accuracies,
        test_accuracy,
        execution_time,
        all_iteration_mean,
        weights,
        empty_neuron_states.thresholds,
        all_loss,
        opti,
        "CNN",
        all_history,
        total_batches[0],
    )

    return result_path

def _flush_json_file(handle):
    handle.flush()
    os.fsync(handle.fileno())


def _write_result_json_prefix(result_path, result_data):
    metadata = dict(result_data)
    metadata.pop("thresholds", None)
    metadata.pop("weights", None)
    metadata_lines = json.dumps(metadata, indent=4).splitlines()
    body_lines = metadata_lines[1:-1]

    with open(result_path + ".json", "w", encoding="utf-8") as f:
        f.write("{\n")
        if body_lines:
            body_lines[-1] = body_lines[-1] + ","
            for line in body_lines:
                f.write(line + "\n")
        f.write('    "thresholds": {\n')
        _flush_json_file(f)


def _append_json_section_entry(result_path, key, value, trailing_comma):
    serializable_value = np.asarray(value).tolist()
    with open(result_path + ".json", "a", encoding="utf-8") as f:
        f.write(f"        {json.dumps(key)}: ")
        json.dump(serializable_value, f)
        if trailing_comma:
            f.write(",")
        f.write("\n")
        _flush_json_file(f)


def _switch_result_json_to_weights(result_path):
    with open(result_path + ".json", "a", encoding="utf-8") as f:
        f.write("    },\n")
        f.write('    "weights": {\n')
        _flush_json_file(f)


def _finalize_result_json(result_path):
    with open(result_path + ".json", "a", encoding="utf-8") as f:
        f.write("    }\n")
        f.write("}\n")
        _flush_json_file(f)


def gather_iteration_means(mean_iterations):
    leader_rank = layer_idx * process_per_layer
    save_root = last_layer_idx * process_per_layer
    all_iteration_mean = []

    if rank == leader_rank:
        payload = np.asarray(mean_iterations).tolist()
        if rank == save_root:
            collected = {layer_idx: payload}
            for i in range(last_layer_idx):
                src = i * process_per_layer
                collected[i] = comm.recv(source=src, tag=51)
            all_iteration_mean = [collected[i] for i in range(1, last_layer_idx + 1)]
            print("all iteration mean: rank", rank, all_iteration_mean)
        else:
            comm.send(payload, dest=save_root, tag=51)

    return all_iteration_mean


def store_training_data_distributed(size, network, mode, all_epoch_accuracies, all_validation_accuracies, test_accuracy, execution_time, all_iteration_mean, weights, thresholds, all_loss, optiname, network_type, all_history=None, total_batches=None):
    save_root = last_layer_idx * process_per_layer
    result_path = None

    if rank == save_root:
        result_path, result_data = prepare_result_payload(
            size,
            network,
            mode,
            all_epoch_accuracies,
            all_validation_accuracies,
            test_accuracy,
            execution_time,
            all_iteration_mean,
            {},
            all_loss,
            thresholds_dict={},
            optiname=optiname,
            network_type=network_type,
        )
        if result_path is not None:
            _write_result_json_prefix(result_path, result_data)

    result_path = comm.bcast(result_path, root=save_root)
    if result_path is None:
        return None

    comm.Barrier()

    for current_layer in range(1, last_layer_idx):
        if rank == current_layer * process_per_layer:
            _append_json_section_entry(
                result_path,
                f"thresholds_{current_layer}",
                thresholds,
                trailing_comma=current_layer < last_layer_idx - 1,
            )
        comm.Barrier()

    if rank == save_root:
        _switch_result_json_to_weights(result_path)
    comm.Barrier()

    for current_layer in range(1, last_layer_idx + 1):
        if rank == current_layer * process_per_layer:
            _append_json_section_entry(
                result_path,
                f"layer_{current_layer}",
                weights,
                trailing_comma=current_layer < last_layer_idx,
            )
        comm.Barrier()

    if rank == save_root:
        _finalize_result_json(result_path)
        print(f"Results saved to {result_path}")
        store_result_artifacts(
            result_path,
            mode,
            all_epoch_accuracies,
            all_validation_accuracies,
            test_accuracy,
            all_loss,
            all_iteration_mean,
            all_history,
            total_batches,
        )

    comm.Barrier()
    return result_path + ".json"


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

    all_history = None
    if layer_idx == last_layer_idx:
        epoch_correct = 0
        epoch_total = 0
        all_history = []

    epoch_iter_sum = 0.0
    epoch_iter_count = 0
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

            # Send label to the last layer via plain mpi4py
            comm.Send(np.ascontiguousarray(np.asarray(batch_y, dtype=np.float32)), dest=last_layer_idx * process_per_layer + rank, tag=10)
        else:
            # outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, jnp.zeros((batch_part_size, layer_sizes[0])))
            batch_data = jnp.zeros((batch_part_size, 1, 4))

            outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, batch_data)
            # jax.debug.print("Rank {} All neuron states values shape: {}, output shape : {}", rank, all_neuron_states.values.shape, outputs)

            if layer_idx == last_layer_idx:
                # jax.debug.print("Rank {} All neuron states values shape: {}, output shape : {}", rank, all_neuron_states.values.shape, outputs)

                y_buf = np.empty((batch_part_size,), dtype=np.float32)
                comm.Recv(y_buf, source=rank - (last_layer_idx * process_per_layer), tag=10)
                y = y_buf
                
                valid_y, batch_correct = accuracy(i, outputs, y, iterations, print=False)                 
                
                epoch_correct += int(batch_correct)
                epoch_total += valid_y.shape[0]
                # store_data_to_json(f"{len(params.layer_sizes)}hidden_single_output.json", outputs.tolist(), y.tolist())

                if params.history_size > 0:
                    # One-hot target → scalar class index
                    history = process_history(all_neuron_states.values_history, all_neuron_states.history_index, y)
                    all_history.append(history)
            # store_data_to_json(f"{len(params.layer_sizes)}hidden_intermediates_layer{rank}.json", outputs.tolist())

        # store_data_to_json(f"{len(params.layer_sizes)}hidden_iterations_layer{rank}.json", iterations.tolist())
        valid_mask = iterations > 1
        epoch_iter_sum += float(jnp.sum(jnp.where(valid_mask, iterations, 0.0)))
        epoch_iter_count += int(jnp.sum(valid_mask))
        # if i >= 0:
        #     break

    mean = epoch_iter_sum / epoch_iter_count if epoch_iter_count > 0 else 0.0
    mean = gather_batch(jnp.array(mean), mpi_config)

    if rank != 0 and debug:
        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, epoch_iter_count*process_per_layer)
    
    epoch_accuracy = -1.0
    if layer_idx == last_layer_idx:
        epoch_accuracy = epoch_correct / epoch_total
        epoch_accuracy = gather_batch(epoch_accuracy, mpi_config)
        if debug:
            jax.debug.print("Epoch Accuracy: {:.2f}%", epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")
    all_iteration_mean = []
    if save:
        all_iteration_mean = gather_iteration_means(mean)

    # Synchronize all MPI processes again
    mpi4jax.barrier(comm=comm)
    end_time = time.time()

    execution_time = end_time - start_time
    if rank == last_layer_idx * process_per_layer and debug:            
        print(f"Execution Time: {execution_time:.6f} seconds")
    if save:
        accuracies = {"train": [-1], "val": [-1], "test": [-1]}
        if dataset in accuracies:
            accuracies[dataset] = [epoch_accuracy]

        store_training_data_distributed(
            size,
            network,
            "inference",
            accuracies["train"],
            accuracies["val"],
            accuracies["test"][0],
            execution_time,
            all_iteration_mean,
            weights,
            empty_neuron_states.thresholds,
            [],
            None,
            "CNN",
            all_history,
            total_batches,
        )
    return epoch_accuracy, mean, end_time - start_time

def get_layer_idx(batch_size, layer_sizes, processes_per_layer=None):
    '''
    Define for each MPI rank:
    - layer_idx:            Which layer it belongs to
    - process_per_layer:    How many MPI processes there are per layer
    - last_layer_idx:           The index of the last layer
    - batch_part_size:           The size of the batch each rank has to process        
    '''
    global layer_idx 
    global process_per_layer
    global last_layer_idx
    global batch_part_size
    global mpi_config
    global processes_per_layer_global

    last_layer_idx = len(layer_sizes)-1
    process_per_layer = size // (last_layer_idx+1)
    layer_idx = rank // process_per_layer
    batch_part_size = batch_size // process_per_layer

    mpi_config = MPIConfig(
        rank=rank,
        layer_idx=layer_idx,
        last_layer_idx=last_layer_idx,
        process_per_layer=process_per_layer,
        batch_part_size=batch_part_size,
        comm=comm
    )
    print(f"Rank {rank}, layer idx: {layer_idx}, batch part: {batch_part_size}, process per layer: {process_per_layer}, last rank: {last_layer_idx}")

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
                case "ncars":
                    loader = partial(torch_NCARS_loader, dedup=config.get('dedup', False), augment=config.get('augment', False))
                    if tuple(layer_sizes[0][1:]) == (60, 50):
                        downsample = True
                case "cifar10":
                    loader = cifar10_loader_manual
                    if layer_sizes[0][1] == 16:
                        downsample = True
                case _:
                    raise ValueError(f"Unknown dataset: {dataset}")
                
            if downsample:
                print("Downsampling the dataset...")
            # Load the data 
            loader_kwargs = dict(batch_size=batch_size, shuffle=False, CNN_preprocess=True,
                                 downsample=downsample, data_dir=data_dir)
            if dataset == "cifar10":
                loader_kwargs['augment'] = config.get('augment', False)
            train_data, val_data, test_data, max_nonzero = loader(**loader_kwargs)
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
            rerun=None,
            top_weights=config['top_weights'],
            max_kernel=max_kernel,
            flat_layer_sizes=(),            # Each layer's shape
            history_size=config['history_size'],                 # How many output states should we keep for plotting output history
            output_decay=config.get('output_decay', 1.0),       # Per-event weight decay at output layer
            augment=config.get('augment', False),
            dedup=config.get('dedup', False),
            use_best=config.get('use_best', False),
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
        network = dataclasses.replace(network, params=params)

        if rank == 0:
            print(f"Number of training batches: {total_train_batches}, validation batches: {total_val_batches}, test batches: {total_test_batches}")
            print(params)
        
        # Select the correct layer computation function
        layer_computation = fc_layer_computation
        if empty_neuron_states.is_conv:
            layer_computation = conv_layer_computation
        
        total_batches = (total_train_batches, total_val_batches, total_test_batches)
        subset = config.get('train_subset_batches', None)
        if subset is not None:
            total_train_batches = min(int(subset), int(total_train_batches))
            total_batches = (total_train_batches, 0, 0)

        mode = config['mode']
        if mode == 'inference':
            # To only run inference
            batch_predict(params, key, total_batches, network, weights, empty_neuron_states, layer_computation, 'test', save=True, debug=True)
        elif mode == 'training':
            # To run the full training pipeline
            result_path = train(params, key, total_batches, network, weights, empty_neuron_states, layer_computation, config.get('optimizer', 'adam'))
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

'''
JAX_PLATFORMS=cpu mpirun -n 5 python async_CNN.py --config "configs/CNN_config.yaml"
'''