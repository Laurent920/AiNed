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
import gc
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
from other_helpers.helpers import update_history, process_history, load_config_with_defaults, parse_unknown_args_and_overrides_config
from forward_backward_pass.backpropagation import MLP_back_prop
from forward_backward_pass.loss_functions import loss_bpp, loss_func

from other_helpers.general_MPI_helper import CNN_data_split, CNN_model_split_custom
from other_helpers.init_weights import init_params
from other_helpers.event_pooling import pool_output_size
from forward_backward_pass.inference import predict, layer_computation as fc_layer_computation, conv_layer_computation

jax.config.update("jax_debug_nans", True)

TQDM_DISABLE = False
STORE_EACH_EPOCH = True
BUFFER_SIZE = 0
# Diagnostic: dump init weights + per-layer grads + logits + input batch for batch 0, then exit.
_GRAD_DUMP = os.environ.get("AED_GRAD_DUMP", "") == "1"
_GRAD_DUMP_DIR = os.environ.get("AED_GRAD_DUMP_DIR", "grad_dump")
END_SIGNAL = jnp.array([-1.0, -1.0, -1.0, -1.0], dtype=jnp.float32)

# Hidden→hidden (FC) backward gradient. DEFAULT: old unmasked dot(next_grad, W.T) (trains deep
# nets faster). Set AINED_LEGACY_BWD=0 to restore the (output_vector>0) ReLU-derivative mask.
_LEGACY_BWD_GRAD = os.environ.get("AINED_LEGACY_BWD", "1") == "1"

# Optimizer schedule. DEFAULT: cosine decay + grad clip (kept ON here for data-augmentation
# stability, see optimizer comment). Set AINED_CONST_LR=1 for constant LR / plain Adam.
_CONST_LR = os.environ.get("AINED_CONST_LR", "0") == "1"
# Grad clip: "auto" = on unless const LR (legacy behavior); "1"/"0" force on/off independently.
_GRAD_CLIP = os.environ.get("AINED_GRAD_CLIP", "auto")

# Optional per-run output subfolder, so parallel sweep runs whose architecture
# (and thus filename) is identical don't collide. Mirrors async_MLP_general.py.
_RUN_TAG = os.environ.get("AINED_RUN_TAG", "")

# Initialize empty global MPI variables
comm = None
rank = None      
size = None

layer_idx = None           # Rank corresponding to the layer
processes_per_layer_global = None    # Number of processes for each layer
last_layer_idx = None            # Rank of last layer
batch_part_size = None           # The size of the batch on each process
mpi_config = None           # MPIConfig class for mpi helpers functions

training_generator = None
validation_generator = None
test_generator = None
max_test_batches = 0  # 0 = no limit

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
                sr = params.sync_rate if isinstance(params.sync_rate, int) else params.sync_rate[i]
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
                                    # Only the output layer records history; other layers keep an empty buffer.
                                    values_history=jnp.zeros((params.history_size if i == len(layer_sizes) - 1 else 0, layer[0])),
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
                    # Pre-pad values: border=-10000 (marks out-of-bounds), valid region=0.
                    # Avoids a jnp.pad allocation on every conv event inside the while-loop.
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
                unpadded_shape = (out_chan, h_out, w_out)  # unpadded dims, used for gradient fields
                # thresholds = jax.random.uniform(subkey, values.shape) * params.init_thresholds + th_bias
                thresholds = jnp.full(values.shape, params.init_thresholds)  # pre-padded (same shape as values)
                weights_shape = (out_chan, in_chan, kernel[0], kernel[1])
                neuron_state = NeuronStates(
                    values=values,                                                    # pre-padded
                    bias=jnp.zeros(unpadded_shape),                                  # unpadded
                    thresholds=thresholds,                                            # pre-padded
                    input_residuals=jnp.zeros(previous_layer.shape),
                    input_order=jnp.full(previous_layer.shape, -1, dtype=int),
                    input_activity=jnp.zeros(previous_layer.shape, dtype=int),
                    layer_activity=jnp.zeros(unpadded_shape),                        # unpadded (used in gradient)
                    output_activity=jnp.zeros_like(values),                          # pre-padded
                    last_sent_iteration=-1,
                    input_vector=jnp.zeros(previous_layer.shape),
                    output_vector=jnp.zeros(unpadded_shape),                         # unpadded (used in gradient)
                    # Conv layers are never the output layer, so no history is stored.
                    values_history=jnp.zeros((0, *unpadded_shape)),                   # unpadded
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
        # params.flat_layer_sizes is only populated after init_weights(), so pass the built one.
        weights = init_params(self.key, self.layers, self.params, layer_idx, self.filename,
                              flat_layer_sizes=self.flat_layer_sizes)
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

#region Training helpers
@partial(jax.jit, static_argnames=['params', 'layer_computation', 'conv_layer_sizes'])
def predict_bwd(params, key, conv_layer_sizes, weights, empty_neuron_states, layer_computation, batch_data):
    '''
    B: batch_size
    '''
    # all_outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, batch_data, grad=True)
    all_outputs, iterations, all_neuron_states, buffer = (predict)( params, 
                                                            mpi_config,
                                                            key, 
                                                            weights, 
                                                            empty_neuron_states, 
                                                            layer_computation, 
                                                            batch_data,
                                                            message_size=4,
                                                            grad=True,
                                                            END_SIGNAL=END_SIGNAL,
                                                            BUFFER_SIZE=BUFFER_SIZE)
    # jax.debug.print("Rank {} finished forward pass in predict_bwd with {} it", mpi_config.rank, iterations)
    # w_sum = l2_weight_regularization(mpi_config, weights)

    # Receive the gradients from the later layers.
    # Sum contributions from all next-layer ranks (may be multiple with model parallelism).
    next_grad = jnp.zeros((batch_part_size,) + params.flat_layer_sizes[layer_idx])
    for _next_rank, _ in mpi_config.next_layer:
        next_grad = next_grad + recv(jnp.zeros((batch_part_size,) + params.flat_layer_sizes[layer_idx]), source=_next_rank, tag=2, comm=comm)

    # Weights/states are model-partitioned (see MPI_partition in main): keep the owned slice
    next_grad = next_grad[:, mpi_config.model_part.start_idx:mpi_config.model_part.end_idx+1]

    # Compute input's gradient and weight gradient
    weight_grad, th_grad, weight_res, bias_grad = MLP_back_prop(params, all_neuron_states, next_grad, layer_idx)
    weight_grad += 2 * params.w_reg * weights

    if layer_idx > 1:
        if _LEGACY_BWD_GRAD:
            # Old form: unmasked gradient propagation (no ReLU-derivative gate on next_grad).
            send_grad = jnp.dot(next_grad, weights.T)
        else:
            cur_relu_mask = (all_neuron_states.output_vector > 0).astype(next_grad.dtype)
            send_grad = jnp.dot(next_grad * cur_relu_mask, weights.T) # Shape: (B, 128) @ (128, 784) = (B, 784)
        # Send gradient to all previous-layer ranks (may be multiple with model parallelism)
        for _prev_rank, _ in mpi_config.previous_layer:
            send(send_grad, dest=_prev_rank, tag=2, comm=comm)

    
    # Sparsity loss gradients 
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
    
    scaling = jax.lax.cond(params.sparsity_impact[layer_idx] > 0,
                           lambda _: params.sparsity_impact[layer_idx] / (all_iterations * batch_part_size * mpi_config.get_process_per_batch) ,
                           lambda _: 0.0,
                           None)
    
    input_activity = jnp.sum(all_neuron_states.input_activity, axis=0) # Shape (784)
    layer_activity = jnp.sum(all_neuron_states.layer_activity, axis=0) # Shape (128)
    
    layer_activity = mpi_config.gather_batch(layer_activity, average=False) # Gather the weight gradients from all ranks in the same layer
    input_activity = mpi_config.gather_batch(input_activity, average=False)
    
    sparsity_residuals = scaling * layer_activity # Shape: (128,)
    # jax.debug.print("Rank {}, scaling mean: {}, sparsity_residuals mean: {}, sparsity_residuals sum: {}", rank, scaling, jnp.mean(sparsity_residuals), jnp.sum(sparsity_residuals))
    
    th_sparsity_grad = -sparsity_residuals
    weight_sparsity_grad = jnp.outer(input_activity, sparsity_residuals) # Shape: (784, 128)
    # jax.debug.print("Rank {}, th_sparsity_grad: {}, weight_sparsity_grad: {}", rank, jnp.mean(th_sparsity_grad), np.mean(weight_sparsity_grad))
    # jax.debug.print("rank {} finished predict bwd", mpi_config.rank)

    return all_outputs, iterations, all_neuron_states, (weight_grad, th_grad, weight_sparsity_grad, th_sparsity_grad) 

@partial(jax.jit, static_argnames=['params', 'layer_computation', 'conv_layer_sizes'])
def conv_predict_bwd(params, key, conv_layer_sizes, weights, empty_neuron_states, layer_computation, batch_data):
    # all_outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, batch_data, grad=True)
    all_outputs, iterations, all_neuron_states, buffer = (predict)( params,
                                                            mpi_config,
                                                            key, 
                                                            weights, 
                                                            empty_neuron_states, 
                                                            layer_computation, 
                                                            batch_data,
                                                            message_size=4,
                                                            grad=True,
                                                            END_SIGNAL=END_SIGNAL,
                                                            BUFFER_SIZE=BUFFER_SIZE)
    # jax.debug.print("Rank {} finished forward pass in conv_predict_bwd with {} it", mpi_config.rank, iterations)
    # w_sum = l2_weight_regularization(mpi_config, weights)

    # Receive the gradients from the later layers.
    # With model parallelism, there may be multiple next-layer ranks each owning a different
    # output region of this layer. Sum all their contributions to get the full gradient.
    out_layer_shape = params.flat_layer_sizes[layer_idx]
    next_grad_1 = jnp.zeros((batch_part_size,) + out_layer_shape)
    for _next_rank, _ in mpi_config.next_layer:
        next_grad_1 = next_grad_1 + recv(jnp.zeros((batch_part_size,) + out_layer_shape), source=_next_rank, tag=2, comm=comm)

    if empty_neuron_states.pooling != "":
        ph, pw = empty_neuron_states.pool_size
        sh, sw = empty_neuron_states.pool_stride

        # Route the pooled gradient to the true argmax cell of each window. output_vector
        # holds the max winning (pooled) value per pre-pool cell, so the vjp of max-pool
        # sends the gradient to the cell that actually produced the pooled output — unlike
        # layer_activity (firing counts), whose argmax can be the wrong cell and whose error
        # compounds across stacked pooling layers.
        pre_pool_proxy = all_neuron_states.output_vector.astype(next_grad_1.dtype)

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
    if _GRAD_DUMP:
        def _dump_internals(ng1, ng, ir):
            os.makedirs(_GRAD_DUMP_DIR, exist_ok=True)
            np.savez(os.path.join(_GRAD_DUMP_DIR, f"conv{int(layer_idx)}_intern.npz"),
                     next_grad_1=np.asarray(ng1), next_grad=np.asarray(ng), input_residuals=np.asarray(ir))
        jax.debug.callback(_dump_internals, next_grad_1, next_grad, input_residuals)
    weight_grad = jax.vmap(grad_w)(input_residuals, next_grad) # Shape: (B, 5, 3, 3, 3) #TODO Use batches in convolution directly instead of vmap?

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

    th_grad = -jnp.mean(next_grad * layer_activity, axis=0)
    if params.init_thresholds != 0:
        thresholds = all_neuron_states.thresholds[0]
        th_grad = th_grad * thresholds * (1 - thresholds)

    if layer_idx > 1:
        # Compute gradient w.r.t. previous layer's activations and send to all previous-layer ranks.
        # With model parallelism, each model-parallel rank in this layer sends its partial grad_x
        # contribution (from its owned output channels/rows/cols) to the previous layer. The previous
        # layer sums contributions from all senders in next_grad accumulation above.
        send_grad = (grad_x)(next_grad, weights)
        for _prev_rank, _ in mpi_config.previous_layer:
            send(send_grad, dest=_prev_rank, tag=2, comm=comm)

     # Sparsity loss gradients 
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
    
    scaling = jax.lax.cond(params.sparsity_impact[layer_idx] > 0,
                           lambda _: params.sparsity_impact[layer_idx] / (all_iterations * batch_part_size * mpi_config.get_process_per_batch) ,
                           lambda _: 0.0,
                           None)
    
    input_activity = jnp.sum(all_neuron_states.input_activity, axis=0) # Shape (1, 28, 28)
    layer_activity = jnp.sum(all_neuron_states.layer_activity, axis=0) # Shape (3, 28, 28)
    
    layer_activity = mpi_config.gather_batch(layer_activity, average=False) 
    input_activity = mpi_config.gather_batch(input_activity, average=False)
    
    sparsity_residuals = scaling * layer_activity # Shape: (128,)
    # jax.debug.print("Rank {}, scaling mean: {}, sparsity_residuals: {}, " \
    # "layer_activity {} input_activity: {}, ", rank, scaling, jnp.mean(sparsity_residuals), jnp.mean(layer_activity), jnp.mean(input_activity))
    
    th_sparsity_grad = -sparsity_residuals
    weight_sparsity_grad = grad_w(input_activity.astype(jnp.float32), sparsity_residuals) # Shape: (784, 128)
    # jax.debug.print("rank {}, sparsity weight grad {} {}", rank, weight_sparsity_grad.shape, weights.shape)
    # weight_sparsity_grad = jnp.zeros_like(weights)
    # jax.debug.print("rank {} finished conv predict bwd", mpi_config.rank)

    return all_outputs, iterations, all_neuron_states, (weight_grad, th_grad, weight_sparsity_grad, th_sparsity_grad) 

# Define the loss function
@partial(jax.jit, static_argnames=['params', 'layer_computation',])
def loss_fn(params, key, weights, empty_neuron_states, layer_computation, target, batch_data):
    # all_outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, batch_data, grad=True)
    all_outputs, iterations, all_neuron_states, buffer = (predict)( params, 
                                                            mpi_config,
                                                            key, 
                                                            weights, 
                                                            empty_neuron_states, 
                                                            layer_computation, 
                                                            batch_data,
                                                            message_size=4,
                                                            grad=True,
                                                            END_SIGNAL=END_SIGNAL,
                                                            BUFFER_SIZE=BUFFER_SIZE)
    # jax.debug.print("Rank {} finished forward pass in loss_fn with {} and {} it", rank, all_outputs, iterations)
    # w_sum = l2_weight_regularization(mpi_config, weights)

    # Reconstruct the full output vector from the model-partitioned last-layer ranks
    full_outputs = mpi_config.gather_model_partition(all_outputs)

    # Compute Loss and loss gradient
    loss, loss_grad = jax.value_and_grad(loss_func)(full_outputs, target)
    loss_grad /= mpi_config.get_process_per_batch  # Shape (B, 10)
    loss_grad = loss_grad[:, mpi_config.model_part.start_idx:mpi_config.model_part.end_idx+1] # Keep this rank's owned slice
    # loss += params.w_reg * w_sum

    # Compute output gradient and weight gradient
    out_grad, weight_grad = jax.vmap(loss_bpp, in_axes=(None, 0, 0))(weights, all_neuron_states, loss_grad) # Shape (B, 128), (B, 128, 10)
    mean_weight_grad = jnp.mean(weight_grad, axis=0) # Shape: (128, 10)
    mean_weight_grad += 2 * params.w_reg * weights
    mean_weight_grad = jnp.expand_dims(mean_weight_grad, axis=0)  # Shape: (1, 128, 10)

    # Send gradient to all previous-layer ranks (may be multiple with model parallelism)
    for _prev_rank, _ in mpi_config.previous_layer:
        send(out_grad, dest=_prev_rank, tag=2, comm=comm)

    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)

    total_loss = loss + sparsity_L 

    acc_history, avg_rank = None, None
    if params.history_size > 0:
        # One-hot target → scalar class index
        target_labels = jnp.argmax(target, axis=-1)
        acc_history, avg_rank = process_history(all_neuron_states.values_history, all_neuron_states.history_index, target_labels)

    return (loss, full_outputs, iterations, total_loss, (acc_history, avg_rank)), (mean_weight_grad, loss_grad)

def sparsity_loss(params, all_neuron_states, iterations):
    '''
    Compute the sparsity loss based on the input residuals and the number of iterations
    '''
    if params.sparsity_impact[layer_idx] <= 0.0:
        return 0, 1, 0
    
    # Gather all the activations at the last layer to compute the sparsity loss
    leader_rank = layer_idx * processes_per_layer_global
    # jax.debug.print("Rank {}, activations shape: {}, iterations shape: {}, last layer neuron values: {}", rank, (activations.shape), jnp.mean(iterations.shape), jnp.sum(all_neuron_states.values))

    activations = mpi_config.gather_batch(all_neuron_states.input_residuals, average=False) # Gather the weight gradients from all ranks in the same layer
    iterations = mpi_config.gather_batch(iterations, average=True) # Gather the iterations from all ranks in the same layer
    # jax.debug.print("Rank {}, activations shape: {}, iterations shape: {}, last layer neuron values: {}", rank, (activations.shape), jnp.mean(iterations.shape), jnp.sum(all_neuron_states.values))

    all_iterations = 0.0
    all_activations = 0.0
    sparsity_L = 0.0
    if layer_idx != last_layer_idx and rank == leader_rank:
        # jax.debug.print("Rank {}, sending activations {} and iterations {} to the last rank", rank, jnp.sum(activations), jnp.mean(iterations))
        send(jnp.sum(activations), dest=last_layer_idx * processes_per_layer_global, tag=6,comm=comm)
        if rank == 0:
            send(jnp.mean(iterations), dest=last_layer_idx * processes_per_layer_global, tag=6,comm=comm)
    elif layer_idx == last_layer_idx and rank == leader_rank:
        for i in range(last_layer_idx):
            # Storing the thresholds
            act_sum = recv(jnp.zeros(1), source=i * processes_per_layer_global, tag=6, comm=comm)
            all_activations = all_activations + act_sum[0] # Sum of all activations in the hidden layers
            
            if i == 0: # Get iterations of input data
                iter_mean = recv(jnp.zeros(1), source=i * processes_per_layer_global, tag=6, comm=comm)
                all_iterations = iter_mean[0]
        all_activations += jnp.sum(activations) # Adding the activations of the last layer

        sparsity_L = all_activations /  (all_iterations * batch_part_size * processes_per_layer_global)
        # jax.debug.print("Rank {}, sparsity L: {}, all iterations: {}, all activations: {}", rank, sparsity_L, all_iterations, all_activations)
    all_iterations = bcast(all_iterations, root=last_layer_idx*processes_per_layer_global, comm=comm)

    return all_activations, all_iterations, sparsity_L

# region TRAINING
def train(params: Params, key, total_batches, network, weights, empty_neuron_states, layer_computation, opti, trial=None, extra_fields=None):
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
    # Cosine LR decay (to 1/10 of base) + global-norm gradient clipping stabilize training
    # under data augmentation: clipping caps the per-batch gradient spikes that augmentation
    # amplifies, and the decay damps the late-epoch divergence seen with a constant LR.
    steps_per_epoch = max(1, int(total_batches[0]))
    total_steps = max(1, params.num_epochs * steps_per_epoch)
    if _CONST_LR:
        lr_schedule = params.learning_rate
    else:
        lr_schedule = optax.cosine_decay_schedule(params.learning_rate, decay_steps=total_steps, alpha=0.1)
    _clip_on = (_GRAD_CLIP == "1") or (_GRAD_CLIP == "auto" and not _CONST_LR)
    grad_clip = optax.clip_by_global_norm(1.0) if _clip_on else optax.identity()
    if opti == "adam":
        solver = optax.chain(grad_clip, optax.adam(learning_rate=lr_schedule))
    elif opti == "adamw":
        solver = optax.chain(grad_clip, optax.adam(learning_rate=lr_schedule))
    elif opti == "sgd":
        solver = optax.chain(grad_clip, optax.sgd(learning_rate=lr_schedule))
    elif opti == "rmsprop":
        solver = optax.chain(grad_clip, optax.rmsprop(learning_rate=lr_schedule, decay=0.9, eps=1e-8))
        print("amsgrad optimizer selected")
        solver = optax.chain(grad_clip, optax.amsgrad(learning_rate=lr_schedule))
    elif opti == "lion":
        solver = optax.chain(grad_clip, optax.lion(learning_rate=lr_schedule))
    else:
        solver = None
    if solver is not None:
        opt_state = solver.init(weights)
    
    th_solver = optax.adam(learning_rate=params.threshold_lr)
    if params.init_thresholds != 0:
        th_opt_state = th_solver.init(jax.scipy.special.logit(empty_neuron_states.thresholds))
    else:
        th_opt_state = th_solver.init(empty_neuron_states.thresholds)
    
    # Synchronize all ranks and start timer
    mpi4jax.barrier(comm=comm)
    start_time = time.time()

    best_val_acc = -float('inf')
    best_weights = weights
    best_neuron_states = empty_neuron_states
    best_epoch = 0

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
            
        for i in tqdm(range(total_batches[0]), miniters=total_batches[0]//10, maxinterval=float('inf'), disable=TQDM_DISABLE):
            neuron_states = empty_neuron_states
            if layer_idx == 0: # Input layer
                batch_x, batch_y = mpi_config.split_batch(params, batch_iterator, 4)

                # Send labels to the output layer via plain mpi4py to avoid mpi4jax cache pollution
                # comm.Send(np.ascontiguousarray(np.asarray(batch_y, dtype=np.float32)), dest=last_layer_idx * processes_per_layer_global + rank, tag=10)
                mpi_config.send_labels(batch_y, mpi_config.batch_first_and_last_rank[0]) # Send to the labels to the output layer

                # Run the forward pass
                # outputs, iterations, all_neuron_states = (conv_predict)(params, subkey, weights, neuron_states, layer_computation, jnp.array(batch_x))
                outputs, iterations, all_neuron_states, buffer = (predict)( params, 
                                                                    mpi_config,
                                                                    subkey, 
                                                                    weights, 
                                                                    neuron_states, 
                                                                    layer_computation, 
                                                                    jnp.array(batch_x),
                                                                    message_size=4,
                                                                    grad=False,
                                                                    END_SIGNAL=END_SIGNAL,
                                                                    BUFFER_SIZE=BUFFER_SIZE)
                all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
                if _GRAD_DUMP and i == 0:
                    os.makedirs(_GRAD_DUMP_DIR, exist_ok=True)
                    np.savez(os.path.join(_GRAD_DUMP_DIR, "input.npz"),
                             batch_x=np.asarray(batch_x), batch_y=np.asarray(batch_y))
                    comm.Barrier(); sys.exit(0)
            else:
                if layer_idx==last_layer_idx: # Output layer
                    # Receive the labels from the input layer via plain mpi4py
                    # y_buf = np.empty((batch_part_size,), dtype=np.float32)
                    # comm.Recv(y_buf, source=rank - (last_layer_idx * processes_per_layer_global), tag=10)
                    # y = y_buf

                    y = mpi_config.recv_labels()
                    y_encoded = jnp.array(one_hot_encode(y, num_classes=params.layer_sizes[-1][0]))

                    # Run the forward and backward pass for the output layer
                    (loss, outputs, iterations, total_loss, history), gradients = (loss_fn)(params, subkey, weights, neuron_states, layer_computation, y_encoded, jnp.zeros((batch_part_size, 1, 4)))

                    weight_grad = gradients[0]
                    # weight_grad = gather_batch(weight_grad, mpi_config, average=True)
                    weight_grad = mpi_config.combine_batch_avg(weight_grad) # Gather the weight gradients from all ranks in the same layer
                    # No sum_model_parallel: the last layer is linear, each rank owns a disjoint weight slice

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
                    
                    threshold_grad = mpi_config.gather_batch(threshold_grad, average=True) # Gather the thresholds' gradients from all ranks in the same layer

                    # weight_grad = gather_batch(weight_grad, mpi_config, average=True)
                    weight_grad = mpi_config.combine_batch_avg(weight_grad) # Gather the weight gradients from all ranks in the same layer

                    # Conv ranks replicate the full weight tensor and compute partial gradients
                    # for their owned region — sum them so every rank applies the same update.
                    # Linear layers are model-partitioned (disjoint weight slices), so no sum.
                    if empty_neuron_states.is_conv:
                        weight_grad = mpi_config.sum_model_parallel(weight_grad)
                        threshold_grad = mpi_config.sum_model_parallel(threshold_grad)

                    # Add sparsity loss' impact to the gradient if relevant
                    if params.sparsity_impact[layer_idx] > 0:
                        weight_grad = weight_grad + weight_sparsity_grad
                        threshold_grad = threshold_grad + threshold_sparsity_grad
                    
                    # Update thresholds
                    if params.threshold_lr != 0:
                        th_updates, th_opt_state = th_solver.update(threshold_grad, th_opt_state, empty_neuron_states.thresholds)
                        if params.init_thresholds != 0:
                            new_th = jax.nn.sigmoid(optax.apply_updates(
                                         jax.scipy.special.logit(empty_neuron_states.thresholds), th_updates))
                        else:
                            new_th = optax.apply_updates(empty_neuron_states.thresholds, th_updates)
                        empty_neuron_states = empty_neuron_states.replace(thresholds=new_th)
      
                if _GRAD_DUMP and i == 0:
                    os.makedirs(_GRAD_DUMP_DIR, exist_ok=True)
                    _d = {"layer_idx": np.asarray(layer_idx),
                          "weights": np.asarray(weights),
                          "weight_grad": np.asarray(weight_grad)}
                    if layer_idx == last_layer_idx:
                        _d["logits"] = np.asarray(outputs)
                        _d["targets"] = np.asarray(y)
                    np.savez(os.path.join(_GRAD_DUMP_DIR, f"layer{int(layer_idx)}.npz"), **_d)
                    comm.Barrier(); sys.exit(0)

                # Update weights
                if solver is not None:
                    # Optax optimizer
                    updates, opt_state = solver.update(weight_grad, opt_state, weights)
                    weights = jax.block_until_ready(optax.apply_updates(weights, updates))
                else:
                    # Basic GD
                    weights -= params.learning_rate * weight_grad
            # if i == 0: # Run a few epochs for testing
                # break
                # return
            valid_mask = iterations > 1
            epoch_iter_sum += float(jnp.sum(jnp.where(valid_mask, iterations, 0.0)))
            epoch_iter_count += int(jnp.sum(valid_mask))

        # Compute the average iterations for each layer
        mean = epoch_iter_sum / epoch_iter_count if epoch_iter_count > 0 else 0.0
        mean = mpi_config.gather_batch(jnp.array(mean))
        all_mean_iterations.append(float(mean))

        if layer_idx != 0:
            jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points and a mean threshold of {}", rank, mean, epoch_iter_count, jnp.mean(empty_neuron_states.thresholds))
        
        # Inference on the validation set
        val_accuracy, val_mean, _ = batch_predict(params, key, total_batches, network, weights, empty_neuron_states, layer_computation, dataset="val", save=False, debug=False)

        epoch_accuracy = 0.0
        if layer_idx == last_layer_idx:
            # Store loss values
            mean_loss = jnp.mean(jnp.array(epoch_loss))
            mean_loss = mpi_config.gather_batch(mean_loss)
            all_loss.append(float(mean_loss))

            # Store training and validation accuracies
            epoch_accuracy = epoch_correct / epoch_total
            epoch_accuracy = mpi_config.gather_batch(epoch_accuracy)
            all_epoch_accuracies.append(float(epoch_accuracy))
            all_validation_accuracies.append(float(val_accuracy))
            if rank == size-1:
                jax.debug.print("Epoch {} , Training Accuracy: {:.2f}%, Validation Accuracy: {:.2f}%, mean loss: {}, mean val iterations: {}", epoch, all_epoch_accuracies[-1] * 100, val_accuracy * 100, mean_loss, val_mean)
                jax.debug.print("----------------------------\n")
        epoch_accuracy = bcast(epoch_accuracy, root=size-1, comm=comm)
        val_accuracy_bcast = float(bcast(jnp.array([float(val_accuracy)]), root=size-1, comm=comm)[0])
        if params.use_best and val_accuracy_bcast > best_val_acc:
            best_val_acc = val_accuracy_bcast
            best_weights = weights
            best_neuron_states = empty_neuron_states
            best_epoch = epoch
        gc.collect()
        if epoch_accuracy >= 0.9999:
            break
        if STORE_EACH_EPOCH: 
            all_iteration_mean = gather_iteration_means_per_layer(all_mean_iterations)

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
    final_weights = best_weights if params.use_best else weights
    final_states = best_neuron_states if params.use_best else empty_neuron_states
    if params.use_best and mpi_config.is_last_layer_leader:
        print(f"Using best checkpoint from epoch {best_epoch} (val acc={best_val_acc:.4f})")
    test_accuracy, test_mean, _ = batch_predict(params, key, total_batches, network, final_weights, final_states, layer_computation, dataset="test", save=False, debug=False)

    all_iteration_mean = gather_iteration_means_per_layer(all_mean_iterations)

    # Synchronize all MPI processes again
    mpi4jax.barrier(comm=comm)
    end_time = time.time()

    execution_time = end_time - start_time
    if mpi_config.is_last_layer_leader:
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
        final_weights,
        final_states.thresholds,
        all_loss,
        opti,
        f"CNN/{_RUN_TAG}" if _RUN_TAG else "CNN",
        all_history,
        total_batches[0],
        extra_fields=extra_fields,
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


# def gather_iteration_means(mean_iterations):
#     leader_rank = layer_idx * processes_per_layer_global
#     save_root = last_layer_idx * processes_per_layer_global
#     all_iteration_mean = []

#     if rank == leader_rank:
#         payload = np.asarray(mean_iterations).tolist()
#         if rank == save_root:
#             collected = {layer_idx: payload}
#             for i in range(last_layer_idx):
#                 src = i * processes_per_layer_global
#                 collected[i] = comm.recv(source=src, tag=51)
#             all_iteration_mean = [collected[i] for i in range(1, last_layer_idx + 1)]
#             print("all iteration mean: rank", rank, all_iteration_mean)
#         else:
#             comm.send(payload, dest=save_root, tag=51)

#     return all_iteration_mean


def gather_iteration_means_per_layer(mean_iterations):
    """
    Each layer's leader rank sends its per-epoch mean-iteration list to the
    save_root. Returns a list-of-lists [layer_1_means, ..., layer_last_means]
    on the save_root, and an empty list on every other rank.

    Layer 0 is excluded (input layer has no meaningful iteration metric).
    """
    save_root = mpi_config.get_last_layer_batch_leader
    leader_ranks = mpi_config.all_leader_ranks
    payload = list(mean_iterations)

    if rank == save_root:
        collected = {mpi_config.layer_idx: payload}
        for layer, leader in enumerate(leader_ranks):
            if leader == rank:
                continue
            collected[layer] = comm.recv(source=leader, tag=51)
        return [collected[layer] for layer in range(1, last_layer_idx + 1)]

    if rank in leader_ranks:
        comm.send(payload, dest=save_root, tag=51)
    return []


def store_training_data_distributed(size, network, mode, all_epoch_accuracies, all_validation_accuracies, test_accuracy, execution_time, all_iteration_mean, weights, thresholds, all_loss, optiname, network_type, all_history=None, total_batches=None, extra_fields=None):
    # Linear layers are model-partitioned: reassemble the full weights/thresholds
    # across the layer's ranks before storing (conv layers replicate full arrays).
    if layer_idx != 0 and not network.layers[layer_idx].is_conv:
        weights = mpi_config.concatenate_model_partition(weights, dim=weights.ndim)
        thresholds = mpi_config.concatenate_model_partition(thresholds, dim=thresholds.ndim)

    # save_root = last_layer_idx * processes_per_layer_global
    save_root = mpi_config.get_last_layer_batch_leader
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
            extra_fields=extra_fields,
        )
        if result_path is not None:
            _write_result_json_prefix(result_path, result_data)

    result_path = comm.bcast(result_path, root=save_root)
    if result_path is None:
        return None

    comm.Barrier()

    for current_layer in range(1, last_layer_idx):
        if rank == mpi_config.all_leader_ranks[current_layer]:
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
        if rank == mpi_config.all_leader_ranks[current_layer]:
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
def batch_predict(params, key, total_batches, network, weights, empty_neuron_states, layer_computation, dataset:str="train", save=True, debug=True, readInputJson=False, extra_fields=None):
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
        if max_test_batches > 0:
            total_batches = min(total_batches, max_test_batches)
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
    for i in tqdm(range(total_batches), miniters=total_batches//10, maxinterval=float('inf'), disable=TQDM_DISABLE):
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
                batch_x, batch_y = mpi_config.split_batch(params, batch_iterator, 4)
            # store_data_to_json(f"{len(params.layer_sizes)}hidden_single_input.json", batch_x.tolist()) # Store for hardware usage
            
            # batch_x, batch_y = jnp.array([(0.0,1.0,1.0, 1.0), (0.0,2.0,2.0, 2.0), (0.0, 1.0, 0.0, 3.0), (0.0, 4.0, 4.0, 4.0), (0.0, 3.0, 3.0, 5.0), (-2, -2, -2 ,-2)]), jnp.array([1])
            # batch_x = jnp.expand_dims(batch_x, axis=0)
            # print("batch x shape, batch y shape", batch_x.shape, batch_y.shape)

            # outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, jnp.array(batch_x))
            outputs, iterations, all_neuron_states, buffer = (predict)( params, 
                                                                mpi_config,
                                                                key, 
                                                                weights, 
                                                                empty_neuron_states, 
                                                                layer_computation, 
                                                                jnp.array(batch_x),
                                                                message_size=4,
                                                                grad=False,
                                                                END_SIGNAL=END_SIGNAL,
                                                                BUFFER_SIZE=BUFFER_SIZE)
            # Send label to the last layer via plain mpi4py
            # comm.Send(np.ascontiguousarray(np.asarray(batch_y, dtype=np.float32)), dest=last_layer_idx * processes_per_layer_global + rank, tag=10)
            mpi_config.send_labels(batch_y, mpi_config.batch_first_and_last_rank[0]) # Send to the labels to the output layer

        else:
            batch_data = jnp.zeros((batch_part_size, 1, 4))

            # outputs, iterations, all_neuron_states = (conv_predict)(params, key, weights, empty_neuron_states, layer_computation, batch_data)
            outputs, iterations, all_neuron_states, buffer = (predict)( params, 
                                                                mpi_config,
                                                                key, 
                                                                weights, 
                                                                empty_neuron_states, 
                                                                layer_computation, 
                                                                batch_data,
                                                                message_size=4,
                                                                grad=False,
                                                                END_SIGNAL=END_SIGNAL,
                                                                BUFFER_SIZE=BUFFER_SIZE)
            # jax.debug.print("Rank {} All neuron states values shape: {}, output shape : {}", rank, all_neuron_states.values.shape, outputs)

            if layer_idx == last_layer_idx:
                # jax.debug.print("Rank {} All neuron states values shape: {}, output shape : {}", rank, all_neuron_states.values.shape, outputs)

                # y_buf = np.empty((batch_part_size,), dtype=np.float32)
                # comm.Recv(y_buf, source=rank - (last_layer_idx * processes_per_layer_global), tag=10)
                # y = y_buf
                y = mpi_config.recv_labels()

                # Reconstruct the full output vector from the model-partitioned last-layer ranks
                outputs = mpi_config.gather_model_partition(outputs)

                valid_y, batch_correct = accuracy(i, outputs, y, iterations, print=False)
                
                epoch_correct += int(batch_correct)
                epoch_total += valid_y.shape[0]
                # print(f"[infer:{dataset}] batch {i+1}/{total_batches}  running_acc={epoch_correct/epoch_total:.4f}", flush=True)
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
    mean = mpi_config.gather_batch(jnp.array(mean))

    if rank != 0 and debug:
        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, epoch_iter_count*mpi_config.get_process_per_batch)
    
    epoch_accuracy = -1.0
    if layer_idx == last_layer_idx:
        epoch_accuracy = epoch_correct / epoch_total
        epoch_accuracy = mpi_config.gather_batch(epoch_accuracy)
        if debug:
            jax.debug.print("Epoch Accuracy: {:.2f}%", epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")
    all_iteration_mean = gather_iteration_means_per_layer([float(mean)])

    # Synchronize all MPI processes again
    mpi4jax.barrier(comm=comm)
    end_time = time.time()

    execution_time = end_time - start_time
    # if rank == last_layer_idx * processes_per_layer_global and debug:            
    if mpi_config.is_last_layer_leader and debug:
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
            extra_fields,
        )
    return epoch_accuracy, mean, end_time - start_time

def get_layer_idx(batch_size, layer_sizes, processes_per_layer=None, split_dims=None):
    '''
    Define for each MPI rank:
    - layer_idx:            Which layer it belongs to
    - processes_per_layer_global:    How many MPI processes there are per layer
    - last_layer_idx:           The index of the last layer
    - batch_part_size:           The size of the batch each rank has to process

    processes_per_layer: if provided, use CNN model parallelism split on spatial/channel dims.
                         Must be a tuple of ints (one per layer) summing to size.
                         If None, use data parallelism (CNN_data_split).
    split_dims: tuple of strings (one per layer) e.g. ('x', 'cx', 'y', None, None, None).
                Passed through to CNN_model_split_custom. None defaults to 'x'-only split.
    '''
    global layer_idx
    global last_layer_idx
    global batch_part_size
    global mpi_config
    global processes_per_layer_global

    if processes_per_layer is not None:
        mpi_config = CNN_model_split_custom(rank, comm, size, batch_size, layer_sizes, processes_per_layer, split_dims)
    else:
        mpi_config = CNN_data_split(rank, comm, size, batch_size, layer_sizes)

    layer_idx = mpi_config.layer_idx
    last_layer_idx = mpi_config.last_layer_idx
    batch_part_size = mpi_config.batch_part.get_size

    mpi_config.print()

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
    
    processes_per_layer = config.get('processes_per_layer', None)
    if processes_per_layer is not None:
        processes_per_layer = tuple(processes_per_layer)
        if sum(processes_per_layer) != size:
            print(f"Error: sum of processes_per_layer ({sum(processes_per_layer)}) must equal MPI size ({size})")
            sys.exit(1)
    else:
        if size % len(layer_sizes) != 0:
            print(f"Error: layer_sizes ({len(layer_sizes)}) must match number of MPI ranks ({size})")
            sys.exit(1)

    split_dims = config.get('split_dims', None)
    if split_dims is not None:
        split_dims = tuple(split_dims)

    get_layer_idx(batch_size, layer_sizes, processes_per_layer, split_dims)
    
    if batch_size % mpi_config.get_process_per_batch != 0:
        print(f"Error: one batch ({batch_size}) must be divisible by the number of processes per layer ({mpi_config.get_process_per_batch})")
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
                    loader = partial(torch_nmnist_loader, first_saccade_only=config['first_saccade_only'])
                case "dvs":
                    loader = partial(torch_DVSGesture_loader)
                    if layer_sizes[0][1] == 64:
                        downsample = True
                case "ncars":
                    loader = partial(torch_NCARS_loader, dedup=config.get('dedup', False), augment=config.get('augment', False))
                    if tuple(layer_sizes[0][1:]) == (60, 50):
                        downsample = True
                case "cifar10":
                    loader = partial(cifar10_loader_manual, augment=config.get('augment', False))
                    if layer_sizes[0][1] == 16:
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
            rerun=None,
            top_weights=config['top_weights'],
            max_kernel=max_kernel,
            flat_layer_sizes=(),            # Each layer's shape
            history_size=config['history_size'],                 # How many output states should we keep for plotting output history
            output_decay=config.get('output_decay', 1.0),       # Per-event weight decay at output layer
            dedup=config.get('dedup', False),
            augment=config.get('augment', False),
            use_best=config.get('use_best', False),
            dropout=tuple(config['dropout']) if config.get('dropout') is not None else None,
            dropout_invert_scaling=config.get('dropout_invert_scaling', False),
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

            # The first build used the config's sync_rate; rebuild with the resolved
            # params so params-derived neuron state (notably sync_rate_vector) reflects
            # the checkpoint/override values rather than the config.
            network = Network.build(params, key, layer_sizes=layer_sizes,
                                    flat_layer_sizes=(), conv_layer_sizes=(),
                                    th_bias=0.0)
            empty_neuron_states = network.layers[layer_idx]

            if layer_idx > 0:
                empty_neuron_states = network.rerun(thresholds)

        # Slice linear-layer weights and neuron states to this rank's model partition,
        # mirroring async_MLP_general. Conv layers keep full arrays and mask their owned region
        # inside conv_layer_computation instead.
        if layer_idx != 0 and not empty_neuron_states.is_conv:
            weights, empty_neuron_states = mpi_config.MPI_partition(weights, empty_neuron_states)

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

        mode = config['mode']
        extra_fields = {"processes_per_layer": list(processes_per_layer) if processes_per_layer else None,
                        "split_dims": list(split_dims) if split_dims else None}
        if dataset == "nmnist":
            extra_fields["first_saccade_only"] = config['first_saccade_only']
        if mode == 'inference':
            # To only run inference
            batch_predict(params, key, total_batches, network, weights, empty_neuron_states, layer_computation, 'test', save=True, debug=True,
                          extra_fields=extra_fields)
        elif mode == 'training':
            # To run the full training pipeline
            result_path = train(params, key, total_batches, network, weights, empty_neuron_states, layer_computation, "adam",
                               extra_fields=extra_fields)
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
    parser.add_argument('--debug-level', type=int, default=0,
                       help='Debug level for bug tests in inference.py (0=off, 1=BT1, 2=BT2, 3=BT3)')
    parser.add_argument('--max-test-batches', type=int, default=0,
                       help='Cap number of test batches (0 = no limit, for quick BT runs)')
    parser.add_argument('--exp3-ordered-recv', action='store_true', default=False,
                       help='EXP3: force serialized per-sender receives instead of MPI.ANY_SOURCE')
    args, unknown = parser.parse_known_args()

    random_seed = args.seed
    key = jax.random.key(random_seed)

    import forward_backward_pass.inference as _inf_mod
    _inf_mod._DEBUG_LEVEL = args.debug_level
    _inf_mod._EXP3_ORDERED_RECV = args.exp3_ordered_recv
    max_test_batches = args.max_test_batches

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()      # Real rank
    size = comm.Get_size()

    main(random_seed, key, rank, size, comm, config_path=args.config, data_dir=args.data_dir)

'''
JAX_PLATFORMS=cpu mpirun -n 5 python async_CNN_general.py --config "configs/CNN_config.yaml"
'''
