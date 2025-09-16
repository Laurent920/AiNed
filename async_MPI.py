from ast import Load
import os
os.environ["JAX_PLATFORMS"] = "cpu"

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
from flax.struct import dataclass
from flax.core import FrozenDict

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
from data_helpers.shd_helper import torch_SHD_loader
from data_helpers.iris_species_helper import torch_iris_loader
from data_helpers.network_helper import one_hot_encode

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
    def __init__(self, values, thresholds, input_residuals, input_order, input_activity, layer_activity, output_activity, last_sent_iteration, seen_mask, activated_mask):
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
        self.seen_mask = seen_mask
        self.activated_mask = activated_mask

    # Tell JAX how to flatten this object
    def tree_flatten(self):
        children = (self.values, self.thresholds, self.input_residuals,
                    self.input_order, self.input_activity, self.layer_activity,
                    self.output_activity, self.last_sent_iteration, 
                    self.seen_mask, self.activated_mask)
        aux_data = None  # no extra static data
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

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
    shuffle_input:bool      # Shuffle the data in each layer to simulate async individual neurons
    threshold_lr: float
    sparsity_impact: float
    rerun: str
    async_layer: int # The layer that is training asynchronously while all other layers are training sync, if -1 then all layers are async

# region INFERENCE
@custom_jvp # If thresholds == 0 then this behaves as a ReLu activation function 
def activation_func(neuron_states, activations):
    # return jax.nn.relu(activations)
    return jnp.where(activations > neuron_states.thresholds, activations, 0.0)

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

@jax.jit
def sparse_two_mask_update(seen_mask, activated_mask, input_idx, output):
    """
    Computes the residuals for weights update in the backpropagation
    """
    # output: shape (output_shape,)
    # mark row as seen (all zeros become seen). Setting whole row is ok:
    seen_mask = seen_mask.at[input_idx, :].set(True)

    # indices of active columns (with filler -1)
    active_cols = jnp.nonzero(output, size=output.shape[0], fill_value=-1)[0]
    mask = active_cols != -1  # boolean mask for valid cols

    def no_act(arg):
        return arg

    def do_act(arg):
        s_mask, a_mask = arg

        # gather all cols (with padding)
        cols = active_cols
        s_block = s_mask[:, cols]    # shape (input_shape, output_shape)
        a_block = a_mask[:, cols]

        # only update where cols are valid
        new_a_block = jnp.where(mask, a_block | s_block, a_block)
        a_mask = a_mask.at[:, cols].set(new_a_block)

        return (s_mask, a_mask)

    seen_mask, activated_mask = jax.lax.cond(
        jnp.any(mask),
        do_act,
        no_act,
        operand=(seen_mask, activated_mask)
    )

    return seen_mask, activated_mask


@partial(jax.jit, static_argnames=['params'])
def process_activated_output(key, arr: jnp.ndarray, params):
    '''
    Processed the output of a layer from (1d array) to (2d array) with [(neuron idx, value)]
    value == 0 are filled with index==-2
    '''
    max_len = params.layer_sizes[split_rank]

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

    return pairs_out

@partial(jax.jit, static_argnames=['params'])
def layer_computation(params, key, neuron_idx, layer_input, weights, neuron_states, iteration=0):    
    # activations = jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values
    activations = jax.lax.cond(neuron_idx < 0,
                            lambda _: neuron_states.values,
                            lambda _: jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values,
                            None
                            )
    #TODO being able to compute multiple incoming index neurons
    #TODO store the weight residuals of last layer and neuron state of input layer in sparse matrix representation to reduce space utilization because unused
    
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
        # dummy_activations = jnp.zeros((activations.shape[0]))
        dummy_activations = jnp.zeros((activations.shape[0], 2))
        return jnp.array(0), dummy_activations, NeuronStates(
                                            values=activations, 
                                            thresholds=neuron_states.thresholds, 
                                            input_residuals=new_input_residuals, 
                                            input_order=neuron_states.input_order, 
                                            input_activity=new_input_activity,
                                            layer_activity=neuron_states.layer_activity,
                                            output_activity=neuron_states.output_activity,
                                            last_sent_iteration=neuron_states.last_sent_iteration,
                                            seen_mask=neuron_states.seen_mask,
                                            activated_mask=neuron_states.activated_mask)
    
    def hidden_layer_case(_):
        fire = (iteration-neuron_states.last_sent_iteration) >= params.sync_rate # Fire if sync rate reached
        async_fire = jnp.logical_or(params.async_layer < 0, split_rank <= params.async_layer) # Fire if async_layer or no async_layer condition (-1)
        fire = jnp.logical_and(fire, async_fire) 
        fire = jnp.logical_or(fire, neuron_idx < 0) # Fire if last input received

        # APPLY THE SYNC RATE  
        activated_output = jax.lax.cond(fire, 
                                        lambda args: activation_func(args[0], args[1]), 
                                        lambda _: jnp.zeros(activations.shape),
                                        (neuron_states, activations))
        
        # APPLY THE FIRING NUMBER        
        activated_output = keep_top_k(activated_output, params.firing_nb) # Get the top k activations
        # jax.debug.print("{}, iteration: {}, neuron idx: {}", activated_output, iteration, neuron_idx)


        layer_activity = neuron_states.layer_activity
        
        # APPLY THE RESTRICTION
        penalty = jax.lax.cond(params.restrict[split_rank] <= 0, 
                               lambda _: jnp.zeros(activated_output.shape), 
                               lambda _: activated_output*params.restrict[split_rank], None)
        
        # Store the neurons that activated
        active_indexes = jnp.where(activated_output > 0, 1, 0)
        new_layer_activities = layer_activity + active_indexes # Update the layer activity by adding the active neurons
        
        
        last_neuron_idx = jnp.argmax(neuron_states.input_order) # Last neuron index in the input order
        new_neuron_idx = jax.lax.cond(neuron_idx < 0,
                     lambda _: last_neuron_idx, 
                     lambda _: neuron_idx,
                     None)
        
        new_input_order = neuron_states.input_order.at[new_neuron_idx].set(iteration) # Update the input activity by setting the input neuron to the iteration number        
        
        # jax.debug.print("{} {}", active_indexes.shape, new_input_activities.shape)
        new_output_activity = neuron_states.output_activity.at[new_neuron_idx].add(active_indexes)
        
        new_last_sent_iteration = jax.lax.cond(fire, lambda _: iteration, lambda _: neuron_states.last_sent_iteration, None)

        new_values = activations - activated_output - penalty
        # jax.debug.print("Rank {}, neuron idx: {}, new values: {}, active output: {}, penalty: {}, iteration: {}", rank, neuron_idx, new_values, active_output, penalty, iteration)
        
        valid_elements = jnp.count_nonzero(activated_output)
        processed_output = process_activated_output(key, activated_output, params)
        # new_seen_mask, new_activated_mask = sparse_two_mask_update(neuron_states.seen_mask, neuron_states.activated_mask, neuron_idx, activated_output)
        new_seen_mask, new_activated_mask = neuron_states.seen_mask, neuron_states.activated_mask

        new_neuron_states = NeuronStates(   values=new_values, 
                                            thresholds=neuron_states.thresholds, 
                                            input_residuals=new_input_residuals, 
                                            input_order=new_input_order, 
                                            input_activity=new_input_activity,
                                            layer_activity=new_layer_activities,
                                            output_activity=new_output_activity,
                                            last_sent_iteration=new_last_sent_iteration,
                                            seen_mask=new_seen_mask,
                                            activated_mask=new_activated_mask)

        return valid_elements, processed_output, new_neuron_states
    
    cond = split_rank == last_rank #jnp.logical_or(split_rank == last_rank, neuron_idx < 0)
    return jax.lax.cond(cond, last_layer_case, hidden_layer_case, None)
    
    jax.lax.cond(cond, last_layer_case, hidden_layer_case, None)
    # TEST MPI WITH CONTROLLED NUMBER OF ACTIVATIONS
    def first_hidden(activations):
        return jnp.ones(activations.shape), neuron_states
    
    def other_hidden(activations):
        half_ones = jnp.ones(1)  # half ones
        half_zeros = jnp.zeros(activations.shape[0]-1)  # half zeros

        # Concatenate them
        arr = jnp.concatenate([half_ones, half_zeros])
        return arr, neuron_states
    
    return jax.lax.cond(rank == 1, first_hidden, other_hidden, (activations))


@partial(jax.jit, static_argnames=['params'])
def predict(params, key, weights, empty_neuron_states, batch_data: jnp.ndarray):
    #region JAX loop
    def input_layer(args):
        neuron_states, x = args # x is shape (input_layer_size,)
        
        x_p = jnp.array(x)
        if params.shuffle_input:
            perm = jax.random.permutation(key, x_p.shape[0])
            x_p = x_p[perm]
            
        def send_input(i, carry):
            count = carry
            data = x_p[i]
            send(data, dest=rank+process_per_layer, tag=0, comm=comm)
            return i

        def first_not_minus2(row):
            return (row != -2)
        mask = jax.vmap(first_not_minus2)(x_p)
        loop_iterations = (jnp.count_nonzero(mask)/2).astype(int)
        # loop_iterations = x_p.shape[0]
        # jax.debug.print("input data type {}, {} ", (loop_iterations), len(x_p))

        iteration = jax.lax.fori_loop(0, loop_iterations, send_input, (0))

        # Send end signal
        send(jnp.array([-1.0, 0.0]), dest=rank+process_per_layer, tag=0, comm=comm)

        return jnp.zeros(()), neuron_states, iteration
    
    def other_layers(args):
        neuron_states, _ = args
        def cond(state): # end of input has been reached -> break the while loop
            _, _, neuron_idx, _= state            
            return neuron_idx != -1
        
        def forward_pass(state):
            layer_input, neuron_states, neuron_idx, iteration = state
            
            def hidden_layers(args): # Send activation to the next layers
                loop_iterations, activated_output = args
                # jax.debug.print("activated output shape: {}, {}", activated_output.shape, activated_output[:, 0])
                # loop_iterations = jnp.count_nonzero(input)
                # activated_output = process_activated_output(key, input, params)

                def send_activation(i, _):
                    out_val = activated_output[i]
                    send(out_val, dest=rank+process_per_layer, tag=0, comm=comm)
                    return None
                jax.lax.fori_loop(0, loop_iterations, send_activation, None)
                
                return None
            
            # Receive neuron values from previous layers and compute the activations
            (neuron_idx, layer_input) = recv(jnp.zeros((2,)), source=rank-process_per_layer, tag=0, comm=comm)
            loop_iterations, activated_output, new_neuron_states= layer_computation(params, key, neuron_idx.astype(int), layer_input, weights, neuron_states, iteration)
            
            neuron_states = new_neuron_states
            
            jax.lax.cond(split_rank == last_rank, lambda _: None, hidden_layers, (loop_iterations, activated_output)) # Don't send if we reach the last layer
            return layer_input, neuron_states, neuron_idx, iteration+1
        
        neuron_idx = 0
        layer_input = jnp.zeros(())
        initial_state = (layer_input, neuron_states, neuron_idx, 0)
        
        # Loop until the rank receives a -1 neuron_idx
        layer_input, neuron_states, neuron_idx, iteration = jax.lax.while_loop(cond, forward_pass, initial_state)
        
        # Send -1 to the next rank when all incoming data has been processed
        jax.lax.cond(
            split_rank != last_rank,
            lambda _: send(jnp.array([-1.0, 0.0]), dest=rank + process_per_layer, tag=0, comm=comm),
            lambda _: [],
            operand=None
        )
        return layer_input, neuron_states, iteration-1

    # jax.debug.print("rank {} data has shape {}", rank, batch_data.shape)

    # Loop over batches, accumulate output values and return them
    def loop_over_batches(_, x):
        neuron_states = empty_neuron_states  
        layer_input, new_neuron_states, iterations = jax.lax.cond(split_rank==0, input_layer, other_layers, (neuron_states, x))
        
        return None, (new_neuron_states.values, iterations, new_neuron_states)
    
    _, (all_outputs, all_iterations, all_neuron_states) = jax.lax.scan(loop_over_batches, None, batch_data)    
    
    # Synchronize all ranks before starting the backward pass
    mpi4jax.barrier(comm=comm)

    return all_outputs, all_iterations, all_neuron_states

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
def compute_w_residuals(input_order, output_activity):
    '''
    Compute the weights that activated and need to be updated by taking into account previous timesteps influence
    input_order: shape (784, ) containing last iteration number or -1 if never fired
    output_activity:   shape (784, 128)
    '''
    # Preprocess the input activity by computing the ordering of the indices
    activity_ordered = jnp.argsort(input_order)
    
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
            input_order[j]>-1,
            update_if_active_fn,
            lambda carry: carry,
            operand=(activates, output_activity)
        )

    # Initial state
    activates = jnp.zeros((output_activity.shape[1],), dtype=jnp.int32) #(128,)

    # Reverse loop with fori_loop
    n = input_order.shape[0] # 784
    activates, output_activity = jax.lax.fori_loop(
        0, jnp.sum(input_order!=-1), # Don't loop over the non relevant values
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

@partial(jax.jit, static_argnames=['params'])
def predict_bwd(params, key, weights, empty_neuron_states, batch_data):
    '''
    B: batch_size
    '''
    all_outputs, iterations, all_neuron_states = (predict)(params, key, weights, empty_neuron_states, batch_data)
    next_grad = recv(jnp.zeros((batch_part, layer_sizes[split_rank])), source=rank + process_per_layer, tag=2, comm=comm) # Shape: (B, 128)
    # jax.debug.print("Rank {} received next_grad shape: {}", rank, next_grad)
    
    # "input order": Shape (B, 784, 1), "output activity": Shape (B, 784, 128)
    weight_res = jax.vmap(compute_w_residuals, in_axes=(0, 0))(all_neuron_states.input_order, all_neuron_states.output_activity) # Shape: (B, 784, 128)

    # weight_res = all_neuron_states.activated_output # incorrect residual but faster for testing
    # jax.debug.print("Rank {} weight_res shape: {}, weight_res max: {}", rank, weight_res.shape, jnp.max(weight_res))

    
    next_weight_res = jnp.ones((batch_part, params.layer_sizes[split_rank], params.layer_sizes[split_rank+1])) # Shape: (B, 128, 10)
    # jax.debug.print("Rank {} received next_grad shape: {}, next_weight_res shape: {}", rank, next_grad.shape, next_weight_res.shape)
    (next_weight_res) = jax.lax.cond(split_rank < last_rank - 1, 
                                   lambda _: recv(next_weight_res, source=rank + process_per_layer, tag=3, comm=comm),
                                   lambda _: (next_weight_res), None) 
    # jax.debug.print("Rank {} received next_grad shape: {}", rank, next_weight_res)

    weight_res = jax.lax.cond(split_rank < last_rank - 1,
                                lambda args: jax.vmap(recompute_w_residuals, in_axes=(0, 0))(args[0], args[1]), # Shape: (B, 784, 128)
                                lambda _: weight_res,
                                (weight_res, next_weight_res))    
    
    weight_res = jax.vmap(apply_restrict_to_residuals, in_axes=(None, 0, 0))(params, weight_res, all_neuron_states.layer_activity)
    
    # Perform element-wise multiplication
    z_grad = jax.vmap(z_gradient, in_axes=(0, 0))(weight_res, next_grad) # Shape: (B, 784, 128)
    
    # jax.debug.print("Rank {}, input activity shape: {}, input activity: {}", rank, input_activity.shape, jnp.sum(input_activity > 1))
    x = all_neuron_states.input_residuals #/ jnp.where(input_activity == 0, 1.0, input_activity)# Shape (B, 784)
    x_reshaped = x[..., jnp.newaxis]   # Shape becomes (B, 784, 1)
    
    weight_grad = x_reshaped * z_grad # (B, 784, 128)
    
    # jax.debug.print("weight_grad: {}, x: {}, z_grad: {}, next_grad_expanded: {}, weight_res: {}", jnp.isnan(weight_grad).any(), jnp.isnan(x).any(), jnp.isnan(z_grad).any(), jnp.isnan(next_grad_expanded).any(), jnp.isnan(weight_res).any())
    mean_weight_grad = jnp.mean(weight_grad, axis=0) # (784, 128)

    # jax.debug.print("x {}, x_reshaped{}", x.shape, x)
    # jax.debug.print("next_grad_expanded {}, {}", next_grad_expanded.shape, next_grad_expanded)
    # jax.debug.print("weight residuals {}, {}", weight_res.shape, weight_res)
    # jax.debug.print("z_grad {}, {}", z_grad.shape, z_grad)
    # jax.debug.print("weight_grad {}, mean_weight_grad{}", weight_grad.shape, mean_weight_grad.shape)

    layer_activity = jnp.where(all_neuron_states.layer_activity > 0, 1, 0)
    th_grad = -jnp.mean(next_grad * layer_activity, axis=0)  # Shape: (128)
    thresholds = empty_neuron_states.thresholds
    before_th_grad = th_grad
    th_grad = th_grad * thresholds * (thresholds - 1)
    # jax.debug.print("Rank {}, before th_grad shape: {}, th_grad: {}, next grad {}", rank, before_th_grad, th_grad, next_grad) # Shape: (128,)
    # jax.debug.print("rank {}, weight grad {}", rank, jnp.mean(th_grad))
    
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
    
    return all_outputs, iterations, all_neuron_states, (weight_grad, th_grad, weight_sparsity_grad, th_sparsity_grad, weight_res) 

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

@partial(jax.jit, static_argnames=['params'])
def loss_fn(params, key, weights, empty_neuron_states, target, batch_data):
    all_outputs, iterations, all_neuron_states = (predict)(params, key, weights, empty_neuron_states, batch_data)
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
    send(out_grad, dest=rank-process_per_layer, tag=2,comm=comm)
    
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)

    total_loss = loss + sparsity_L 

    return (loss, all_outputs, iterations, total_loss), (out_grad, weight_grad, loss_grad, weight_grad)

def sparsity_loss(params, all_neuron_states, iterations):
    '''
    Compute the sparsity loss based on the input residuals and the weight residuals
    '''
    if all(x <= 0.0 for x in params.sparsity_impact):
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
            all_activations = all_activations + (params.sparsity_impact[i] * act_sum[0]) # Sum of all activations in the hidden layers
            
            if i == 0: # Get iterations of input data
                iter_mean = recv(jnp.zeros(1), source=i * process_per_layer, tag=6, comm=comm)
                all_iterations = iter_mean[0]
        all_activations += params.sparsity_impact[split_rank] * jnp.sum(activations) # Adding the activations of the last layer

        sparsity_L = all_activations /  (all_iterations * batch_part * process_per_layer)
        # jax.debug.print("Rank {}, sparsity L: {}, all iterations: {}, all activations: {}", rank, sparsity_L, all_iterations, all_activations)
    all_iterations = bcast(all_iterations, root=last_rank*process_per_layer, comm=comm)

    return all_activations, all_iterations, sparsity_L

def share_split_rank_data(data):
    data = jnp.array(data)
    leader_rank = split_rank * process_per_layer
    if rank == leader_rank:
        for i in range(process_per_layer-1): # Sharing the data to all the corresponding ranks
            send(data, dest=rank+i+1, tag=20, comm=comm)
    else:
        data = recv(data, source=leader_rank, tag=20, comm=comm)        
    return data

def split_batch(params, batch_iterator):
    if rank == 0:
        all_batch_x, all_batch_y = next(batch_iterator)
        # print(f"rank {rank} before split batch data has shape {(all_batch_x.shape)}, {(all_batch_y.shape)}")                

        all_batch_y = jnp.array(all_batch_y, dtype=jnp.float32)
        all_batch_x = jnp.array(all_batch_x, dtype=jnp.float32)
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
        # batch_x = recv(jnp.zeros((batch_part, layer_sizes[0])), source=0, tag=4, comm=comm)  
        batch_x = recv(jnp.zeros((batch_part, params.max_nonzero, 2)), source=0, tag=4, comm=comm)  
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
            
    leader_rank = split_rank * process_per_layer
    if rank == leader_rank:
        avg = data
        for i in range(0, process_per_layer-1): # Receive the data from all the corresponding ranks in one split rank
            received_data = recv(data, source=rank+i+1, tag=20, comm=comm)
            avg = jnp.concatenate([avg, received_data], axis=0)            
        if average:
            # print(f"Rank {rank} combining batches, avg shape: {avg.shape}")
            avg = jnp.mean(avg, axis=0)

        for i in range(process_per_layer-1): # Resharing the average data to all the corresponding ranks
            send(avg, dest=rank+i+1, tag=20, comm=comm)
    else:
        send(data, dest=leader_rank, tag=20, comm=comm)
        avg = recv(jnp.zeros((data.shape[1], data.shape[2])), source=leader_rank, tag=20, comm=comm)
        
    return avg


# region TRAINING
def train(params: Params, key, weights, empty_neuron_states, opti):     
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
        print("adamw optimizer selected")
        solver = optax.adam(learning_rate=params.learning_rate)
    elif opti == "sgd":
        print("sgd optimizer selected")
        solver = optax.sgd(learning_rate=params.learning_rate)
    elif opti == "rmsprop":
        print("rmsprop optimizer selected")
        solver = optax.rmsprop(learning_rate=params.learning_rate, decay=0.9, eps=1e-8)
    elif opti == "amsgrad":
        print("amsgrad optimizer selected")
        solver = optax.amsgrad(learning_rate=params.learning_rate)
    elif opti == "lion":
        print("lion optimizer selected")
        solver = optax.lion(learning_rate=params.learning_rate)
    else: 
        solver = None
    if solver is not None:
        opt_state = solver.init(weights)
    
    th_solver = optax.adam(learning_rate=params.threshold_lr)
    th_opt_state = th_solver.init(jax.scipy.special.logit(empty_neuron_states.thresholds))
    
    mpi4jax.barrier(comm=comm)
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
        # print("epoch ", epoch)
        for i in range(total_train_batches):
            neuron_states = empty_neuron_states
            # threshold_grad = 0.0
            if split_rank == 0:
                # print(i)
                batch_x, batch_y = split_batch(params, batch_iterator)
                # print(f"rank {rank} data has shape {type(batch_x)}, {type(batch_y)}")
                # return None

                send(batch_y, dest=last_rank * process_per_layer + rank, tag=10,comm=comm) # Destination rank: last_rank * process_per_layer + rank
                outputs, iterations, all_neuron_states = (predict)(params, subkey, weights, neuron_states, batch_data=jnp.array(batch_x))
                all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
            else:
                # return None
                if split_rank==last_rank:
                    # Receive y
                    y = recv(jnp.zeros((batch_part,)), source=rank - (last_rank * process_per_layer), tag=10, comm=comm)  # Source rank opposite operation: rank - (last_rank * process_per_layer)
                    y_encoded = jnp.array(one_hot_encode(y, num_classes=layer_sizes[-1]))
                    # print("encoded y: ", y, y_encoded.shape, y_encoded)              
                    (loss, outputs, iterations, total_loss), gradients = (loss_fn)(params, subkey, weights, neuron_states, y_encoded, jnp.zeros((batch_part, layer_sizes[0])))

                    epoch_loss.append(loss)
                    
                    weight_grad = gradients[1]
                                        
                    valid_y, batch_correct = accuracy(i, outputs, y, iterations, False)                 
                        
                    epoch_correct += batch_correct
                    epoch_total += valid_y.shape[0]
                    # weight_grad = gather_batch(weight_grad, average=True)
                    weight_grad = combine_batch(weight_grad, average=True) # Gather the weight gradients from all ranks in the split rank
                else:
                    outputs, iterations, all_neuron_states, grads = (predict_bwd)(params, subkey, weights, neuron_states, jnp.zeros((batch_part, layer_sizes[0])))
                    weight_grad, threshold_grad, weight_sparsity_grad, threshold_sparsity_grad, weight_res = grads
                    # print(f"rank {rank}, weight_res: {weight_res[0].tolist()}, shape: {weight_res.shape}")

                    # print(f"Rank {rank} finished predict_bwd for batch {i}, outputs shape: {outputs.shape}, iterations: {iterations.shape}, weight_grad shape: {weight_grad.shape}")
                    threshold_grad = gather_batch(threshold_grad, average=True) # Gather the weight gradients from all ranks in the split rank

                    # weight_grad = gather_batch(weight_grad, average=True)
                    weight_grad = combine_batch(weight_grad, average=True) # Gather the weight gradients from all ranks in the split rank
                    
                    if jnp.any(jnp.array(params.sparsity_impact) > 0):
                        weight_grad = weight_grad + weight_sparsity_grad
                        threshold_grad = threshold_grad + threshold_sparsity_grad
                    # Update thresholds
                    # print(f"new thresholds: {jnp.mean(empty_neuron_states.thresholds)}, threshold_grad: {jnp.mean(threshold_grad)}")
                    # empty_neuron_states.thresholds = jax.nn.sigmoid(empty_neuron_states.thresholds - (threshold_grad * params.threshold_lr))
                    
                    if params.threshold_lr != 0:
                        # print(f"average threshold grad: {jnp.mean(threshold_grad)}")
                        th_updates, th_opt_state = solver.update(threshold_grad, th_opt_state, empty_neuron_states.thresholds)
                        empty_neuron_states = dataclasses.replace(empty_neuron_states,
                                                                  thresholds=jax.nn.sigmoid(
                                                                    optax.apply_updates(
                                                                        jax.scipy.special.logit(empty_neuron_states.thresholds), th_updates))
                                                                 )                    
                        # print(empty_neuron_states.thresholds)
                
                # print("Rank {}, batch {}, mean weight_grad: {}, max weight_grad: {}, min weight_grad: {}".format(rank, i, jnp.mean(weight_grad), jnp.max(weight_grad), jnp.min(weight_grad)))
                # Update weights
                if solver is not None and (params.async_layer < 0 or split_rank == params.async_layer):
                    # Optax optimizer
                    updates, opt_state = solver.update(weight_grad, opt_state, weights)
                    weights = optax.apply_updates(weights, updates)
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
        val_accuracy, val_mean, _ = batch_predict(params, key, weights, empty_neuron_states, dataset="val", save=False, debug=False)
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
    test_accuracy, test_mean, _ = batch_predict(params, key, weights, empty_neuron_states, dataset="test", save=True, debug=True)
    # test_accuracy = 0
    # Gather the weights and iteration values at the last layer
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_iter_th(weights, jnp.array(all_mean_iterations), empty_neuron_states.thresholds)
    
    # Synchronize all MPI processes again
    mpi4jax.barrier(comm=comm)
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
    result_path = bcast(result_path, root=last_rank*process_per_layer, comm=comm)
    result_path = bytes(result_path).decode("utf-8").rstrip("\x00")
    mpi4jax.barrier(comm=comm)

    return result_path
    
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
        "shuffle activations": params.shuffle_activations,
        "shuffle input": params.shuffle_input,
        "rerun": params.rerun,
        "processes": size,
        "firing number": params.firing_nb,
        "synchronization rate": params.sync_rate,
        "async layer": params.async_layer,
        "restrict": params.restrict,
        "sparsity impact": params.sparsity_impact,
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
    
#region Initialization
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = jax.random.split(key)
    return scale * jax.random.normal(w_key, (n, m))#, scale * jax.random.normal(b_key, (n,))
    # return jnp.full((n, m), 0.1)

def init_params(key, load_file=False, best=False):
    # Initialize weights for each layer
    keys = jax.random.split(key, len(layer_sizes))
    
    if split_rank != 0:
        neuron_states = NeuronStates(   values=jnp.zeros(layer_sizes[split_rank]),
                                        thresholds=jnp.full((layer_sizes[split_rank]), init_thresholds),
                                        input_residuals=np.zeros((layer_sizes[split_rank-1],)),
                                        input_order=jnp.full((layer_sizes[split_rank-1],), -1, dtype=int), 
                                        input_activity=jnp.full((layer_sizes[split_rank-1],), 0, dtype=int),
                                        layer_activity=jnp.zeros((layer_sizes[split_rank],), dtype=int),
                                        output_activity=jnp.zeros((layer_sizes[split_rank-1], layer_sizes[split_rank])),
                                        last_sent_iteration=0,
                                        seen_mask=jnp.zeros((layer_sizes[split_rank-1], layer_sizes[split_rank]), dtype=bool),
                                        activated_mask=jnp.zeros((layer_sizes[split_rank-1], layer_sizes[split_rank]), dtype=bool)
                                        )
        if load_file:
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
                    return weights, neuron_states      
        
        # Random initializatoin of the weights       
        weights = random_layer_params(layer_sizes[split_rank], layer_sizes[split_rank-1], keys[split_rank])        
        return weights, neuron_states


def pad_batch(batch_x, batch_y, batch_size):
    # Pad the x data with 0 and the y data with nan for the last batch
    current_size = batch_y.shape[0]
    if current_size < batch_size:
        pad_amount = batch_size - current_size
        pad_y = jnp.full((pad_amount,), -1.0, dtype=jnp.float32)
        pad_x = jnp.zeros((pad_amount,) + batch_x.shape[1:], dtype=batch_x.dtype)

        batch_y = jnp.concatenate([batch_y, pad_y], axis=0)
        batch_x = jnp.concatenate([batch_x, pad_x], axis=0)
    
    return batch_x, batch_y


def gather_w_iter_th(weights, mean_iterations, thresholds):
    # Gather all the weights and iteration values at the last layer to store them
    leader_rank = split_rank * process_per_layer

    weights_dict = {}
    all_iteration_mean = []
    thresholds_dict = {}
    
    # print(rank, thresholds.shape, mean_iterations)
    if split_rank != last_rank and rank == leader_rank:
        send(weights, dest=last_rank * process_per_layer, tag=5,comm=comm)
        send(mean_iterations, dest=last_rank * process_per_layer, tag=5,comm=comm)
        send(thresholds, dest=last_rank * process_per_layer, tag=5,comm=comm)
    elif split_rank == last_rank and rank == leader_rank:
        for i in range(last_rank):
            # Storing the weights 
            w = recv(jnp.zeros((layer_sizes[i-1], layer_sizes[i])), source=i * process_per_layer, tag=5, comm=comm)   
            weights_dict[f"layer_{i}"] = w.tolist()
            
            # Storing mean iterations
            iter_mean = recv(mean_iterations, source=i * process_per_layer, tag=5, comm=comm)
            all_iteration_mean.append(iter_mean)
            
            # Storing the thresholds
            thr = recv(jnp.zeros(layer_sizes[i]), source=i * process_per_layer, tag=5, comm=comm)
            if i == 0: continue  # Skip the input layer thresholds
            thresholds_dict[f"thresholds_{i}"]= thr.tolist()
            
        all_iteration_mean.append(mean_iterations)  # Append the mean iterations of the last layer
        weights_dict[f"layer_{last_rank}"] = weights.tolist()
        all_iteration_mean = all_iteration_mean[1:] # Don't keep the value of the input layer
        print("all iteration mean: rank", rank, all_iteration_mean)

    return weights_dict, all_iteration_mean, thresholds_dict

def create_runtime_video(runtimes_dict, activations_dict, output_file="runtimes_animation.mp4"):
    ranks = list(runtimes_dict.keys())
    num_iters = max(len(v) for v in runtimes_dict.values())
    num_ranks = len(ranks)

    fig, ax_runtime = plt.subplots()

    # Secondary axis for activations
    ax_activation = ax_runtime.twinx()

    # Scatter plots for both datasets
    scat_runtime = ax_runtime.scatter([], [], s=50, color="blue", label="Runtime")
    scat_activation = ax_activation.scatter([], [], s=50, color="red", label="Activation")

    # X-axis setup
    ax_runtime.set_xlim(-0.5, num_ranks - 0.5)
    ax_runtime.set_xticks(range(num_ranks))
    ax_runtime.set_xticklabels(ranks)

    # Y-axis limits
    ax_runtime.set_ylim(
        min(min(v) for v in runtimes_dict.values()),
        max(max(v) for v in runtimes_dict.values())
    )
    ax_activation.set_ylim(
        min(min(v) for v in activations_dict.values()),
        max(max(v) for v in activations_dict.values())
    )

    # Labels and titles
    ax_runtime.set_xlabel("Rank")
    ax_runtime.set_ylabel("Runtime (seconds)", color="blue")
    ax_activation.set_ylabel("Activation", color="red")
    ax_runtime.set_title("Per-rank runtimes & activations over iterations")
    ax_runtime.grid(True)

    # Legends
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax_runtime.transAxes)

    def init():
        scat_runtime.set_offsets(np.empty((0, 2)))
        scat_activation.set_offsets(np.empty((0, 2)))
        return scat_runtime, scat_activation

    def update(frame):
        x = np.arange(num_ranks)

        # Runtimes
        y_runtime = [runtimes_dict[r][frame] if frame < len(runtimes_dict[r]) else np.nan for r in ranks]
        points_runtime = np.column_stack((x, y_runtime))
        scat_runtime.set_offsets(points_runtime)

        # Activations
        y_activation = [activations_dict[r][frame] if frame < len(activations_dict[r]) else np.nan for r in ranks]
        points_activation = np.column_stack((x, y_activation))
        scat_activation.set_offsets(points_activation)

        ax_runtime.set_title(f"Runtimes & Activations — Iteration {frame}")
        return scat_runtime, scat_activation

    ani = animation.FuncAnimation(fig, update, frames=num_iters, init_func=init, blit=True, interval=10)
    ani.save(output_file, fps=5, extra_args=['-vcodec', 'libx264'])
    plt.close()



def compute_runtime_plot(all_runtimes, all_activations):
    global rank
    leader_rank = split_rank * process_per_layer

    runtimes_dict = {}
    activations_dict = {}
    if split_rank != last_rank and rank == leader_rank:
        send(jnp.array(all_runtimes), dest=last_rank * process_per_layer, tag=5,comm=comm)
        send(jnp.array(all_activations), dest=last_rank * process_per_layer, tag=5,comm=comm)
    elif split_rank == last_rank and rank == leader_rank:
        for i in range(last_rank):
            runtimes = recv(jnp.array(all_runtimes), source=i * process_per_layer, tag=5, comm=comm)
            activations = recv(jnp.array(all_activations), source=i * process_per_layer, tag=5, comm=comm)

            runtimes_dict[f"rank_{i}"] = runtimes.tolist()
            activations_dict[f"rank_{i}"] = activations.tolist()
            print(i, runtimes.shape)
    runtimes_dict[f"rank_{last_rank}"] = all_runtimes
    activations_dict[f"rank_{last_rank}"] = all_activations
    
    if rank == last_rank:
        # print(runtimes_dict.keys(), activations_dict.keys(), (all_activations), (runtimes_dict.values()))
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

        # First boxplot: runtimes
        axes[0].boxplot(runtimes_dict.values(), tick_labels=runtimes_dict.keys(), showfliers=False)
        axes[0].set_xlabel("Process rank")
        axes[0].set_ylabel("Runtime (seconds)")
        axes[0].set_title("Per-rank average runtimes")
        axes[0].grid(True)

        # Second boxplot: activations
        # activations = [np.ravel(v) for v in activations_dict.values()]
        axes[1].boxplot(activations_dict.values(), tick_labels=activations_dict.keys(), showfliers=False)
        axes[1].set_xlabel("Process rank")
        axes[1].set_ylabel("Activation number")
        axes[1].set_title("Per-rank average activations")
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(f'Plots/runtimes_and_activations_{size}.png')
        plt.close()
        
        create_runtime_video(runtimes_dict, activations_dict, f"Plots/runtimes_per_rank_video{size}.mp4")
    return
    
# region Inference
def batch_predict_time(params, key, weights, empty_neuron_states, dataset:str="train", save=True, debug=True):    
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
    
    all_runtimes = []
    all_activations = []
    epoch_iterations = []
    for i in range(total_batches):
        neuron_states = empty_neuron_states
        
        if split_rank == 0:                 
            batch_x, batch_y = split_batch(params, batch_iterator)
            # print(f"batch {i} has shape {batch_x.shape}, {batch_y.shape}")
        
        mpi4jax.barrier(comm=comm)
        start_predict_time = time.time()   
        if split_rank == 0:                 
            # outputs, iterations, all_neuron_states = (predict_batched)(weights, neuron_states, max_nonzero, batch_x)
            outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, jnp.array(batch_x))
            end_predict_time = time.time()

            # Send label to the last layer
            send(batch_y, dest=last_rank * process_per_layer + rank, tag=10,comm=comm)
        else:
            outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, jnp.zeros((batch_part, layer_sizes[0]))) 
            end_predict_time = time.time()
            # outputs, iterations, all_neuron_states = (predict_batched)(weights, neuron_states, max_nonzero, jnp.zeros((batch_size, layer_sizes[0])))
            # jax.debug.print("Rank {} All neuron states shape: {}, output shape : {}", rank, all_neuron_states.input_residuals.shape, outputs.shape)

        all_runtimes.append(end_predict_time - start_predict_time)
        all_activations.append((iterations.item()))
        # print(f"rank {rank} finished computing in: {end_predict_time - start_predict_time} seconds (start: {start_predict_time}, end: {end_predict_time})")
        mpi4jax.barrier(comm=comm)
        
        if split_rank != 0:
            if split_rank == last_rank:
                y = recv(jnp.zeros((batch_part,)), source=rank - (last_rank * process_per_layer), tag=10, comm=comm)   
                
                valid_y, batch_correct = accuracy(i, outputs, y, iterations, print=False)                 
                
                epoch_correct += batch_correct
                epoch_total += valid_y.shape[0]
        epoch_iterations.append(iterations)
        # jax.debug.print("Rank {}, iterations: {}", rank, iterations)
        # if i > 5:
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
    
    compute_runtime_plot(all_runtimes, all_activations)
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_iter_th(weights, mean, empty_neuron_states.thresholds)
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


def batch_predict(params, key, weights, empty_neuron_states, dataset:str="train", save=True, debug=True):    
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
    for i in range(total_batches):
        neuron_states = empty_neuron_states
        
        if split_rank == 0:                 
            batch_x, batch_y = split_batch(params, batch_iterator)
            # print(f"batch {i} has shape {batch_x.shape}, {batch_y.shape}")                 
            # outputs, iterations, all_neuron_states = (predict_batched)(weights, neuron_states, max_nonzero, batch_x)
            outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, jnp.array(batch_x))

            # Send label to the last layer
            send(batch_y, dest=last_rank * process_per_layer + rank, tag=10,comm=comm)
        else:
            outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, jnp.zeros((batch_part, layer_sizes[0]))) 
            # outputs, iterations, all_neuron_states = (predict_batched)(weights, neuron_states, max_nonzero, jnp.zeros((batch_size, layer_sizes[0])))
            # jax.debug.print("Rank {} All neuron states shape: {}, output shape : {}", rank, all_neuron_states.input_residuals.shape, outputs.shape)

        # print(f"rank {rank} finished computing in: {end_predict_time - start_predict_time} seconds (start: {start_predict_time}, end: {end_predict_time})")
        
        if split_rank != 0:
            if split_rank == last_rank:
                y = recv(jnp.zeros((batch_part,)), source=rank - (last_rank * process_per_layer), tag=10, comm=comm)   
                
                valid_y, batch_correct = accuracy(i, outputs, y, iterations, print=False)                 
                
                epoch_correct += batch_correct
                epoch_total += valid_y.shape[0]
        epoch_iterations.append(iterations)
        # jax.debug.print("Rank {}, iterations: {}", rank, iterations)
        # if i > 5:
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
    
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_iter_th(weights, mean, empty_neuron_states.thresholds)
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
def rerun_init(data_file_path, new_epoch_nb, dataset, th_lr=0, sparsity_impact=0, async_layer=-1):
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
    # sparsity_impact = tuple(stored_data["sparsity impact"])
    sparsity_impact = 0
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
        restrict=tuple(restrict),
        firing_nb=firing_nb,
        sync_rate=sync_rate,
        max_nonzero=max_nonzero,
        shuffle_input=shuffle_input,
        threshold_lr=th_lr,
        sparsity_impact=sparsity_impact,
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

if __name__ == "__main__":
    random_seed = 42
    key = jax.random.key(random_seed)
    
    dataset = 'mnist'
    # dataset = 'shd'
    
    # Network structure and parameters
    # layer_sizes = (28*28, 512, 256, 128, 64, 32, 16, 10)
    # layer_sizes = (28*28, 256, 128, 64, 32, 16, 10)
    # layer_sizes = (28*28, 128, 64, 32, 16, 10)
    # layer_sizes = (28*28, 128, 64, 32, 10)
    # layer_sizes = (28*28, 128, 64, 10)
    # layer_sizes = (28*28, 128, 10)
    all_layers = [] 
    
    # MNIST layers
    # all_layers.append((28*28, 32, 32, 32, 32, 32, 32, 32, 10))
    # all_layers.append((28*28, 64, 64, 64, 64, 64, 64, 64, 10))
    # all_layers.append((28*28, 128, 128, 128, 128, 128, 128, 128, 10))

    # all_layers.append((28*28, 32, 32, 32, 32, 32, 32, 10))
    # all_layers.append((28*28, 64, 64, 64, 64, 64, 64, 10))
    # all_layers.append((28*28, 128, 128, 128, 128, 128, 128, 10))

    # all_layers.append((28*28, 32, 32, 32, 32, 32, 10))
    # all_layers.append((28*28, 64, 64, 64, 64, 64, 10))    
    # all_layers.append((28*28, 128, 128, 128, 128, 128, 10))

    # all_layers.append((28*28, 32, 32, 32, 32, 10))
    # all_layers.append((28*28, 64, 64, 64, 64, 10))    
    # all_layers.append((28*28, 128, 128, 128, 128, 10))

    # all_layers.append((28*28, 32, 32, 32, 10))    
    # all_layers.append((28*28, 64, 64, 64, 10))
    # all_layers.append((28*28, 128, 128, 128, 10))
    
    # all_layers.append((28*28, 32, 32, 10))
    # all_layers.append((28*28, 64, 64, 10))    
    # all_layers.append((28*28, 128, 128, 10))

    # all_layers.append((28*28, 32, 10))
    # all_layers.append((28*28, 64, 10))    
    all_layers.append((28*28, 128, 10))    
    
    # SHD layers 
    # all_layers.append((700, 128, 128, 128, 20))    
    # all_layers.append((700, 64, 64, 64, 20))    
    # all_layers.append((700, 32, 32, 32, 20))    
    
    # all_layers.append((700, 128, 128, 20))    
    # all_layers.append((700, 64, 64, 20))    
    # all_layers.append((700, 32, 32, 20))    
    
    # all_layers.append((700, 128, 20))    
    # all_layers.append((700, 64, 20))    
    # all_layers.append((700, 32, 20))    
    
    layer_sizes = all_layers[0]
    best = False
    # layer_sizes = [4, 5, 3]
     
    load_file = False
    batch_size = 36
    restrict = (0,) * len(layer_sizes)
    # restrict = (2, 2, 2, 2, 1, 1)

    if size % len(layer_sizes) != 0:
        print(f"Error: layer_sizes ({len(layer_sizes)}) must match number of MPI ranks ({size})")
        sys.exit(1)
    # if len(layer_sizes) != len(restrict):
    #     print(f"Error: restrict ({len(restrict)}) must have the same size as layer_sizes ({len(layer_sizes)})")
    #     sys.exit(1)
    
    get_split_rank() # Compute the split rank for training/inference with multiple processes per batch

    if batch_size % process_per_layer != 0:
        print(f"Error: one batch ({batch_size}) must be divisible by the number of processes per layer ({process_per_layer})")
        sys.exit(1)
    
    for layer_sizes in all_layers:#[1, 2, 4, 8, 16, 32, 64, 128]:
        # restrict = (r,) * len(layer_sizes)        
        init_thresholds = 0.0#float(jnp.sqrt(2))
        key, subkey = jax.random.split(key) 
        thresholds = jax.nn.sigmoid(jax.random.normal(subkey, (layer_sizes[split_rank]))*init_thresholds)*0.0
        
        # test_surrogate_grad()
        rerun = "network_results/mnist/training/constant_layer_activation_test_with_fnb2/42_ep2_batch36_784_128_10_acc0.936_adam_.json"
        # rerun = "network_results/mnist/training/constant_layer_activation_test_with_fnb2/42_ep2_batch36_784_128_128_10_acc0.920_adam_.json"
        # rerun = "network_results/mnist/training/constant_layer_activation_test_with_fnb2/42_ep2_batch36_784_128_128_128_10_acc0.955_adam_.json"
        # rerun = "network_results/mnist/training/constant_layer_activation_test_with_fnb2/42_ep2_batch36_784_128_128_128_128_10_acc0.948_adam_.json"
        # rerun = "network_results/mnist/training/constant_layer_activation_test_with_fnb2/42_ep2_batch36_784_128_128_128_128_128_10_acc0.940_adam_.json"
        # rerun = "network_results/mnist/training/constant_layer_activation_test_with_fnb2/42_ep2_batch36_784_128_128_128_128_128_128_10_acc0.933_adam_.json"
        # rerun = "network_results/mnist/training/constant_layer_activation_test_with_fnb2/42_ep2_batch36_784_128_128_128_128_128_128_128_10_acc0.918_adam_.json"
        # rerun = "network_results/shd/training/ReLu_threshold_trained/42_ep20_batch36_700_128_128_20_acc0.580_adam_.json"
        rerun = "network_results/mnist/training/basic/load_false/42_ep20_batch36_784_128_10_acc0.976_adam_.json"
        # rerun = "network_results/mnist/training/basic/load_false/42_ep20_batch36_784_128_128_10_acc0.974_adam_.json"
        
        
        # rerun = "network_results/shd/training/basic/42_ep40_batch36_700_32_20_acc0.577_adam_.json"
        
        # rerun = "network_results/shd/training/basic/42_ep40_batch36_700_64_20_acc0.563_adam_.json"
        # rerun = "network_results/shd/training/sparsity_loss/42_ep10_batch36_700_128_128_128_20_acc0.048_adam_.json"
        
        # rerun = "network_results/shd/training/basic/42_ep10_batch36_700_32_32_20_acc0.520_adam_.json"
        
        # rerun = "network_results/shd/training/basic/42_ep10_batch36_700_64_64_20_acc0.572_adam_.json"
        
        
        # rerun = "network_results/shd/training/5hidden_parameter_tuning/42_ep10_batch36_700_128_128_128_20_acc0.045_adam_(1).json"
        # rerun = "network_results/shd/training/5hidden_parameter_tuning/42_ep10_batch36_700_128_128_128_20_acc0.522_adam_.json"
        
        # rerun = "network_results/shd/training/basic/42_ep20_batch36_700_128_128_20_acc0.580_adam_.json"
        # rerun = "network_results/shd/training/basic/42_ep10_batch36_700_128_128_20_acc0.612_adam_.json"
        # rerun = "network_results/shd/training/restrict/42_ep10_batch36_700_128_128_20_acc0.567_adam_.json"
        
        # rerun = 'network_results/shd/training/basic/42_ep10_batch36_700_128_128_20_acc0.612_adam_.json'
        # rerun = "network_results/shd/training/restrict/42_ep10_batch36_700_128_20_acc0.572_adam_.json"
        # rerun = "network_results/shd/training/sparsity_loss/42_ep10_batch36_700_128_20_acc0.613_adam_.json"
        rerun = None
        # range 4: 96.8 -> 75.9 -> 96.2 -> 49.6
        async_layer = -1
        # async_layer = 1
        cont = True
        # for i in range(5):#[0.0001, 0.001, 0.01]: #TODO rerun sigmoid 4 because multi layer training missed the dependency between 2 hidden layers' activations
            # for th_lr in [0.0001, 0.001, 0.01]:
        while cont:
                # Initialize parameters (input data for rank 0 and weights for other ranks)
                key, subkey = jax.random.split(key) 
                total_train_batches, total_val_batches, total_test_batches = 0, 0, 0
                if split_rank != 0:
                    weights, neuron_states = init_params(subkey, load_file=load_file, best=best)
                    batch_iterator = None
                    max_nonzero = layer_sizes[split_rank]
                if split_rank == 0:
                    max_nonzero = 0
                    if rank == 0:
                        # Load the data 
                        match dataset:
                            case "mnist":
                                loader = torch_mnist_loader_manual
                            case "shd":
                                loader = torch_SHD_loader
                            case _:
                                raise ValueError(f"Unknown dataset: {dataset}")
            
                        (training_generator, total_train_batches), (validation_generator, total_val_batches), (test_generator, total_test_batches), max_nonzero = loader(batch_size, shuffle=False)
                        print("max non zero: ", max_nonzero)
                    weights = jnp.zeros((layer_sizes[-1], layer_sizes[0]))
                
                # Broadcast total_batches to all other ranks
                total_train_batches, total_val_batches, total_test_batches = bcast(jnp.array([total_train_batches, total_val_batches, total_test_batches]), root=0 , comm=comm)

                if split_rank == 0:
                    max_nonzero = share_split_rank_data(max_nonzero) 
                    max_nonzero = max_nonzero.tolist() # share_split_rank_data returns an array
                    
                params = Params(
                    dataset=dataset,
                    random_seed=random_seed,
                    layer_sizes=layer_sizes, 
                    init_thresholds=init_thresholds, 
                    num_epochs=2, 
                    learning_rate=0.0001, 
                    batch_size=batch_size,
                    load_file=load_file,
                    shuffle_activations=False,
                    restrict=restrict,
                    firing_nb=4,
                    sync_rate=1,
                    max_nonzero=max_nonzero,
                    shuffle_input=False,
                    threshold_lr=0.0, 
                    sparsity_impact=tuple([0.0000, 0.0000, 0.0000, 0.0000, 0.0000]), # Beta sparse
                    rerun="",
                    async_layer=async_layer
                )
                
                folder = "" #"network_results/training/"
                # rerun = "42_ep20_batch36_784_128_64_10_acc0.967_adam_.json"
                # rerun = "42_ep20_batch36_784_128_64_10_acc0.973_adam_.json"
                # rerun = "42_ep1_batch36_784_128_64_10_acc0.799_adam_.json"
                # rerun = None
                if rerun is not None:
                    new_epoch_number = 10 # Number of training epoch to run again
                    th_lr, beta = 1, 0.0
                    
                    # if async_layer >= last_rank:
                    #     async_layer = -1
                    # elif async_layer == -1:
                    #     cont = False
                    #     continue
                    # else:
                    #     async_layer += 1
                    
                    # if i % 2:
                    #     new_epoch_number = 1
                    #     beta = 0.01
                    
                    params, weights, thresholds = rerun_init(folder+rerun, new_epoch_number, dataset, th_lr, beta, async_layer=async_layer)
                    if len(layer_sizes) != len(params.layer_sizes):
                        print(f"Error: rerun file {rerun} has different layer sizes than the current network structure {layer_sizes}.")
                        sys.exit(1)
                
                if rank == 0:
                    print(f"Number of training batches: {total_train_batches}, validation batches: {total_val_batches}, test batches: {total_test_batches}")
                    print(params)
                
                empty_neuron_states = NeuronStates( values=jnp.zeros(layer_sizes[split_rank]),
                                                    thresholds=thresholds,
                                                    input_residuals=np.zeros((layer_sizes[split_rank-1],)),
                                                    input_order=jnp.full((layer_sizes[split_rank-1],), -1, dtype=int), 
                                                    input_activity=jnp.full((layer_sizes[split_rank-1],), 0, dtype=int),
                                                    layer_activity=jnp.zeros((layer_sizes[split_rank],), dtype=int),
                                                    output_activity=jnp.zeros((layer_sizes[split_rank-1], layer_sizes[split_rank])),
                                                    last_sent_iteration=0,
                                                    seen_mask=jnp.zeros((layer_sizes[split_rank-1], layer_sizes[split_rank]), dtype=bool),
                                                    activated_mask=jnp.zeros((layer_sizes[split_rank-1], layer_sizes[split_rank]), dtype=bool))
                
                t = 2
                all_time = 0
                # for i in range(t):
                #     _, _, ex_time = batch_predict(params, key, weights, empty_neuron_states, "test", save=False, debug=True)
                #     all_time += ex_time
                # print("average execution time : {}", all_time/t)

                # batch_predict(params, key, weights, empty_neuron_states, "test", save=False, debug=True)
                result_path = train(params, key, weights, empty_neuron_states, "adam")
                # rerun = result_path
                # print(rerun)
                break
                