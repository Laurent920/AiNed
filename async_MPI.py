import os

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

from z_helpers.mnist_helper import torch_loader_manual
from z_helpers.iris_species_helper import torch_loader
from z_helpers.network_helper import one_hot_encode

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
@dataclasses.dataclass
@tree_math.struct
class Neuron_states:
    '''
    values: jnp.ndarray                     # Current state of the neurons in the layer, shape: (layer_sizes[rank],) __ (128,)
    thresholds: jnp.float32                 # An array of thresholds, one per neuron, shape: (layer_sizes[rank],) __ (128,)
    input_residuals: jnp.ndarray            # Sum of all inputs for each neuron, shape: (layer_sizes[rank-1], 1) __ (784, 1)
    weight_residuals: dict[str, jnp.ndarray]
        - "input order"                     # Set input neuron to the iteration at which the input is received to record the order of input received, shape: (layer_sizes[rank-1], 1) __ (784, 1)
        - "input activity"                  # Count the number of times a input neuron fired, shape: (layer_sizes[rank-1], 1) __ (784, 1)
        - "layer activity"                  # Count the number of times a neuron activated in this layer, only used for restrict parameter, shape: (layer_sizes[rank],) __ (128,)
        - "output activity"                 # For each input neuron stores the hidden neurons that fire, shape: (layer_sizes[rank-1], layer_sizes[rank]) __ (784, 128)
    '''
    values: jnp.ndarray                                  
    thresholds: jnp.ndarray           
    input_residuals: jnp.ndarray      
    weight_residuals: dict[str, jnp.ndarray]
    last_sent_iteration: int

@dataclasses.dataclass(frozen=True)
class Params:
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

# region INFERENCE
@custom_jvp # If thresholds == 0 then this behaves as a ReLu activation function 
def activation_func(neuron_states, activations):
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
def layer_computation(neuron_idx, layer_input, weights, neuron_states, params, iteration=0):    
    # activations = jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values
    activations = jax.lax.cond(neuron_idx < 0,
                            lambda _: neuron_states.values,
                            lambda _: jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values,
                            None
                            )
    #TODO being able to compute multiple incoming index neurons
    #TODO store the weight residuals of last layer and neuron state of input layer in sparse matrix representation to reduce space utilization because unused
    jax.lax.cond(jnp.isnan(jnp.array(layer_input)).any(), 
                           lambda _: jax.debug.print("Rank {}: layer_input is NaN: {}, idx: {}, iteration:{}", rank, layer_input, neuron_idx, iteration), 
                           lambda _: None, None)
    
    
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
        return jnp.zeros_like(activations), Neuron_states(
                                            values=activations, 
                                            thresholds=neuron_states.thresholds, 
                                            input_residuals=new_input_residuals, 
                                            weight_residuals=neuron_states.weight_residuals,
                                            last_sent_iteration=neuron_states.last_sent_iteration
                                            )
    
    def hidden_layer_case(_):
        fire = (iteration-neuron_states.last_sent_iteration) >= params.sync_rate 
        fire = jnp.logical_or(fire, neuron_idx < 0) # fire if sync rate reached or last input received
        # jax.debug.print("Rank {}, neuron idx: {}, fire: {}, iteration: {}, last sent iteration: {}", rank, neuron_idx, fire, iteration, neuron_states.last_sent_iteration)

        # APPLY THE SYNC RATE  
        activated_output = jax.lax.cond(fire, 
                                        lambda args: activation_func(args[0], args[1]), 
                                        lambda _: jnp.zeros(activations.shape),
                                        (neuron_states, activations))
        
        # APPLY THE FIRING NUMBER        
        activated_output = keep_top_k(activated_output, params.firing_nb) # Get the top k activations
        # jax.debug.print("{}, iteration: {}, neuron idx: {}", activated_output, iteration, neuron_idx)

        layer_activity = neuron_states.weight_residuals["layer activity"]
        
        # APPLY THE RESTRICTION
        restrict_cond = jnp.logical_and(params.restrict > 0, layer_activity >= params.restrict) 
        penalty_mask = jnp.where(restrict_cond, 1.0, 0.0) # Create a mask to keep the neurons that have activated above the restricted value 
        
        penalty = penalty_mask * activated_output
        
        # active_output = activated_output * (output_mask) # Apply the mask to the activated output to get the actual activations
        
        active_output = activated_output
        active_indexes = jnp.where(active_output > 0, 1, 0)
        
        new_layer_activities = layer_activity + active_indexes # Update the layer activity by adding the active neurons

        last_neuron_idx = jnp.argmax(neuron_states.weight_residuals["input order"]) # Last neuron index in the input order
        # jax.lax.cond(neuron_idx == -1,
        #              lambda _: jax.debug.print("{}, iteration: {}, neuron idx: {}", neuron_idx, iteration, last_neuron_idx),
        #              lambda _: None,
        #              None)
        new_neuron_idx = jax.lax.cond(neuron_idx == -1,
                     lambda _: last_neuron_idx, 
                     lambda _: neuron_idx,
                     None)
        
        new_input_activities = neuron_states.weight_residuals["input order"].at[new_neuron_idx].set(iteration) # Update the input activity by setting the input neuron to the iteration number        
        
        # jax.debug.print("{} {}", active_indexes.shape, new_input_activities.shape)
        # new_values = update_new_values(neuron_states.weight_residuals["output activity"], active_indexes, new_input_activities) # Update input activity before updating the values
        new_output_activity = neuron_states.weight_residuals["output activity"].at[new_neuron_idx].add(active_indexes)
        
        jax.lax.cond(neuron_idx == -2,
                     lambda _: jax.debug.print("{}, iteration: {}, neuron idx: {}", layer_input, iteration, neuron_idx),
                     lambda _: None,
                     None)

        new_weight_residuals = {"input order": new_input_activities, 
                                "input activity": new_input_activity,
                                "layer activity": new_layer_activities,
                                "output activity": new_output_activity}

        new_last_sent_iteration = jax.lax.cond(fire, lambda _: iteration, lambda _: neuron_states.last_sent_iteration, None)

        new_neuron_states = Neuron_states(values=activations - active_output - penalty, 
                                          thresholds=neuron_states.thresholds, 
                                          input_residuals=new_input_residuals, 
                                          weight_residuals=new_weight_residuals,
                                          last_sent_iteration=new_last_sent_iteration)
        return active_output, new_neuron_states
    
    cond = split_rank == last_rank #jnp.logical_or(split_rank == last_rank, neuron_idx < 0)
    return jax.lax.cond(cond, last_layer_case, hidden_layer_case, None)

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

@partial(jax.jit, static_argnames=['max_nonzero'])
def preprocess_to_sparse_data_padded(x, max_nonzero):
    # Pre-allocate max possible
    processed_data = jnp.full((max_nonzero, 2), -2.0)
        
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
            # Case 1: neuron_val == 0 and activates[j] == 1 → set to 1
            condition = (vals == 0) & (activates == 1)
            new_vals = jnp.where(condition, 1, vals)
            output_activity = output_activity.at[j].set(new_vals)

            # Case 2: neuron_val == 1 and activates[j] == 0 → set activates[j] = 1
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
    activates = jnp.zeros((output_activity.shape[1],), dtype=jnp.int32) #(128, 1)

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
    numeric_mask = (~mask).astype(current_res.dtype)  # invert to keep columns where mask is False
    # if rank == 1:
    #     jax.debug.print("numeric_mask shape: {}, mask: {}", numeric_mask.shape, jnp.sum(numeric_mask))
    
    # Broadcast to shape (128, 64)
    full_mask_a = jnp.expand_dims(numeric_mask, axis=0)  # (1, 64)
    full_mask = jnp.broadcast_to(full_mask_a, current_res.shape)  # (128, 64)
    # jax.debug.print("rank {}: {}, {}", rank, full_mask_a, full_mask)

    out = current_res * full_mask
    return out

@partial(jax.jit, static_argnames=['params'])
def predict_bwd(params, key, batch_data, weights, empty_neuron_states, token):
    '''
    B: batch_size
    '''
    token, all_outputs, iterations, all_neuron_states = (predict)(params, key, weights, empty_neuron_states, token, batch_data)
    next_grad, token = recv(jnp.zeros((batch_part, layer_sizes[split_rank])), source=rank + process_per_layer, tag=2, comm=comm) # Shape: (B, 128)
    # jax.debug.print("Rank {} received next_grad shape: {}", rank, next_grad)
    
    # "input order": Shape (B, 784, 1), "output activity": Shape (B, 784, 128)
    weight_res = jax.vmap(compute_w_residuals, in_axes=(0, 0))(all_neuron_states.weight_residuals["input order"], all_neuron_states.weight_residuals["output activity"]) # Shape: (B, 784, 128)
    # weight_res = weight_res["output activity"] # incorrect residual but faster for testing
    
    next_weight_res = jnp.ones((batch_part, params.layer_sizes[split_rank], params.layer_sizes[split_rank+1])) # Shape: (B, 128, 10)
    # jax.debug.print("Rank {} received next_grad shape: {}, next_weight_res shape: {}", rank, next_grad.shape, next_weight_res.shape)
    (next_weight_res, token) = jax.lax.cond(split_rank < last_rank - 1, 
                                   lambda _: recv(next_weight_res, source=rank + process_per_layer, tag=3, comm=comm),
                                   lambda _: (next_weight_res, token), None) 
    # jax.debug.print("Rank {} received next_grad shape: {}", rank, next_weight_res)

    weight_res = jax.lax.cond(split_rank < last_rank - 1,
                                lambda args: jax.vmap(recompute_w_residuals, in_axes=(0, 0))(args[0], args[1]), # Shape: (B, 784, 128)
                                lambda _: weight_res,
                                (weight_res, next_weight_res))    
    
    # Perform element-wise multiplication
    z_grad = jax.vmap(z_gradient, in_axes=(0, 0))(weight_res, next_grad) # Shape: (B, 784, 128)
    
    # Sanity check
    # p = jnp.logical_and(all_neuron_states.input_residuals == 0, all_neuron_states.weight_residuals["input activity"] == 0)
    # n = jnp.logical_and(all_neuron_states.input_residuals != 0, all_neuron_states.weight_residuals["input activity"] != 0)
    # t = jnp.logical_or(p, n) 
    # jax.debug.print("Rank {}, check values {}/{}", rank, jnp.sum(t)/12, 784) 
    
    input_activity = all_neuron_states.weight_residuals["input activity"] # Shape (B, 784)
    # jax.debug.print("Rank {}, input activity shape: {}, input activity: {}", rank, input_activity.shape, jnp.sum(input_activity > 1))
    x = all_neuron_states.input_residuals #/ jnp.where(input_activity == 0, 1.0, input_activity)# Shape (B, 784)
    x_reshaped = x[..., jnp.newaxis]   # Shape becomes (B, 784, 1)
    
    weight_grad = x_reshaped * z_grad # (B, 784, 128)
    # weight_grad = jnp.squeeze(weight_grad, axis=2)
    # jax.debug.print("weight_grad: {}, x: {}, z_grad: {}, next_grad_expanded: {}, weight_res: {}", jnp.isnan(weight_grad).any(), jnp.isnan(x).any(), jnp.isnan(z_grad).any(), jnp.isnan(next_grad_expanded).any(), jnp.isnan(weight_res).any())
    mean_weight_grad = jnp.expand_dims(jnp.mean(weight_grad, axis=0), axis=0) # (1, 784, 128)

    # jax.debug.print("x {}, x_reshaped{}", x.shape, x)
    # jax.debug.print("next_grad_expanded {}, {}", next_grad_expanded.shape, next_grad_expanded)
    # jax.debug.print("weight residuals {}, {}", weight_res.shape, weight_res)
    # jax.debug.print("z_grad {}, {}", z_grad.shape, z_grad)
    # jax.debug.print("weight_grad {}, mean_weight_grad{}", weight_grad.shape, mean_weight_grad.shape)
    
    if split_rank > 1:
        tmp_send_grad = weights @ next_grad.T # Shape: (784, B)
        send_grad = jnp.reshape(tmp_send_grad, (batch_part, tmp_send_grad.shape[0])) # Shape: (B, 784)
        # send_grad = jnp.ones((batch_part, params.layer_sizes[split_rank-1])) 

        token = send(send_grad, dest=rank-process_per_layer, tag=2,comm=comm, token=token)
        token = send(weight_res, dest=rank-process_per_layer, tag=3,comm=comm, token=token)

    return token, all_outputs, iterations, all_neuron_states, weight_grad 

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
    weight_grad = loss_grad_expanded * all_residuals  # Shape: (128, 10)
    return weight_grad

@partial(jax.jit, static_argnames=['params'])
def loss_fn(params, key, batch_data, weights, empty_neuron_states, token, target):
    token, all_outputs, iterations, all_neuron_states = (predict)(params, key, weights, empty_neuron_states, token, batch_data)
    # jax.debug.print("output shape: {}, target shape: {}", all_outputs.shape, target.shape)
    
    # loss = jnp.mean((all_outputs - target) ** 2)
    # N = all_outputs.shape[0]  
    # loss_grad = (2 / N) * (all_outputs - target)
    all_residuals = all_neuron_states.input_residuals # Shape: (B, 128)
    # jax.debug.print("weight shape: {} {}", all_neuron_states.thresholds[0], all_neuron_states.thresholds)

    thr_impact = params.threshold_impact
    threshold = all_neuron_states.thresholds[0] # Shape all_neuron_states.thresholds is (B, layer_size)
    threshold_loss = thr_impact * jnp.mean(jnp.sum(all_residuals, axis=1), axis=0) #+ (1/threshold)) # average over the batches for the sum of activations outputed from the last hidden layer
    threshold_grad = -thr_impact #/ (threshold ** 2)
    # jax.debug.print("threshold loss shape: {} {}", threshold_loss, threshold_grad)
    
    # jax.debug.print("regularized average iterations: {},  {}/{}", reg_avg_iterations, jnp.mean(iterations), jnp.max(iterations))
    loss, loss_grad = jax.value_and_grad(mean_loss)(all_outputs, target)
    loss_grad /= process_per_layer # Shape (B, 10)
    # total_loss = loss + threshold_loss[0]
    # jax.debug.print("rank {}, loss: {}", rank, loss)
    
    out_grad = jax.vmap(output_gradient, in_axes=(None, 0))(weights, loss_grad) # Shape (B, 128)
    # jax.debug.print("rank {}, loss: {}, loss_grad shape: {}, out_grad shape: {}", rank, loss, loss_grad.shape, out_grad.shape)
    
    weight_grad =  jax.vmap(output_weight_grad, in_axes=(0, 0))(loss_grad, all_residuals) #Shape (B, 128, 10)
    # jax.debug.print("Rank {}, all_residuals shape: {}, threshold shape: {}, weight grad shape: {}", rank, all_residuals.shape, all_neuron_states.thresholds.shape, weight_grad.shape)

    weight_grad = jnp.reshape(weight_grad, (weight_grad.shape[0], weight_grad.shape[2], weight_grad.shape[1])) # Shape: (B, 10, 128)
    mean_weight_grad = jnp.expand_dims(jnp.mean(weight_grad, axis=0), axis=0) # Shape: (1, 10, 128)
    # jax.debug.print("rank {}, mean weight grad: {}", rank, mean_weight_grad)
    
    # jax.debug.print("loss: {}, loss gradient: {}", loss, loss_grad.shape)
    # jax.debug.print("out grad {}, {}", out_grad.shape, out_grad)
    # jax.debug.print("all residuals {}, {}", all_residuals.shape, all_residuals.dtype)
    # jax.debug.print("weight_grad {}, mean_weight_grad{}", weight_grad.shape, mean_weight_grad.shape)

    return (loss, all_outputs, iterations), (out_grad, weight_grad, threshold_grad, loss_grad, weight_grad)

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
    
    
    if jnp.isnan(batch_x).any() or jnp.isnan(batch_y).any():
        jax.debug.print("Rank {}: NaN detected in batch_x: {}, batch_y: {}", rank, batch_x, batch_y)
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
    
    if jnp.isnan(data).any():
        jax.debug.print("Rank {} NaN detected in data: {}", rank, data.shape)
        
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
def train(token, params: Params, key, weights, empty_neuron_states):     
    """
    tag 0:  forward computation, data format: (previous_layer_neuron_index, neuron_value)
            end of input is encoded with the index -1
    tag 2: backward computation, last layer gradient shape: (layer_sizes[-1], 1)
    tag 3: weight residuals, shape: (layer_sizes[rank], layer_sizes[rank+1])
    tag 4: communication between processes that have the data
    tag 5: weights
    tag 10: data labels(y)
    tag 20: other data
    """   
    global training_generator
    global validation_generator
    global test_generator
        
    if split_rank == last_rank:
        all_epoch_accuracies = []
        all_validation_accuracies = []
        all_loss = []
    all_mean_iterations = []
        
    solver = optax.adam(learning_rate=params.learning_rate)
    # print("AMSGrad")
    # solver = optax.amsgrad(learning_rate=params.learning_rate)
    opt_state = solver.init(weights)
    
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
            threshold_grad = 0.0
            if split_rank == 0:
                # print(i)
                token, batch_x, batch_y = split_batch(token, batch_iterator)
                if jnp.isnan(batch_x).any():
                    jax.debug.print("Rank {}: i {}, NaN detected in batch_x: {}, batch_y: {}", rank, i, jnp.sum(jnp.isnan(batch_x)), jnp.sum(jnp.isnan(batch_y)))
                token = send(batch_y, dest=last_rank * process_per_layer + rank, tag=10,comm=comm, token=token) # Destination rank: last_rank * process_per_layer + rank

                token, outputs, iterations, all_neuron_states = (predict)(params, subkey, weights, neuron_states, token, batch_data=jnp.array(batch_x))
            else:
                if split_rank==last_rank:
                    # Receive y
                    y, token = recv(jnp.zeros((batch_part,)), source=rank - (last_rank * process_per_layer), tag=10, comm=comm, token=token)  # Source rank opposite operation: rank - (last_rank * process_per_layer)
                    y_encoded = jnp.array(one_hot_encode(y, num_classes=layer_sizes[-1]))
                    # print("encoded y: ", y, y_encoded.shape, y_encoded)              
                    (loss, outputs, iterations), gradients = (loss_fn)(params, subkey, jnp.zeros((batch_part, layer_sizes[0])), weights, neuron_states, token, y_encoded)

                    epoch_loss.append(loss)
                    
                    weight_grad = gradients[1]
                    threshold_grad = gradients[2]
                    
                    # Send gradient to previous layers                
                    token = send(gradients[0], dest=rank-process_per_layer, tag=2,comm=comm, token=token)
                    
                    valid_y, batch_correct = accuracy(i, outputs, y, iterations, False)                 
                        
                    epoch_correct += batch_correct
                    epoch_total += valid_y.shape[0]
                else:
                    token, outputs, iterations, all_neuron_states, weight_grad = (predict_bwd)(params, subkey, jnp.zeros((batch_part, layer_sizes[0])), weights, neuron_states, token)
                    # print(f"Rank {rank} finished predict_bwd for batch {i}, outputs shape: {outputs.shape}, iterations: {iterations.shape}, weight_grad shape: {weight_grad.shape}")
                    
                    if jnp.isnan(weight_grad).any():
                        numnans = jnp.sum(jnp.isnan(weight_grad))
                        print(f"Rank {rank} encountered NaN in weight_grad at epoch {epoch}, batch {i}, number nans: {numnans}. Skipping update.")

                # print(f"Rank {rank} before combine_batch")
                token, weight_grad = combine_batch(token, weight_grad, average=True) # Gather the weight gradients from all ranks in the split rank
                # print(f"Rank {rank} after combine_batch")
                
                if jnp.isnan(weight_grad).any():
                    numnans = jnp.sum(jnp.isnan(weight_grad))
                    print(f"Rank {rank} encountered NaN in weight_grad at epoch {epoch}, batch {i}, number nans: {numnans}. Skipping update.")
                    continue
                
                # Update weights
                # Optax optimizer
                # updates, opt_state = solver.update(weight_grad, opt_state, weights)
                # weights = optax.apply_updates(weights, updates)
                
                # Basic GD
                weights -= params.learning_rate * weight_grad 

            # Update threshold
            # threshold_grad, token = bcast(threshold_grad, root=size-1, comm=comm, token=token)
            # empty_neuron_states.thresholds -= threshold_grad * params.threshold_lr 
            
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
    thresholds = empty_neuron_states.thresholds
    # Inference on the test set
    test_accuracy, test_mean, _ = batch_predict(params, key, token, weights, empty_neuron_states, dataset="test", save=False, debug=False)
    # test_accuracy = 0
    # Gather the weights and iteration values at the last layer
    weights_dict, all_iteration_mean = gather_weights_and_iterations(weights, jnp.array(all_mean_iterations), token)
    
    jax.block_until_ready(token)

    # Synchronize all MPI processes again
    token = mpi4jax.barrier(comm=comm, token=token)
    end_time = time.time()
        
    if rank == last_rank * process_per_layer:
        # Execution time
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.6f} seconds")
        store_training_data(params, 
                            "train",
                            all_epoch_accuracies, 
                            all_validation_accuracies, 
                            test_accuracy,
                            execution_time,
                            all_iteration_mean,
                            weights_dict,
                            all_loss, 
                            thresholds)
        
# region SAVE DATA
def store_training_data(params, mode, all_epoch_accuracies, all_validation_accuracies, test_accuracy, execution_time, all_iteration_mean, weights_dict, all_loss, thresholds):    
    # Choose the saving folder
    if mode == "train":
        result_dir = os.path.join("network_results", "training")
        filename_header = f"{params.random_seed}" + f"_ep{params.num_epochs}" + f"_batch{params.batch_size}_"
    elif mode == "inference":
        result_dir = os.path.join("network_results", "mnist")
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
    result_path = os.path.join(result_dir, filename)
    
    if os.path.exists(result_path + ".json"):
        index = 1
        while True:
            new_result_path = result_path+f"({index})"
            if os.path.exists(new_result_path + ".json"):
                index+=1
            else:
                result_path = new_result_path
                break                

    # Store the results
    thresholds =  np.array(thresholds).tolist()
    result_data = {
        "time": float(execution_time),
        "loadfile": params.load_file,
        "shuffle data": params.shuffle,
        "shuffle input": params.shuffle_input,
        "rerun": params.rerun,
        "processes": size,
        "firing number": params.firing_nb,
        "synchronization rate": params.sync_rate,
        "training accuracy": np.array(all_epoch_accuracies).tolist(),
        "validation accuracy": np.array(all_validation_accuracies).tolist(),
        "test accuracy": test_accuracy,
        "layer_sizes": params.layer_sizes,
        "batch_size": params.batch_size,
        "learning rate": params.learning_rate,
        "thresholds": thresholds,
        "iterations mean": np.array(all_iteration_mean).tolist(),
        "restrict": params.restrict,
        "threshold impact": params.threshold_impact,
        "threshold lr": params.threshold_lr,
        "loss": [float(loss) for loss in all_loss],
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
        neuron_states = Neuron_states(values=jnp.zeros(layer_sizes[split_rank]),
                                      thresholds=jnp.full((layer_sizes[split_rank]), init_thresholds),
                                      input_residuals=np.zeros((layer_sizes[split_rank-1],)),
                                      weight_residuals={"input order": jnp.full((layer_sizes[split_rank-1],), -1, dtype=int),
                                                        "input activity": jnp.full((layer_sizes[split_rank-1],), 0, dtype=int),  
                                                        "layer activity": jnp.zeros((layer_sizes[split_rank],), dtype=int), 
                                                        "output activity": jnp.zeros((layer_sizes[split_rank-1], layer_sizes[split_rank]))},
                                      last_sent_iteration=0
                                      )

        if load_file:
            print("Loading the weight file...")
            filename = f"tensor_data_{'_'.join(map(str, layer_sizes))}.npz"
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
        pad_x = jnp.zeros((pad_amount, batch_x.shape[1]))  

        batch_y = jnp.concatenate([batch_y, pad_y], axis=0)
        batch_x = jnp.concatenate([batch_x, pad_x], axis=0)
    
    return batch_x, batch_y

def preprocess_data(data_generator):
    # Preprocess the data
    preprocessed = data_generator
    
    return iter(preprocessed)

def gather_weights_and_iterations(weights, mean_iterations, token):
    # Gather all the weights and iteration values at the last layer to store them
    leader_rank = split_rank * process_per_layer

    weights_dict = {}
    all_iteration_mean = []
    if split_rank != last_rank and rank == leader_rank:
        token = send(weights, dest=last_rank * process_per_layer, tag=5,comm=comm, token=token)
        token = send(mean_iterations, dest=last_rank * process_per_layer, tag=5,comm=comm, token=token)
    elif split_rank == last_rank and rank == leader_rank:
        for i in range(last_rank):
            w, token = recv(jnp.zeros((layer_sizes[i-1], layer_sizes[i])), source=i * process_per_layer, tag=5, comm=comm, token=token)   
            weights_dict[f"layer_{i}"] = w.tolist()
            
            iter_mean, token = recv(mean_iterations, source=i * process_per_layer, tag=5, comm=comm, token=token)
            all_iteration_mean.append(iter_mean)
        all_iteration_mean.append(mean_iterations)  # Append the mean iterations of the last layer
        weights_dict[f"layer_{last_rank}"] = weights.tolist()
        all_iteration_mean = all_iteration_mean[1:] # Don't keep the value of the input layer
        print("all iteration mean: rank", rank, all_iteration_mean)
        
    return weights_dict, all_iteration_mean


# region Main 
def batch_predict(params, key, token, weights, empty_neuron_states, dataset:str="train", save=True, debug=True):    
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
            # token, outputs, iterations, all_neuron_states = (predict_batched)(weights, neuron_states, token, max_nonzero, batch_x)
            token, outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, token, jnp.array(batch_x))
            
            # Send label to the last layer
            token = send(batch_y, dest=last_rank * process_per_layer + rank, tag=10,comm=comm, token=token)
        else:
            token, outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, token, jnp.zeros((batch_part, layer_sizes[0])))
            # token, outputs, iterations, all_neuron_states = (predict_batched)(weights, neuron_states, token, max_nonzero, jnp.zeros((batch_size, layer_sizes[0])))
            # jax.debug.print("Rank {} All neuron states shape: {}, output shape : {}", rank, all_neuron_states.input_residuals.shape, outputs.shape)

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
    
    weights_dict, all_iteration_mean = gather_weights_and_iterations(weights, mean, token)
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
                                empty_neuron_states.thresholds)
    return epoch_accuracy, mean, end_time - start_time

def rerun_init(data_file_path, new_epoch_nb):
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
    init_thresholds = stored_data["thresholds"]
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
        threshold_lr=threshold_lr,
        threshold_impact=threshold_impact,
        rerun=data_file_path
    )
    
    weights = jnp.array(weights_dict["layer_"+str(split_rank)])
    return params, weights

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
    # Network structure and parameters
    layer_sizes = (28*28, 128, 64, 10)
    # layer_sizes = (28*28, 128, 10)
    # layer_sizes = (28*28, 64, 10)
    best = False
    # layer_sizes = [4, 5, 3]
     
    load_file = False
    init_thresholds = 0
    batch_size = 36
    shuffle = False
    
    if size % len(layer_sizes) != 0:
        print(f"Error: layer_sizes ({len(layer_sizes)}) must match number of MPI ranks ({size})")
        sys.exit(1)
    
    get_split_rank() # Compute the split rank for training/inference with multiple processes per batch

    if batch_size % process_per_layer != 0:
        print(f"Error: one batch ({batch_size}) must be divisible by the number of processes per layer ({process_per_layer})")
        sys.exit(1)
    
    # test_surrogate_grad()
    for r in [200]:
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
                (training_generator, total_train_batches), (validation_generator, total_val_batches), (test_generator, total_test_batches), max_nonzero = torch_loader_manual(batch_size, shuffle=shuffle)
                # training_generator, train_set, test_set, total_batches = torch_loader(batch_size, shuffle=shuffle)
                print("max non zero: ",max_nonzero)
            weights = jnp.zeros((layer_sizes[-1], layer_sizes[0]))
        
        # Broadcast total_batches to all other ranks
        (total_train_batches, total_val_batches, total_test_batches), token = bcast(jnp.array([total_train_batches, total_val_batches, total_test_batches]), root=0 , comm=comm)

        if split_rank == 0:
            token, max_nonzero = share_split_rank_data(token, max_nonzero) 
            max_nonzero = max_nonzero.tolist() # share_split_rank_data returns an array
            
        params = Params(
            random_seed=random_seed,
            layer_sizes=layer_sizes, 
            init_thresholds=init_thresholds, 
            num_epochs=1, 
            learning_rate=0.001, 
            batch_size=batch_size,
            load_file=load_file,
            shuffle=shuffle,
            restrict=-1,
            firing_nb=128,
            sync_rate=784,
            max_nonzero=max_nonzero,
            shuffle_input=False,
            threshold_lr=0,#1e-3,
            threshold_impact=0,
            rerun=""
        )
        
        folder = "network_results/training/" 
        rerun = "42_ep40_batch36_784_128_10_acc0.968.json"
        rerun = "42_ep10_batch36_784_128_64_10_acc0.833.json"
        rerun = None
        # print(rerun, rerun is not None)
        if rerun is not None:
            new_epoch_number = 11 # Number of training epoch to run again
            params, weights = rerun_init(folder+rerun, new_epoch_number)
            if len(layer_sizes) != len(params.layer_sizes):
                print(f"Error: rerun file {rerun} has different layer sizes than the current network structure {layer_sizes}.")
                sys.exit(1)
        
        if rank == 0:
            print(f"Number of training batches: {total_train_batches}, validation batches: {total_val_batches}, test batches: {total_test_batches}")
            print(params)
            
        empty_neuron_states = Neuron_states(
                                values=jnp.zeros((layer_sizes[split_rank])), 
                                thresholds=jnp.full((layer_sizes[split_rank]), init_thresholds), 
                                input_residuals=np.zeros((layer_sizes[split_rank-1],)),
                                weight_residuals={"input order": jnp.full((layer_sizes[split_rank-1],), -1, dtype=int), 
                                                  "input activity": jnp.full((layer_sizes[split_rank-1],), 0, dtype=int), 
                                                  "layer activity": jnp.zeros((layer_sizes[split_rank],), dtype=int), 
                                                  "output activity": jnp.zeros((layer_sizes[split_rank-1], layer_sizes[split_rank]))},
                                last_sent_iteration=0
                                )
        t = 100
        all_time = 0
        # for i in range(t):
        #     _, _, ex_time = batch_predict(params, key, token, weights, empty_neuron_states, "val", save=False, debug=True)
        #     all_time += ex_time
        # print("average execution time : {}", all_time/t)

        # batch_predict(params, key, token, weights, empty_neuron_states, "train", save=True, debug=True)
        train(token, params, key, weights, empty_neuron_states)