from network_helper import one_hot_encode
from mpi4py import MPI
import jax
import jax.numpy as jnp
from jax import custom_jvp, jit
from jax.tree_util import Partial
from functools import partial

from jax import jacfwd, jacrev
import matplotlib.pyplot as plt
import time

import json
import os
import sys
import numpy as np
import mpi4jax
from mpi4jax import send, recv, bcast

import tree_math
import dataclasses
from typing import Generic, Any, Union, TypeVar, Tuple

from z_helpers.mnist_helper import torch_loader_manual
from z_helpers.iris_species_helper import torch_loader

# jax.config.update("jax_debug_nans", True)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@dataclasses.dataclass
@tree_math.struct
class Neuron_states:
    values: jnp.ndarray
    threshold: jnp.float32
    input_residuals: jnp.ndarray
    weight_residuals: dict[str, jnp.ndarray]
    last_sent_iteration: int

@dataclasses.dataclass(frozen=True)
class Params:
    random_seed: int
    layer_sizes: tuple[int, ...]
    thresholds: tuple[float, ...] # Starting thresholds
    num_epochs: int 
    learning_rate: float
    batch_size: int
    load_file: bool
    shuffle: bool           # Shuffle the dataset
    restrict: int           # The amount of times a single neuron can fire accross all inputs
    firing_nb: int          # The maximum number of neurons that can fire for one input at each layer
    firing_rate: int        # The number of inputs that needs to be accumulated before firing  
    max_nonzero: int
    shuffle_input:bool      # Shuffle the data in each layer to simulate async individual neurons

# region INFERENCE
@custom_jvp # If threshold == 0 then this behaves as a ReLu activation function 
def activation_func(neuron_states, activations):
    return jnp.where(activations > neuron_states.threshold, activations, 0.0)

@activation_func.defjvp
def activation_func_jvp(primals, tangents, k=1.0):
    # Surrogate gradient, redefine the function for the backward pass
    neuron_states, activations, = primals
    neuron_states_dot, activations_dot, = tangents
    ans = activation_func(neuron_states, activations)
    ans_dot = jnp.where(activations > neuron_states.threshold, activations, 0.0)
    return ans, ans_dot

def keep_top_k(x, k):
    # Get the top-k values and their indices
    _, top_indices = jax.lax.top_k(x, k)

    # Create a mask with 1s at top-k indices, 0 elsewhere
    mask = jnp.zeros_like(x)
    mask = mask.at[top_indices].set(1)

    return x * mask

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
    
    # jax.debug.print("Rank {rank}: Input values: {input}, Neuron idx: {i}, weights: {w}", rank=rank, input=layer_input, i=neuron_idx, w=weights[neuron_idx])
    # jax.debug.print("Rank {}: Activation values: {}", rank, activations)
    
    # jax.debug.print("Rank {} input_residuals shape: {}, neuron_idx: {}, input: {}", rank, neuron_states.input_residuals.shape, neuron_idx, layer_input)
    new_input_residuals = neuron_states.input_residuals.at[neuron_idx].add(layer_input)
    # jax.debug.print("Rank {} new input_residuals: {}, neuron_idx: {}, input: {}", rank, new_input_residuals, neuron_idx, layer_input)

    def last_layer_case(_):
        return jnp.zeros_like(activations), Neuron_states(
                                            values=activations, 
                                            threshold=neuron_states.threshold, 
                                            input_residuals=new_input_residuals, 
                                            weight_residuals=neuron_states.weight_residuals,
                                            last_sent_iteration=neuron_states.last_sent_iteration
                                            )
    
    def hidden_layer_case(_):
        fire = (iteration-neuron_states.last_sent_iteration) > params.firing_rate
        fire = jnp.logical_or(fire, neuron_idx < 0) # fire if firing rate reached or last input received
        
        top_activations = keep_top_k(activations, params.firing_nb) # Get the top k activations
        # jax.debug.print("Rank {} activations: {}, top activations: {}", rank, activations, top_activations)

        activated_output = jax.lax.cond(fire, 
                                        lambda args: activation_func(args[0], args[1]), 
                                        lambda _: jnp.zeros(activations.shape),
                                        (neuron_states, top_activations))
        layer_activity = neuron_states.weight_residuals["layer activity"]
        
        non_activated_neurons = jnp.where(layer_activity, 0, 1) # Get the neurons of the layer that have not activated yet
        
        # Apply a mask to only allow each neuron to activate once
        output_mask = jax.lax.cond(params.restrict, 
                                   lambda args: jnp.logical_and(jnp.squeeze(args[1]), args[0]), 
                                   lambda args, : jnp.full(args[0].shape, True),
                                   (activated_output, non_activated_neurons))
        
                
        new_layer_activities = jnp.logical_or(jnp.squeeze(layer_activity), activated_output).reshape(layer_activity.shape) # Update the layer activity with the new activated neurons
        
        # jax.debug.print("Rank {}, activations {}, non active neurons: {}, activated output: {}, weight_gradient shape: {}, new layer activities: {}", rank, activations.shape, non_activated_neurons.shape, activated_output.shape, active_gradient.shape, new_layer_activities.shape)
        
        active_output = jnp.where(output_mask > 0, 1, 0) # Cumulating the derivatives of the ReLu
        new_values = neuron_states.weight_residuals["values"].at[neuron_idx].add(active_output)
        
        
        new_input_activities = neuron_states.weight_residuals["input activity"].at[neuron_idx].set(True)            
        new_weight_residuals = {"input activity": new_input_activities, 
                                "layer activity": new_layer_activities,
                                "values": new_values}
        
        new_last_sent_iteration = jax.lax.cond(fire, lambda _: iteration, lambda _: neuron_states.last_sent_iteration, None)
        new_neuron_states = Neuron_states(values=activations - activated_output, 
                                          threshold=neuron_states.threshold, 
                                          input_residuals=new_input_residuals, 
                                          weight_residuals=new_weight_residuals,
                                          last_sent_iteration=new_last_sent_iteration)
        return activated_output*output_mask, new_neuron_states
    
    return jax.lax.cond(rank == size-1, last_layer_case, hidden_layer_case, None)

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
                return send(data, dest=1, tag=0, comm=comm, token=t), count + 1

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
        token = send(jnp.array([-1.0, 0.0]), dest=1, tag=0, comm=comm, token=token)

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
                        lambda t: send(jnp.array([i, out_val]), dest=rank+1, tag=0, comm=comm, token=t),
                        lambda t: t,
                        operand=token
                    )

                token = jax.lax.fori_loop(0, activated_output.shape[0], send_activation, token)
                return token
            
            # Receive neuron values from previous layers and compute the activations
            (neuron_idx, layer_input), token = recv(jnp.zeros((2,)), source=rank-1, tag=0, comm=comm, token=token)
            activated_output, new_neuron_states= layer_computation(neuron_idx.astype(int), layer_input, weights, neuron_states, params, iteration)
            
            # jax.debug.print("Rank {} received data {} at neuron idx: {}", rank, layer_input, neuron_idx)
            neuron_states = new_neuron_states
            
            token = jax.lax.cond(rank==size-1, lambda input: input[0], hidden_layers, (token, activated_output))
            return token, layer_input, neuron_states, neuron_idx, iteration+1
        
        neuron_idx = 0
        layer_input = jnp.zeros(())
        initial_state = (token, layer_input, neuron_states, neuron_idx, 0)
        
        # Loop until the rank receives a -1 neuron_idx
        token, layer_input, neuron_states, neuron_idx, iteration = jax.lax.while_loop(cond, forward_pass, initial_state)
        # jax.debug.print("rank {} exited the while loop with neuron_idx: {} and neuron state {}", rank, neuron_idx, neuron_states)
        
        # Send -1 to the next rank when all incoming data has been processed
        token = jax.lax.cond(
            rank != size - 1,
            lambda t: send(jnp.array([-1.0, 0.0]), dest=rank + 1, tag=0, comm=comm, token=t),
            lambda t: t,
            operand=token
        )
        return token, layer_input, neuron_states, iteration-1
    
    # Loop over batches, accumulate output values and return them
    def loop_over_batches(token, x):
        neuron_states = empty_neuron_states  
        token, layer_input, new_neuron_states, iterations = jax.lax.cond(rank==0, input_layer, other_layers, (token, neuron_states, x))
        
        return token, (new_neuron_states.values, iterations, new_neuron_states)
    
    token, (all_outputs, all_iterations, all_neuron_states) = jax.lax.scan(loop_over_batches, token, batch_data)    
    
    # Synchronize all ranks before starting the backward pass
    token = mpi4jax.barrier(comm=comm, token=token)

    # jax.block_until_ready(all_outputs)
    # jax.debug.print("rank {} finished computing and waiting at the barrier after scanning over {} elements", rank, all_outputs.shape)
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

#region Batched inference
def clone_neuron_state(template: Neuron_states, batch_size: int) -> Neuron_states:
    def tile(x):
        return jnp.broadcast_to(x, (batch_size,) + x.shape)

    return Neuron_states(
        values=tile(template.values),
        threshold=jnp.broadcast_to(template.threshold, (batch_size,)),
        input_residuals=tile(template.input_residuals),
        weight_residuals={
            k: tile(v) for k, v in template.weight_residuals.items()
        },
        last_sent_iteration=tile(template.last_sent_iteration)
    )

@partial(jax.jit, static_argnames=['params'])
def predict_batched(params, weights, empty_neuron_states, token, batch_data: jnp.ndarray):
    """
        sending: (batch_size, 2): list of neuron_index and values
    """
    empty_neuron_states = clone_neuron_state(empty_neuron_states, batch_size)
    
    def input_layer(args):
        token, neuron_states, x = args
        
        x = jax.vmap(lambda x: preprocess_to_sparse_data_padded(x, params.max_nonzero))(x)
        # TODO preprocess the input before calling predict
        
        # Forward pass (Send input data to Layer 1)
        nb_neurons = x.shape[1]
        def cond_send_input(carry):
            i, _ = carry
            out_val = x[:, i]
            return jnp.logical_and(i < nb_neurons, jnp.any(out_val != -2))

        def send_input(carry):
            i, token = carry
            out_val = x[:, i]
            token = send(out_val, dest=rank + 1, tag=0, comm=comm, token=token)
            return i + 1, token

        iteration, token = jax.lax.while_loop(cond_send_input, send_input, (0, token))

        # Send end signal
        token = send(jnp.full((batch_size, 2), -1.0), dest=1, tag=0, comm=comm, token=token)

        return token, jnp.zeros((batch_size)), neuron_states, iteration
    
    def other_layers(args):
        token, neuron_states, _ = args
        def cond(state): # end of input has been reached -> break the while loop
            _, _, _, neuron_idx, _= state   
            # jax.debug.print("Rank {} neuron idx in while cond {}, shape: {}", rank, neuron_idx, neuron_idx.shape)         
            return jnp.all(neuron_idx != -1)
        
        def forward_pass(state):
            token, layer_input, neuron_states, neuron_idx, iteration = state
            
            def hidden_layers(input): # Send activation to the next layers
                token, activated_output = input
                nb_neurons = activated_output.shape[1]
                activated_output = jax.vmap(lambda x: preprocess_to_sparse_data_padded(x, nb_neurons))(activated_output)

                # jax.debug.print("Rank {}, nb neurons: {}", rank, nb_neurons)
                def cond_send_activation(carry):
                    i, _ = carry
                    out_val = activated_output[:, i]
                    return jnp.logical_and(i < nb_neurons, jnp.any(out_val != -2))

                def send_activation(carry):
                    i, token = carry
                    out_val = activated_output[:, i]
                    token = send(out_val, dest=rank + 1, tag=0, comm=comm, token=token)
                    return i + 1, token

                _, token = jax.lax.while_loop(cond_send_activation, send_activation, (0, token))
                return token
            
            # Receive neuron values from previous layers and compute the activations
            input_data, token = recv(jnp.zeros((batch_size, 2), dtype=jnp.float32), source=rank-1, tag=0, comm=comm, token=token)
            # if rank == 2:
            #     jax.debug.print("{}",input_data)
            neuron_idx, layer_input = input_data[:, 0].astype(jnp.int32), input_data[:, 1]
            # jax.debug.print("Rank {} neuron states shape: {} dtype: {}", rank, neuron_states.values.shape, neuron_states.values.dtype)
            # jax.debug.print("Rank {} received data type {} at neuron idx type: {}", rank, layer_input.dtype, neuron_idx.dtype)
            
            activated_output, new_neuron_states = jax.vmap(
                    layer_computation,
                    in_axes=(0, 0, None, 0)  # neuron_idx[batch], layer_input[batch], weights[shared], neuron_states[batch]
                    )(neuron_idx, layer_input, weights, neuron_states, params, iteration)
            
            # activated_output, new_neuron_states = jnp.zeros((layer_sizes[rank])), neuron_states
            # jax.debug.print("Rank {} activated outputs {}", rank, activated_output.shape)
            # jax.debug.print("Rank {} received data {} at neuron idx: {}", rank, layer_input, neuron_idx)
            neuron_states = new_neuron_states
            token = jax.lax.cond(rank==size-1, lambda input: input[0], hidden_layers, (token, activated_output))
            return token, layer_input, neuron_states, neuron_idx, iteration+1
        
        neuron_idx = jnp.zeros((batch_size), dtype=jnp.int32)
        layer_input = jnp.zeros((batch_size))
        initial_state = (token, layer_input, neuron_states, neuron_idx, 0)
        
        # Loop until the rank receives a -1 neuron_idx
        token, layer_input, neuron_states, neuron_idx, iteration = jax.lax.while_loop(cond, forward_pass, initial_state)
        # jax.debug.print("rank {} exited the while loop with neuron_idx: {} and neuron state {}", rank, neuron_idx, neuron_states)
        
        # Send -1 to the next rank when all incoming data has been processed
        token = jax.lax.cond(
            rank != size - 1,
            lambda t: send(jnp.full((batch_size, 2), -1.0), dest=rank + 1, tag=0, comm=comm, token=t),
            lambda t: t,
            operand=token
        )
        return token, layer_input, neuron_states, iteration-1
       
    token, all_outputs, all_neuron_states, all_iterations = jax.lax.cond(rank==0, input_layer, other_layers, (token, empty_neuron_states, batch_data))
    all_outputs = all_neuron_states.values
    
    # Synchronize all ranks before starting the backward pass
    token = mpi4jax.barrier(comm=comm, token=token)

    # jax.block_until_ready(all_outputs)
    # jax.debug.print("rank {} finished computing and waiting at the barrier after scanning over {} elements", rank, all_outputs.shape)
    return token, all_outputs, all_iterations, all_neuron_states

#region Training helpers
def z_gradient(weight_res, next_grad_expanded):
    # weight_res shape: (4, 5)
    # next_grad_expanded shape: (1, 5)

    # Perform element-wise multiplication
    z_grad = weight_res * next_grad_expanded

    return z_grad

def weight_res_complete(activity, values):
    activates = jnp.zeros((values.shape[1], 1))
    for i in reversed(range(len(activity))):
        jax.debug.print("in loop {}", i)
        if not activity[i]: # update if active input
            continue
        values = jnp.where(values[i] > 0, 1, 0)
        activates = jnp.where(activates, 1, 0)
    return jnp.array(values)

@jit
def process_single_batch(activity, values):
    def body(i, carry):
        activates, values = carry

        def update_if_active_fn(carry):
            activates, values = carry

            # Extract row i
            vals = values[i]  # shape: (128,)

            # Case 1: neuron_val == 0 and activates[j] == 1 → set to 1
            condition = (vals == 0) & (activates[:, 0] == 1)
            new_vals = jnp.where(condition, 1, vals)
            values = values.at[i].set(new_vals)

            # Case 2: neuron_val == 1 and activates[j] == 0 → set activates[j] = 1
            update_activates = (vals == 1) & (activates[:, 0] == 0)
            new_activates = jnp.where(update_activates, 1, activates[:, 0])
            activates = new_activates[:, None]

            return activates, values

        return jax.lax.cond(
            jnp.squeeze(activity[i]),
            update_if_active_fn,
            lambda carry: carry,
            operand=(activates, values)
        )

    # Initial state
    activates = jnp.zeros((values.shape[1], 1), dtype=jnp.int32)

    # Reverse loop with fori_loop
    n = activity.shape[0]
    activates, values = jax.lax.fori_loop(
        0, n,
        lambda idx, carry: body(n - 1 - idx, carry),  # reversed order
        (activates, values)
    )
    return values

@partial(jax.jit, static_argnames=['params'])
def predict_bwd(params, key, batch_data, weights, empty_neuron_states, token):
    # jax.debug.print("In predict bwd")
    token, all_outputs, iterations, all_neuron_states = (predict)(params, key, weights, empty_neuron_states, token, batch_data)
    next_grad, token = recv(jnp.zeros((batch_size, layer_sizes[rank])), source=rank + 1, tag=2, comm=comm)

    x = all_neuron_states.input_residuals
    
    # Expand the dimensions of next_grad_transposed to match the shape of all_neuron_states.weight_residuals["values"]
    next_grad_expanded = jnp.expand_dims(next_grad, axis=1)  # Shape: (1, 1, 5)
    
    weight_res = all_neuron_states.weight_residuals
    weight_res = jax.vmap(process_single_batch, in_axes=(0, 0))(weight_res["input activity"], weight_res["values"])
    
    # weight_res = weight_res["values"] # incorrect residual but faster for testing
    
    # Perform element-wise multiplication
    z_grad = jax.vmap(z_gradient, in_axes=(0, 0))(weight_res, next_grad_expanded)
    
    x_reshaped = x[..., jnp.newaxis]   # Shape becomes (batch_size, 4, 1, 1)
    weight_grad = x_reshaped * z_grad[:, :, jnp.newaxis, :] # (batch_size, 4, 1, 5)
    mean_weight_grad = jnp.mean(jnp.squeeze(weight_grad, axis=2), axis=0) # (4, 5)
    
    # jax.debug.print("x {}, x_reshaped{}", x.shape, x)
    # jax.debug.print("next_grad_expanded {}, {}", next_grad_expanded.shape, next_grad_expanded)
    # jax.debug.print("weight residuals {}, {}", weight_res.shape, weight_res)
    # jax.debug.print("z_grad {}, {}", z_grad.shape, z_grad)
    # jax.debug.print("weight_grad {}, mean_weight_grad{}", weight_grad.shape, mean_weight_grad)

    
    if rank > 1:
        jax.debug.print("SENDING DATA TO RANK {}", rank-1)
        token = send(z_grad, dest=rank-1, tag=2,comm=comm, token=token)
    
    return token, all_outputs, iterations, all_neuron_states, mean_weight_grad 

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
    # loss_grad shape: (batch_size, 3)
    # all_residuals shape: (batch_size, 5, 1)

    # Expand dimensions of loss_grad to match all_residuals for broadcasting
    loss_grad_expanded = jnp.expand_dims(loss_grad, axis=1)  # Shape: (batch_size, 1, 3)
    loss_grad_expanded = jnp.expand_dims(loss_grad_expanded, axis=-1)  # Shape: (batch_size, 1, 3, 1)

    # Broadcast and perform element-wise multiplication
    weight_grad = loss_grad_expanded * all_residuals  # Shape: (batch_size, 5, 3)

    return weight_grad

@partial(jax.jit, static_argnames=['params'])
def loss_fn(params, key, batch_data, weights, empty_neuron_states, token, target):
    token, all_outputs, iterations, all_neuron_states = (predict)(params, key, weights, empty_neuron_states, token, batch_data)
    # jax.debug.print("output shape: {}, target shape: {}", all_outputs.shape, target.shape)
    
    # loss = jnp.mean((all_outputs - target) ** 2)
    # N = all_outputs.shape[0]  
    # loss_grad = (2 / N) * (all_outputs - target)
    iterations = iterations.flatten()
    reg_avg_iterations = jnp.mean(iterations)*0.001
    # jax.debug.print("regularized average iterations: {},  {}/{}", reg_avg_iterations, jnp.mean(iterations), jnp.max(iterations))
    loss, loss_grad = jax.value_and_grad(mean_loss)(all_outputs, target)
        
    out_grad = jax.vmap(output_gradient, in_axes=(None, 0))(weights, loss_grad)
    
    all_residuals = all_neuron_states.input_residuals
    # jax.debug.print("weight shape: {} {}", weights.dtype, weights.shape)
    
    weight_grad =  jax.vmap(output_weight_grad, in_axes=(0, 0))(loss_grad, all_residuals)
    mean_weight_grad = jnp.mean(weight_grad, axis=0)
    
    # jax.debug.print("loss: {}, loss gradient: {}", loss.shape, loss_grad.shape)
    # jax.debug.print("out grad {}, {}", out_grad.shape, out_grad)
    # jax.debug.print("all residuals {}, {}", all_residuals.shape, all_residuals.dtype)
    # jax.debug.print("weight_grad {}, mean_weight_grad{}", weight_grad.shape, mean_weight_grad.shape)

    return (loss, all_outputs, iterations), (out_grad, mean_weight_grad)

# region TRAINING
def train(token, params: Params, key, weights, empty_neuron_states):     
    """
    tag 0:  forward computation, data format: (previous_layer_neuron_index, neuron_value)
            end of input is encoded with the index -1
    tag 2: backward computation, last layer gradient shape: (layer_sizes[-1], 1)
    tag 5: weights
    tag 10: data labels(y)
    """   
    if rank == size-1:
        all_epoch_accuracies = []
        all_validation_accuracies = []
        all_loss = []
    all_mean_iterations = []
        
    token = mpi4jax.barrier(comm=comm, token=token)
    start_time = time.time()

    for epoch in range(params.num_epochs):
        key, subkey = jax.random.split(key) 

        if rank == size-1:
            epoch_correct = 0
            epoch_total = 0
            epoch_loss = []
            
        epoch_iterations = []
        if rank == 0:
            batch_iterator = iter(training_generator)
        
        for i in range(total_train_batches):
            neuron_states = empty_neuron_states
            if rank == 0:
                batch_x, batch_y = next(batch_iterator)
                batch_y = jnp.array(batch_y, dtype=jnp.float32)
                # print(f"Batch_x: {batch_x}, Batch_y: {batch_y.shape}")
                
                batch_x, batch_y = pad_batch(batch_x, batch_y, batch_size)
                token = send(batch_y, dest=size-1, tag=10,comm=comm, token=token)

                token, outputs, iterations, all_neuron_states = (predict)(params, subkey, weights, neuron_states, token, batch_data=jnp.array(batch_x))
            else:
                if rank==size-1:
                    # Receive y
                    y, token = recv(jnp.zeros((batch_size,)), source=0, tag=10, comm=comm, token=token)      
                    y_encoded = jnp.array(one_hot_encode(y, num_classes=layer_sizes[-1]))
                    # print("encoded y: ", y, y_encoded.shape, y_encoded)              

                    (loss, outputs, iterations), gradients = (loss_fn)(params, subkey, jnp.zeros((batch_size, layer_sizes[0])), weights, neuron_states, token, y_encoded)
                    epoch_loss.append(loss)
                    
                    weight_grad = gradients[1]
                    
                    # Send gradient to previous layers                
                    token = send(gradients[0], dest=rank-1, tag=2,comm=comm, token=token)
                    
                    valid_y, batch_correct = accuracy(i, outputs, y, iterations, False)                 
                        
                    epoch_correct += batch_correct
                    epoch_total += valid_y.shape[0]
                else:
                    token, outputs, iterations, all_neuron_states, weight_grad = (predict_bwd)(params, subkey, jnp.zeros((batch_size, layer_sizes[0])), weights, neuron_states, token)
                    # jax.debug.print("All neuron states shape: {}, values of first input residuals: {}", all_neuron_states.input_residuals.shape, all_neuron_states.input_residuals[0])
                                    
                # num_zeros = weight_grad.size - jnp.count_nonzero(weight_grad)
                # jax.debug.print("Rank {}, number of zero values in the gradient : {}", rank, num_zeros)
                # jax.debug.print("Rank {}, Weights shape: {}, weight grad shape: {}", rank, (weights.shape), (weight_grad.shape))
                
                # Update weights
                weight_grad = jnp.reshape(weight_grad, (weights.shape[0], weights.shape[1]))
                weights -= params.learning_rate * weight_grad
                # jax.debug.print("Rank {}, new Weights shape {}", rank, (weights.shape))
            epoch_iterations.append(iterations)
            # break
        epoch_iterations = jnp.array(epoch_iterations).flatten()
        mean = jnp.mean(epoch_iterations)
        all_mean_iterations.append(mean)
        
        if rank != 0:
            jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, epoch_iterations.shape[0])
        
        # Inference on the validation set
        val_accuracy, val_mean, _ = batch_predict(params, key, token, weights, empty_neuron_states, dataset="val", save=False, debug=False)

        if rank == size-1:
            # Store loss values
            mean_loss = jnp.mean(jnp.array(epoch_loss))
            all_loss.append(mean_loss)
            
            # Store training and validation accuracies
            epoch_accuracy = epoch_correct / epoch_total
            all_epoch_accuracies.append(epoch_accuracy)
            all_validation_accuracies.append(val_accuracy)
            
            jax.debug.print("Epoch {} , Training Accuracy: {:.2f}%, Validation Accuracy: {:.2f}%, mean loss: {}, mean val iterations: {}", epoch, epoch_accuracy * 100, val_accuracy * 100, mean_loss, val_mean)
            jax.debug.print("----------------------------\n")
    
    # Inference on the test set
    test_accuracy, test_mean, _ = batch_predict(params, key, token, weights, empty_neuron_states, dataset="test", save=False, debug=False)
    
    # Gather the weights and iteration values at the last layer
    weights_dict, all_iteration_mean = gather_weights_and_iterations(weights, jnp.array(all_mean_iterations), token)
    
    jax.block_until_ready(token)

    # Synchronize all MPI processes again
    token = mpi4jax.barrier(comm=comm, token=token)
    end_time = time.time()
        
    if rank==size-1:        
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
                            all_loss)
        
# region SAVE DATA
def store_training_data(params, mode, all_epoch_accuracies, all_validation_accuracies, test_accuracy, execution_time, all_iteration_mean, weights_dict, all_loss):    
    # Choose the saving folder
    if mode == "train":
        result_dir = os.path.join("network_results", "training")
        filename_header = f"{params.random_seed}" + f"_ep{params.num_epochs}" + f"_batch{params.batch_size}_"
        last_iterations_mean = [float(mean[-1]) for mean in all_iteration_mean.tolist()[1:]]
    elif mode == "inference":
        result_dir = os.path.join("network_results", "mnist")
        filename_header = f"{params.random_seed}" + f"_load{params.load_file}" + f"_batch{params.batch_size}_"
        last_iterations_mean = [float(mean) for mean in all_iteration_mean.tolist()[1:]]
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
    result_data = {
        "time": float(execution_time),
        "loadfile": params.load_file,
        "shuffle": params.shuffle,
        "processes": size,
        "firing number": params.firing_nb,
        "firing rate": params.firing_rate,
        "training accuracy": train_accuracy,
        "validation accuracy": val_accuracy,
        "test accuracy": test_accuracy,
        "layer_sizes": params.layer_sizes,
        "batch_size": params.batch_size,
        "learning rate": params.learning_rate,
        "thresholds": params.thresholds,
        "iterations mean": last_iterations_mean,
        "restrict": params.restrict,
        "loss": [float(loss) for loss in all_loss],
        "weights": weights_dict
    }

    with open(result_path + ".json", "w") as f:
        json.dump(result_data, f, indent=4)

    print(f"Results saved to {result_path}")

    if mode == "train":
        epochs = [i + 1 for i in range(params.num_epochs)]        
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
            if i==0:
                continue
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
    valid_mask = ~jnp.isnan(y)
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
    
    if rank != 0:
        neuron_states = Neuron_states(values=jnp.zeros(layer_sizes[rank]), 
                                      threshold=thresholds[rank-1], 
                                      input_residuals=np.zeros((layer_sizes[rank-1], 1)),
                                      weight_residuals={"input activity": jnp.zeros((layer_sizes[rank-1], 1), dtype=bool), 
                                                        "layer activity": jnp.zeros((layer_sizes[rank], 1), dtype=bool), 
                                                        "values": jnp.zeros((layer_sizes[rank-1], layer_sizes[rank]))},
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
                if i == rank-1:
                    weights = jnp.array(w_data[k])
                    print(weights.shape)
                    return weights, neuron_states      
                
                      
        
        # Random initializatoin of the weights       
        weights = random_layer_params(layer_sizes[rank], layer_sizes[rank-1], keys[rank])        
        return weights, neuron_states


def pad_batch(batch_x, batch_y, batch_size):
    # Pad the x data with 0 and the y data with nan for the last batch
    current_size = batch_y.shape[0]
    if current_size < batch_size:
        pad_amount = batch_size - current_size
        pad_y = jnp.full((pad_amount,), jnp.nan)
        pad_x = jnp.zeros((pad_amount, batch_x.shape[1]))  

        batch_y = jnp.concatenate([batch_y, pad_y], axis=0)
        batch_x = jnp.concatenate([batch_x, pad_x], axis=0)
    
    return batch_x, batch_y

def preprocess_data(data_generator):
    # Preprocess the data
    preprocessed = data_generator
    
    return iter(preprocessed)

def gather_weights_and_iterations(weights, mean_iterations, token):
    # Send all the weights to the last layer to store them
    weights_dict = {}
    if rank != size-1:
        token = send(weights, dest=size-1, tag=5,comm=comm, token=token)
    else:
        for i in range(size-1):
            w, token = recv(jnp.zeros((layer_sizes[i-1], layer_sizes[i])), source=i, tag=5, comm=comm, token=token)   
            weights_dict[f"layer_{i}"] = w.tolist()
            
        weights_dict[f"layer_{size-1}"] = weights.tolist()
    
    # Gather all the iterations values at the last layer
    all_iteration_mean, token = mpi4jax.gather(mean_iterations, root=size-1, comm=comm, token=token)
    return weights_dict, all_iteration_mean

# region Main 
def batch_predict(params, key, token, weights, empty_neuron_states, dataset:str="train", save=True, debug=True):        
    token = mpi4jax.barrier(comm=comm, token=token)
    start_time = time.time()
    
    if dataset == "train":
        total_batches = total_train_batches
        if rank == 0:
            print(f"Inference on the training set...")
            batch_iterator = iter(training_generator)
    elif dataset == "val":
        total_batches = total_val_batches
        if rank == 0:
            print(f"Inference on the validation set...")
            batch_iterator = iter(validation_generator)
    elif dataset == "test":
        total_batches = total_test_batches
        if rank == 0:
            print(f"Inference on the test set...")
            batch_iterator = iter(test_generator)
    else:
        print("INVALID DATASET")
        return
        
    if rank == size-1:
        epoch_correct = 0
        epoch_total = 0
            
    epoch_iterations = []
    
    for i in range(total_batches):
        neuron_states = empty_neuron_states
        if rank == 0:                     
            batch_x, batch_y = next(batch_iterator)
            batch_x = jnp.array(batch_x)
            batch_y = jnp.array(batch_y, dtype=jnp.float32)
            # print(f"Batch_x: {batch_x}, {batch_y.dtype}")
            
            batch_x, batch_y = pad_batch(batch_x, batch_y, batch_size)
            
            # token, outputs, iterations, all_neuron_states = (predict_batched)(weights, neuron_states, token, max_nonzero, batch_x)
            token, outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, token, batch_x)
            
            # Send label to the last layer
            token = send(batch_y, dest=size-1, tag=10,comm=comm, token=token)
        else:
            token, outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, token, jnp.zeros((batch_size, layer_sizes[0])))
            # token, outputs, iterations, all_neuron_states = (predict_batched)(weights, neuron_states, token, max_nonzero, jnp.zeros((batch_size, layer_sizes[0])))
            # jax.debug.print("Rank {} All neuron states shape: {}, output shape : {}", rank, all_neuron_states.input_residuals.shape, outputs.shape)

            if rank == size-1:

                y, token = recv(jnp.zeros((batch_size,)), source=0, tag=10, comm=comm, token=token)   
                
                valid_y, batch_correct = accuracy(i, outputs, y, iterations, print=False)                 
                
                epoch_correct += batch_correct
                epoch_total += valid_y.shape[0]
        epoch_iterations.append(iterations)
        # jax.debug.print("Rank {}, iterations: {}", rank, iterations)
        # break
    epoch_iterations = jnp.array(epoch_iterations).flatten()
    mean = jnp.mean(epoch_iterations)
    
    jax.debug.print("Rank {}, all iterations shape: {}", rank, (epoch_iterations.shape[0]))
    if rank != 0 and debug:
        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, epoch_iterations.shape[0])
    
    epoch_accuracy = -1.0
    if rank == size-1:
        epoch_accuracy = epoch_correct / epoch_total
        if debug:
            jax.debug.print("Epoch Accuracy: {:.2f}%", epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")
            
    weights_dict, all_iteration_mean = gather_weights_and_iterations(weights, mean, token)
    # jax.debug.print("rank {} all iterations mean: {}, shape: {}", rank, all_iteration_mean, (all_iteration_mean.shape))
    
    jax.block_until_ready(token)

    # Synchronize all MPI processes again
    token = mpi4jax.barrier(comm=comm, token=token)
    end_time = time.time()

    if rank == size-1:
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
                                [])
    return epoch_accuracy, mean, end_time - start_time
    
if __name__ == "__main__":
    random_seed = 42
    key = jax.random.key(random_seed)
    # Network structure and parameters
    layer_sizes = (28*28, 128, 64, 10)
    layer_sizes = (28*28, 128, 10)
    # layer_sizes = (28*28, 64, 10)
    best = False
    # layer_sizes = [4, 5, 3]
     
    load_file = False
    thresholds = (0, 0 ,0)  
    batch_size = 32
    shuffle = False
    
    if len(layer_sizes) != size:
        print(f"Error: layer_sizes ({len(layer_sizes)}) must match number of MPI ranks ({size})")
        sys.exit(1)

    # test_surrogate_grad()
    for f_nb in [4, 8, 16, 32, 64, 128]:
        # Initialize parameters (input data for rank 0 and weights for other ranks)
        key, subkey = jax.random.split(key) 
        if rank != 0:
            weights, neuron_states = init_params(subkey, load_file=load_file, best=best)
            total_train_batches, total_val_batches, total_test_batches = 0, 0, 0
            batch_iterator = None
            max_nonzero = layer_sizes[rank]
        if rank == 0:
            # Preprocess the data 
            (training_generator, total_train_batches), (validation_generator, total_val_batches), (test_generator, total_test_batches), max_nonzero = torch_loader_manual(batch_size, shuffle=shuffle)
            # training_generator, train_set, test_set, total_batches = torch_loader(batch_size, shuffle=shuffle)
            print("max non zero: ",max_nonzero)
            weights = jnp.zeros((layer_sizes[-1], layer_sizes[0]))
        
        # Broadcast total_batches to all other ranks
        (total_train_batches, total_val_batches, total_test_batches), token = bcast(jnp.array([total_train_batches, total_val_batches, total_test_batches]), root=0 , comm=comm)
    
        params = Params(
            random_seed=random_seed,
            layer_sizes=layer_sizes, 
            thresholds=thresholds, 
            num_epochs=8, 
            learning_rate=0.001, 
            batch_size=batch_size,
            load_file=load_file,
            shuffle=shuffle,
            restrict=False,
            firing_nb=f_nb,
            firing_rate=1,
            max_nonzero=max_nonzero,
            shuffle_input=True
        )
        if rank == 0:
            print(f"Number of training batches: {total_train_batches}, validation batches: {total_val_batches}, test batches: {total_test_batches}")
            print(params)
        
        empty_neuron_states = Neuron_states(
                                values=jnp.zeros((layer_sizes[rank])), 
                                threshold=thresholds[(rank-1)%len(thresholds)], 
                                input_residuals=np.zeros((layer_sizes[rank-1], 1)),
                                weight_residuals={"input activity": jnp.zeros((layer_sizes[rank-1], 1), dtype=bool), 
                                                    "layer activity": jnp.zeros((layer_sizes[rank], 1), dtype=bool), 
                                                    "values": jnp.zeros((layer_sizes[rank-1], layer_sizes[rank]))},
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