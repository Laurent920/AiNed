from socketserver import ThreadingMixIn
from mnist_helper import one_hot, one_hot_encode
from mpi4py import MPI
import jax
import jax.numpy as jnp
from jax import custom_jvp, jit
from jax.tree_util import Partial
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

from mnist_helper import torch_loader_manual
from iris_species_helper import torch_loader

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
    
@dataclasses.dataclass
@tree_math.struct
class Batch_Neuron_states:
    Neuron_states: dict

@dataclasses.dataclass
@tree_math.struct
class Params:
    layer_sizes: list[int]
    thresholds: float
    num_epochs: int
    learning_rate: float
    batch_size: int
    

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
    
@jit
def layer_computation(neuron_idx, layer_input, weights, neuron_states):    
    activations = jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values
    #TODO being able to compute multiple incoming index neurons
    #TODO store the weight residuals of last layer and neuron state of input layer in sparse matrix representation to reduce space utilization because unused
    
    # jax.debug.print("Rank {rank}: Input values: {input}, Neuron idx: {i}, weights: {w}", rank=rank, input=layer_input, i=neuron_idx, w=weights[neuron_idx])
    # jax.debug.print("Rank {}: Activation values: {}", rank, activations)
    
    # jax.debug.print("Rank {} input_residuals shape: {}, neuron_idx: {}, input: {}", rank, neuron_states.input_residuals.shape, neuron_idx, layer_input)
    new_input_residuals = neuron_states.input_residuals.at[neuron_idx].add(layer_input)
    # jax.debug.print("Rank {} new input_residuals: {}, neuron_idx: {}, input: {}", rank, new_input_residuals, neuron_idx, layer_input)

    def last_layer_case(_):
        return jnp.zeros_like(activations), Neuron_states(values=activations, threshold=neuron_states.threshold, input_residuals=new_input_residuals, weight_residuals=neuron_states.weight_residuals)
    
    def hidden_layer_case(_):
        activated_output = activation_func(neuron_states, activations)
        
        # jax.debug.print("Rank {} weight_residuals shape: {}, neuron_idx: {}, input: {}", rank, neuron_states.weight_residuals["values"].shape, neuron_idx, layer_input)
        new_activities = neuron_states.weight_residuals["activity"].at[neuron_idx].set(True)            
        new_values = neuron_states.weight_residuals["values"].at[neuron_idx].add(jnp.where(activated_output > 0, 1, 0))
        new_weight_residuals = {"activity": new_activities, 
                                "values": new_values}
        
        new_neuron_states = Neuron_states(values=activations - activated_output, threshold=neuron_states.threshold, input_residuals=new_input_residuals, weight_residuals=new_weight_residuals)
        return activated_output, new_neuron_states
    
    return jax.lax.cond(rank == size-1, last_layer_case, hidden_layer_case, None)


def predict(weights, empty_neuron_states, token, batch_data: jnp.ndarray):
    #region JAX loop
    def input_layer(args):
        token, neuron_states, x = args
        # Forward pass (Send input data to Layer 1)
        def send_input(i, carry):
            token, count = carry
            data = x[i]
            def send_data(t):
                return send(jnp.array([i, data]), dest=1, tag=0, comm=comm, token=t), count + 1

            def skip_data(t):
                return t, count
            
            token, count = jax.lax.cond(
                data != 0,
                send_data,
                skip_data,
                operand=token
            )
            return token, count

        # Initial carry: (token, iteration=0)
        token, iteration = jax.lax.fori_loop(0, x.shape[0], send_input, (token, 0))

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
            activated_output, new_neuron_states= layer_computation(neuron_idx.astype(int), layer_input, weights, neuron_states)

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

#region Python loop      
def python_predict(weights, empty_neuron_states, token, batch_data: jnp.ndarray):
    # region Python Loop
    all_outputs = jnp.zeros((batch_size, layer_sizes[-1]))
    for batch_nb in range(batch_size):
        neuron_states = empty_neuron_states  
        
        if rank == 0:
            # Forward pass (Send input data to Layer 1)
            x = batch_data[batch_nb]

            for input_neuron_idx, data in enumerate(batch_data[batch_nb]):
                # print(data.dtype)
                if data <= 0:
                    continue
                token = send(jnp.array([input_neuron_idx, data]), dest=1, tag=0, comm=comm, token=token)
            token = send(jnp.array([-1.0, 0.0]), dest=1, tag=0, comm=comm, token=token)
            layer_input = jnp.zeros(())
            # print(f"Input layer: {jnp.shape(x)} {x}")
        else:
            # Simulate a layer running on a separate MPI rank
            while True:
                # Receive input from previous layer
                (neuron_idx, layer_input), token = recv(jnp.zeros((2,)), source=rank-1, tag=0, comm=comm, token=token)
                # print(f"Received data in layer {rank} with neuron_idx: {neuron_idx}, {neuron_idx.dtype}, layer_input: {layer_input}")
                
                # Break if all the inputs have been processed (idx=-1)
                if neuron_idx == -1:
                    if rank == 1:
                        token = send(jnp.array([-1.0, 0.0]), dest=rank+1, tag=0, comm=comm)  # Send -1 to next layer  
                    elif rank == size-1:
                        # Last layer: store the output neurons values
                        all_outputs = all_outputs.at[batch_nb].set(new_neuron_states.values)                      
                    break
                output, new_neuron_states= layer_computation(int(neuron_idx), layer_input, weights, neuron_states)
                neuron_states = new_neuron_states

                if rank == 1:
                    # Hidden layers: Receive input, compute, and send output
                    for idx, out_val in enumerate(output):
                        if out_val <= 0:
                            continue
                        token = send(jnp.array([idx, out_val]), dest=rank+1, tag=0, comm=comm)  # Send output to next layer 
        
    # Synchronize all ranks before starting the backward pass
    token = mpi4jax.barrier(comm=comm, token=token)
    
    return token, all_outputs, all_outputs[-1]

def preprocess_to_sparse_data_padded(x, max_nonzero):
        # Pre-allocate max possible
        processed_data = jnp.full((max_nonzero, 2), -1)
        
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

def clone_neuron_state(template: Neuron_states, batch_size: int) -> Neuron_states:
    def tile(x):
        return jnp.broadcast_to(x, (batch_size,) + x.shape)

    return Neuron_states(
        values=tile(template.values),
        threshold=jnp.broadcast_to(template.threshold, (batch_size,)),
        input_residuals=tile(template.input_residuals),
        weight_residuals={
            k: tile(v) for k, v in template.weight_residuals.items()
        }
    )

@jit
def layer_computation_batched(neuron_idx, layer_input, weights, neuron_states):    
    activations = jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values
    new_input_residuals = neuron_states.input_residuals.at[neuron_idx].add(layer_input)
    
    def last_layer_case(_):
        return jnp.zeros_like(activations), Neuron_states(values=activations, threshold=neuron_states.threshold, input_residuals=new_input_residuals, weight_residuals=neuron_states.weight_residuals)
    
    def hidden_layer_case(_):
        activated_output = activation_func(neuron_states, activations)
        
        # jax.debug.print("Rank {} weight_residuals shape: {}, neuron_idx: {}, input: {}", rank, neuron_states.weight_residuals["values"].shape, neuron_idx, layer_input)
        new_activities = neuron_states.weight_residuals["activity"].at[neuron_idx].set(True)            
        new_values = neuron_states.weight_residuals["values"].at[neuron_idx].add(jnp.where(activated_output > 0, 1, 0))
        new_weight_residuals = {"activity": new_activities, 
                                "values": new_values}
        
        new_neuron_states = Neuron_states(values=activations - activated_output, threshold=neuron_states.threshold, input_residuals=new_input_residuals, weight_residuals=new_weight_residuals)
        return activated_output, new_neuron_states
    
    return jax.lax.cond(rank == size-1, last_layer_case, hidden_layer_case, None)

def predict_batched(weights, empty_neuron_states, token, batch_data: jnp.ndarray):
    def input_layer(args):
        token, neuron_states, x = args
        
        max_nonzero = jnp.max(jnp.array([jnp.count_nonzero(row) for row in x]))
        x = jax.vmap(lambda x: preprocess_to_sparse_data_padded(x, max_nonzero))(x)
        # TODO preprocess the input before calling predict
        
        # Forward pass (Send input data to Layer 1)
        def send_input(i, carry):
            token, count = carry
            
            data = x[:, i]
            return send(data, dest=1, tag=0, comm=comm, token=token), count + 1
        
        # Initial carry: (token, iteration=0)
        token, iteration = jax.lax.fori_loop(0, x.shape[1], send_input, (token, 0))

        # Send end signal
        token = send(jnp.full((batch_size, 2), -1.0), dest=1, tag=0, comm=comm, token=token)

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
            input_data, token = recv(jnp.zeros((batch_size, 2)), source=rank-1, tag=0, comm=comm, token=token)
            neuron_idx, layer_input = input_data[:, 0], input_data[:, 1]
            # activated_output, new_neuron_states= layer_computation(neuron_idx.astype(int), layer_input, weights, neuron_states)
            activated_output, new_neuron_states = jax.vmap(
                    layer_computation_batched,
                    in_axes=(0, 0, None, 0)  # neuron_idx[batch], layer_input[batch], weights[shared], neuron_states[batch]
                    )(neuron_idx, layer_input, weights, neuron_states)

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
    # def loop_over_batches(token, x):
    #     neuron_states = empty_neuron_states  
    #     token, layer_input, new_neuron_states, iterations = jax.lax.cond(rank==0, input_layer, other_layers, (token, neuron_states, x))
        
    #     return token, (new_neuron_states.values, iterations, new_neuron_states)
    
    # token, (all_outputs, all_iterations, all_neuron_states) = jax.lax.scan(loop_over_batches, token, batch_data) 
       
    token, (all_outputs, all_iterations, all_neuron_states) = jax.lax.cond(rank==0, input_layer, other_layers, (token, empty_neuron_states, batch_data))
    
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

def predict_bwd(batch_data, weights, empty_neuron_states, token):
    # jax.debug.print("In predict bwd")
    token, all_outputs, all_iterations, all_neuron_states = jit(predict)(weights, empty_neuron_states, token, batch_data)
    next_grad, token = recv(jnp.zeros((batch_size, layer_sizes[rank])), source=rank + 1, tag=2, comm=comm)

    x = all_neuron_states.input_residuals
    
    # Expand the dimensions of next_grad_transposed to match the shape of all_neuron_states.weight_residuals["values"]
    next_grad_expanded = jnp.expand_dims(next_grad, axis=1)  # Shape: (1, 1, 5)
    weight_res = all_neuron_states.weight_residuals

    # weight_res = weight_res_complete(all_neuron_states.weight_residuals)
    # weight_res = jnp.expand_dims(weight_res_complete(all_neuron_states.weight_residuals), axis=0)
    # jax.debug.print("weight residuals {}, {}", weight_res["values"].shape, weight_res["activity"].shape)

    weight_res = jax.vmap(process_single_batch, in_axes=(0, 0))(weight_res["activity"], weight_res["values"])

    # jax.debug.print("weight residuals {}, {}", weight_res.shape, weight_res)

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
    
    return token, all_outputs, all_iterations, all_neuron_states, mean_weight_grad 

# Define the loss function
def softmax_cross_entropy_with_logits(logits, labels):
    # Compute the softmax in a numerically stable way
    logits_max = jnp.max(logits, axis=0, keepdims=True)
    exps = jnp.exp(logits - logits_max)
    softmax = exps / (jnp.sum(exps, axis=0, keepdims=True) + 1e-8)

    # Compute the cross-entropy loss
    cross_entropy = -jnp.sum(labels * jnp.log(softmax + 1e-8), axis=0)
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

# @jax.custom_vjp
def loss_fn(batch_data, weights, empty_neuron_states, token, target):
    token, all_outputs, all_iterations, all_neuron_states = jit(predict)(weights, empty_neuron_states, token, batch_data)
    # jax.debug.print("output shape: {}, target shape: {}", all_outputs, target)
    
    # loss = jnp.mean((all_outputs - target) ** 2)
    # N = all_outputs.shape[0]  
    # loss_grad = (2 / N) * (all_outputs - target)
    
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

    return (loss, all_outputs, all_iterations), (out_grad, mean_weight_grad)

# Define the forward pass for the custom JVP
# def loss_fn_fwd(batch_data, weights, empty_neuron_states, token, target):
#     token, all_outputs, all_iterations, all_neuron_states = jit(predict)(weights, empty_neuron_states, token, batch_data)
#     jax.debug.print("Rank {}, output: {}, target: {}", rank, all_outputs.shape, target.shape)
#     l = jnp.mean((all_outputs - target) ** 2)
#     # jax.debug.print("loss: {}", l)
#     return (l, token, all_iterations), (all_outputs, target, weights, all_neuron_states)

# # Define the backward pass for the custom JVP
# def loss_fn_bwd(residuals, g):
#     all_outputs, target, weights, all_neuron_states = residuals
    
#     N = all_outputs.shape[1]  # Number of elements in output
    
#     # Gradient of the loss with respect to the output
#     grad_output = g * (2 / N) * (all_outputs - target) # dL/dy_t
#     all_residuals = jnp.array([neuron_states.residuals for neuron_states in all_neuron_states])
#     jax.debug.print("Loss bwd rank {}: loss grad shape: {}, residuals shape: {}, weight grad shape: {}", rank, grad_output.shape, all_residuals.shape, grad_output*all_residuals)
#     return grad_output, grad_output*all_residuals, None, None, None  # Gradient of Loss wrt Output and weights and None for the rest

# # Register the custom JVP with the forward and backward functions
# loss_fn.defvjp(loss_fn_fwd, loss_fn_bwd)        


# region TRAINING
def train(token, params: Params, weights):     
    """
    tag 0:  forward computation, data format: (previous_layer_neuron_index, neuron_value)
            end of input is encoded with the index -1
    tag 2: backward computation, last layer gradient shape: (layer_sizes[-1], 1)
    tag 10: data labels
    tag 20: iteration values
    """   
    if rank == size-1:
        correct_predictions = 0
        total_samples = 0 
        all_epoch_accuracies = []
        
    token = mpi4jax.barrier(comm=comm, token=token)
    start_time = time.time()

    empty_neuron_states = Neuron_states(values=jnp.zeros((layer_sizes[rank])), 
                                        threshold=thresholds[(rank-1)%len(thresholds)], 
                                        input_residuals=jnp.zeros((layer_sizes[rank-1], 1)),
                                        weight_residuals={"activity": jnp.zeros((layer_sizes[rank-1], 1), dtype=bool), 
                                                          "values": jnp.zeros((layer_sizes[rank-1], layer_sizes[rank]))})
    for epoch in range(num_epochs):
        if rank == size-1:
            epoch_correct = 0
            epoch_total = 0
            
        all_iterations = []
        
        for i in range(total_batches):
            neuron_states = empty_neuron_states
            if rank == 0:
                batch_x, batch_y = next(batch_iterator)
                batch_y = jnp.array(batch_y, dtype=jnp.float32)
                # print(f"Batch_x: {batch_x}, Batch_y: {batch_y.shape}")
                
                batch_x, batch_y = pad_batch(batch_x, batch_y, batch_size)
                token = send(batch_y, dest=size-1, tag=10,comm=comm, token=token)

                token, outputs, iterations, all_neuron_states = jit(predict)(weights, neuron_states, token, batch_data=jnp.array(batch_x))
                # jax.debug.print("Rank {} finished computing predict", rank)
                all_iterations.append(iterations) 
            else:
                if rank==size-1:
                    # Receive y
                    y, token = recv(jnp.zeros((batch_size,)), source=0, tag=10, comm=comm, token=token)      
                    y_encoded = jnp.array(one_hot_encode(y, num_classes=layer_sizes[-1]))
                    # print("encoded y: ", y, y_encoded.shape, y_encoded)              

                    # (loss, token, iterations), gradients = jax.value_and_grad(loss_fn)(jnp.zeros((batch_size, layer_sizes[0])), weights, neuron_states, token, y_encoded)
                    (loss, outputs, iterations), gradients = jit(loss_fn)(jnp.zeros((batch_size, layer_sizes[0])), weights, neuron_states, token, y_encoded)

                    weight_grad = gradients[1]
                    
                    # Send gradient to previous layers                
                    token = send(gradients[0], dest=rank-1, tag=2,comm=comm, token=token)
                    
                    valid_y, batch_correct = accuracy(i, outputs, y, iterations, False)                 
                        
                    epoch_correct += batch_correct
                    epoch_total += valid_y.shape[0]
                else:
                    token, outputs, iterations, all_neuron_states, weight_grad = jit(predict_bwd)(jnp.zeros((batch_size, layer_sizes[0])), weights, neuron_states, token)
                    # jax.debug.print("All neuron states shape: {}, values of first input residuals: {}", all_neuron_states.input_residuals.shape, all_neuron_states.input_residuals[0])
                all_iterations.append(iterations)
                
                num_zeros = weight_grad.size - jnp.count_nonzero(weight_grad)
                # jax.debug.print("Rank {}, number of zero values in the gradient : {}", rank, num_zeros)
                # jax.debug.print("Rank {}, Weights shape: {}, weight grad shape: {}", rank, jnp.mean(weights), jnp.mean(weight_grad))
                weight_grad = jnp.reshape(weight_grad, (weights.shape[0], weights.shape[1]))
                weights -= learning_rate * weight_grad
                # jax.debug.print("Rank {}, new Weights shape {}", rank, (weights.shape))
            # break
        all_iterations = jnp.array(all_iterations).flatten()
        mean = jnp.mean(all_iterations)

        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, all_iterations.shape[0])
        
        # Calculate epoch accuracy
        if rank != size-1:
            # jax.debug.print("rank {} send tag 20: i:{}", rank, i)                 
            token = send(mean, dest=size-1, tag=20,comm=comm, token=token)
        else:
            epoch_accuracy = epoch_correct / epoch_total
            all_epoch_accuracies.append(epoch_accuracy)
            correct_predictions += epoch_correct
            total_samples += epoch_total
            
            jax.debug.print("Epoch {} Accuracy: {:.2f}%", epoch, epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")
            
            all_iteration_mean = []
            for i in range(size-1):
                mean, token = recv(jnp.zeros(()), source=i, tag=20, comm=comm, token=token)   
                             
                all_iteration_mean.append(mean)
            all_iteration_mean.append(jnp.mean(all_iterations))

    jax.block_until_ready(token)

    # Synchronize all MPI processes again
    token = mpi4jax.barrier(comm=comm, token=token)
    end_time = time.time()
    
    if rank==size-1:
        plt.figure(figsize=(10, 6))
        plt.plot(all_epoch_accuracies, marker='o', linestyle='-', color='b')
        plt.title('Epoch Accuracies')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)

        # Save the plot as an image file
        plt.savefig('epoch_accuracies.png')
        
        # Final accuracy
        total_accuracy = all_epoch_accuracies[-1]
        jax.debug.print("Final Accuracy: {:.2f}%", total_accuracy * 100)
        
        # Execution time
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.6f} seconds")
        
        # Set up file path
        filename = "_".join(map(str, layer_sizes)) + ".json"
        filename = f"{random_seed}" + f"_ep{num_epochs}" + f"_batch{batch_size}" + filename 
        if best:
            filename = "best_" + filename
        result_dir = os.path.join("network_results", "training")
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, filename)

        # Store the results
        result_data = {
            "time": float(execution_time),
            "processes": size,
            "accuracy": float(total_accuracy),
            "iterations mean": [float(mean) for mean in all_iteration_mean[1:]],
            "layer_sizes": layer_sizes,
            "batch_size": batch_size,
            "thresholds": thresholds,
        }

        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=4)

        print(f"Results saved to {result_path}")
    
    
    
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
                                      weight_residuals={"activity": jnp.zeros((layer_sizes[rank-1], 1), dtype=bool), 
                                                          "values": jnp.zeros((layer_sizes[rank-1], layer_sizes[rank]))})

        if load_file:
            # 
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


def test_surrogate_grad():
    neuron_states = Neuron_states(values=jnp.zeros((3,)), threshold=0.25)
    layer_input = jnp.array([0.1, 0.2, 0.3])
    print(f"input layer_input: {layer_input}, threshold: {neuron_states.threshold}")
    output = activation_func(neuron_states, layer_input)
    
    output, grad_output = jax.vmap(jax.value_and_grad(Partial(activation_func, neuron_states)))(layer_input)
    print(f"output: {output}, output grad: {grad_output}")
    
    # test layer grad
    weights = np.ones((3,3))
    if rank != size-1:
        # def layer_loss(layer_input, weights):
        #     output, _ = layer_computation(layer_input, weights, neuron_states)
        #     return loss_fn(output, grad_output)
        # jax.grad(layer_loss, 1)(layer_input, weights)
        
        # activation, new_neuron_states = layer_computation(layer_input, weights, neuron_states)
        # print(activation, new_neuron_states)
        
        value, grad = jax.vmap(jax.value_and_grad(Partial(layer_computation, weights=weights, neuron_states=neuron_states)))(layer_input)
        print(value, grad)
    
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

# region Main 
def batch_predict(token):
    if rank == size-1:
        correct_predictions = 0
        total_samples = 0 
        all_epoch_accuracies = []
        
    token = mpi4jax.barrier(comm=comm, token=token)
    start_time = time.time()

    empty_neuron_states = Neuron_states(values=jnp.zeros((layer_sizes[rank])), 
                                        threshold=thresholds[(rank-1)%len(thresholds)], 
                                        input_residuals=np.zeros((layer_sizes[rank-1], 1)),
                                        weight_residuals={"activity": jnp.zeros((layer_sizes[rank-1], 1), dtype=bool), 
                                                          "values": jnp.zeros((layer_sizes[rank-1], layer_sizes[rank]))})
    for epoch in range(num_epochs):
        if rank == size-1:
            epoch_correct = 0
            epoch_total = 0
            
        all_iterations = []
        
        for i in range(total_batches):
            neuron_states = empty_neuron_states
            batched_predict = jax.vmap(
                                predict,
                                in_axes=(None, None, None, 0),  # Only batch over `batch_data`
                                out_axes=(None, 0, 0, 0)        # Customize based on outputs
                            )
            if rank == 0:
                batch_x, batch_y = next(batch_iterator)
                batch_x = jnp.array(batch_x)
                batch_y = jnp.array(batch_y, dtype=jnp.float32)
                # print(f"Batch_x: {batch_x}, {batch_y.dtype}")
                
                batch_x, batch_y = pad_batch(batch_x, batch_y, batch_size)
                
                token, outputs, iterations, all_neuron_states = jit(predict_batched)(weights, neuron_states, token, batch_x)
                all_iterations.append(iterations)

                token = send(batch_y, dest=size-1, tag=10,comm=comm, token=token)
            else:
                token, outputs, iterations, all_neuron_states = jit(predict_batched)(weights, neuron_states, token, jnp.zeros((batch_size, layer_sizes[0])))
                # jax.debug.print("All neuron states shape: {}, values of first input residuals: {}", all_neuron_states.input_residuals.shape, all_neuron_states.input_residuals[0])

                all_iterations.append(iterations)

                if rank == size-1:

                    y, token = recv(jnp.zeros((batch_size,)), source=0, tag=10, comm=comm, token=token)   
                    
                    valid_y, batch_correct = accuracy(i, outputs, y, iterations, True)                 
                    
                    epoch_correct += batch_correct
                    epoch_total += valid_y.shape[0]
            # break
        all_iterations = jnp.array(all_iterations).flatten()
        mean = jnp.mean(all_iterations)
        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, all_iterations.shape[0])
        
        # Calculate epoch accuracy
        if rank != size-1:
            # jax.debug.print("rank {} send tag 20: i:{}", rank, i)                 
            token = send(mean, dest=size-1, tag=20,comm=comm, token=token)
        else:
            epoch_accuracy = epoch_correct / epoch_total
            all_epoch_accuracies.append(epoch_accuracy)
            correct_predictions += epoch_correct
            total_samples += epoch_total
            
            jax.debug.print("Epoch {} Accuracy: {:.2f}%", epoch, epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")
            
            all_iteration_mean = []
            for i in range(size-1):
                mean, token = recv(jnp.zeros(()), source=i, tag=20, comm=comm, token=token)   
                             
                all_iteration_mean.append(mean)
            all_iteration_mean.append(jnp.mean(all_iterations))

    jax.block_until_ready(token)

    # Synchronize all MPI processes again
    token = mpi4jax.barrier(comm=comm, token=token)
    end_time = time.time()

    if rank == size-1:            
        # Final accuracy
        total_accuracy = all_epoch_accuracies[-1]
        jax.debug.print("Final Accuracy: {:.2f}%", total_accuracy * 100)
        
        # Execution time
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.6f} seconds")
        
        # Set up file path
        filename = "_".join(map(str, layer_sizes)) + ".json"
        if best:
            filename = "best_" + filename
        result_dir = os.path.join("network_results", "mnist")
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, filename)

        # Store the results
        result_data = {
            "time": float(execution_time),
            "processes": size,
            "accuracy": float(total_accuracy),
            "iterations mean": [float(mean) for mean in all_iteration_mean[1:]],
            "layer_sizes": layer_sizes,
            "batch_size": batch_size,
            "thresholds": thresholds
        }

        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=4)

        print(f"Results saved to {result_path}")
    
if __name__ == "__main__":
    random_seed = 42
    key = jax.random.key(random_seed)
    # Network structure and parameters
    layer_sizes = [28*28, 128, 64, 10]
    layer_sizes = [28*28, 128, 10]
    best = False
    load_file = True
    # layer_sizes = [4, 5, 3] 
    # load_file = False
    thresholds = [0, 0 ,0]  
    num_epochs = 10
    learning_rate = 0.01
    batch_size = 64

    params = Params(
        layer_sizes=layer_sizes, 
        thresholds=thresholds, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate, 
        batch_size=batch_size
    )
    
    if len(layer_sizes) != size:
        print(f"Error: layer_sizes ({len(layer_sizes)}) must match number of MPI ranks ({size})")
        sys.exit(1)

    # test_surrogate_grad()

    # Initialize parameters (input data for rank 0 and weights for other ranks)
    key, subkey = jax.random.split(key) 
    if rank != 0:
        weights, neuron_states = init_params(subkey, load_file=load_file, best=best)
        total_batches = 0
        batch_iterator = None
    if rank == 0:
        # Preprocess the data 
        training_generator, train_set, test_set, total_batches = torch_loader_manual(batch_size, shuffle=False)
        # training_generator, train_set, test_set, total_batches = torch_loader(batch_size, shuffle=False)

        # TODO get max_nonzero from the training set 
        batch_iterator = iter(training_generator)

        weights = jnp.zeros(layer_sizes[0])
        
    # Broadcast total_batches to all other ranks
    total_batches, token = bcast(total_batches, root=0, comm=comm)
    print(f"number of batches: {total_batches}")

    # train(params, token, weights, batch_iterator)
    # init_x = jnp.array(1.0)  # Initial value
    # num_steps = 5  # Number of iterations
    # final_x, results = jax.lax.scan(predict, init_x, xs=None, length=num_steps)
    
    # batch_predict(token)
    train(token, params, weights)