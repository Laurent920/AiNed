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
from tqdm import tqdm
import optuna

import mpi4jax
from mpi4jax import send, recv, bcast

from dataset_helpers.mnist_helper import mnist_loader_manual
from dataset_helpers.shd_helper import torch_SHD_loader
from dataset_helpers.nmnist_helper import torch_nmnist_loader
from dataset_helpers.dvs_helper import torch_DVSGesture_loader
from dataset_helpers.ncars_helper import torch_NCARS_loader
from dataset_helpers.iris_species_helper import torch_iris_loader
from dataset_helpers.network_helper import one_hot_encode

from other_helpers.helpers import Params, NeuronStates
from other_helpers.helpers import accuracy, store_training_data, rerun_init
from other_helpers.helpers import update_history, process_history
from other_helpers.backpropagation import MLP_back_prop
from other_helpers.loss_functions import loss_bpp, mean_loss
from other_helpers.general_MPI_helper import concatenate_model_partition, data_split, model_split 
from other_helpers.general_MPI_helper import forward_send, forward_recv, backward_send, backward_recv, send_labels, recv_labels
from other_helpers.general_MPI_helper import split_batch, gather_batch, combine_batch_avg, gather_model_partition, gather_w_it_th

jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)

TQDM_DISABLE = False
BUFFER_SIZE = 0

# Initialize empty global MPI variables
comm = None
rank = None      
size = None

layer_idx = None           # Rank corresponding to the layer
process_per_layer = None    # Number of processes for each layer
last_layer = None            # Rank of last layer
batch_part_size = None           # The size of the batch on each process
mpi_config = None

training_generator = None
validation_generator = None
test_generator = None

# region INFERENCE
# @custom_jvp # If thresholds == 0 then this behaves as a ReLu activation function 
@jit
def activation_func(neuron_states, activations):
    # return jax.nn.relu(activations)
    return jnp.where(activations > neuron_states.thresholds, activations, 0.0)

# @activation_func.defjvp
# def activation_func_jvp(primals, tangents, k=1.0):
#     # Surrogate gradient, redefine the function for the backward pass
#     neuron_states, activations, = primals
#     neuron_states_dot, activations_dot, = tangents
#     ans = activation_func(neuron_states, activations)
#     ans_dot = jnp.where(activations > neuron_states.thresholds, activations, 0.0)
#     return ans, ans_dot

@partial(jax.jit, static_argnames=['k',])
def keep_top_k(x, k):
    # Get the top-k values and their indices
    k_safe = min(k, x.shape[0]) 
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
    max_len = params.layer_sizes[layer_idx]

    # indices of nonzero values (padded with -2)
    idx = jnp.nonzero(arr, size=max_len, fill_value=-2)[0]
    vals = jnp.where(idx != -2, arr[idx], -2)

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
    
    def do_sort_by_value(pairs):
        # Sort by value (descending), with padding entries at the end
        # Use negative values for descending sort, add large number to padding
        mask = (idx != -2).astype(jnp.int32)
        sort_keys = jnp.where(mask == 1, -vals, 1e10)  # valid: -value, padding: large number
        sorted_pairs = pairs[jnp.argsort(sort_keys)]
        return sorted_pairs

    pairs_out = jax.lax.cond(
        params.shuffle_activations,
        do_shuffle,
        do_sort_by_value, #lambda pairs: pairs,
        operand=pairs
    )

    return pairs_out

@partial(jax.jit, static_argnames=['params', 'grad'])
def layer_computation(params, key, neuron_idx, layer_input, weights, neuron_states, iteration=0, grad=False):    
    # Compute the new values of the neuron states
    activations = jax.lax.cond(neuron_idx < 0,
                            lambda _: neuron_states.values,
                            lambda _: jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values,
                            None
                            )
    #TODO being able to compute multiple incoming index neurons
    #TODO store the weight residuals of last layer and neuron state of input layer in sparse matrix representation to reduce space utilization because unused
    
    # jax.lax.cond(neuron_idx == -1,
    #                 lambda _: jax.debug.print("rank {}, iteration: {}, neuron idx: {}", rank, iteration, neuron_idx),
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

        dummy_activations = jnp.zeros((activations.shape[0], 2))
        return jnp.array(0), dummy_activations, NeuronStates(   values=activations, 
                                                                thresholds=neuron_states.thresholds, 
                                                                input_residuals=new_input_residuals, 
                                                                input_order=neuron_states.input_order, 
                                                                input_activity=new_input_activity,
                                                                layer_activity=neuron_states.layer_activity,
                                                                output_activity=neuron_states.output_activity,
                                                                last_sent_iteration=neuron_states.last_sent_iteration,
                                                                input_vector=neuron_states.input_vector,
                                                                output_vector=neuron_states.output_vector,
                                                                values_history=new_values_history,
                                                                history_index=new_history_index)
    
    @jit
    def hidden_layer_case(_):
        fire = (iteration-neuron_states.last_sent_iteration) >= params.sync_rate # Fire if sync rate reached
        async_fire = jnp.logical_or(params.async_layer < 0, layer_idx <= params.async_layer) # Fire if async_layer or no async_layer condition (-1)
        fire = jnp.logical_and(fire, async_fire) 
        fire = jnp.logical_or(fire, neuron_idx < 0) # Fire if last input received

        # APPLY THE SYNC RATE  
        activated_output = jax.lax.cond(fire, 
                                        lambda args: activation_func(args[0], args[1]), 
                                        lambda _: jnp.zeros(activations.shape),
                                        (neuron_states, activations))
        
        # APPLY THE FIRING NUMBER        
        activated_output = keep_top_k(activated_output, params.firing_nb) # Get the top k activations
        
        # APPLY THE RESTRICTION
        penalty = jax.lax.cond(params.restrict[layer_idx] <= 0,
                               lambda _: activated_output, 
                               lambda _: activated_output*params.restrict[layer_idx], None)
        if grad:
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
            new_layer_activity = neuron_states.layer_activity
            new_input_order = neuron_states.input_order
            new_output_activity = neuron_states.output_activity
            new_input_vector = neuron_states.input_vector
            new_output_vector = neuron_states.output_vector

        new_last_sent_iteration = jax.lax.cond(fire, lambda _: iteration, lambda _: neuron_states.last_sent_iteration, None)
        new_neuron_states = NeuronStates(   values=activations - penalty, 
                                            thresholds=neuron_states.thresholds, 
                                            input_residuals=new_input_residuals, 
                                            input_order=new_input_order, 
                                            input_activity=new_input_activity,
                                            layer_activity=new_layer_activity,
                                            output_activity=new_output_activity,
                                            last_sent_iteration=new_last_sent_iteration,
                                            input_vector=new_input_vector,
                                            output_vector=new_output_vector,
                                            values_history=neuron_states.values_history,
                                            history_index=neuron_states.history_index)
        valid_elements = jnp.count_nonzero(activated_output)
        processed_output = process_activated_output(key, activated_output, params)

        return valid_elements, processed_output, new_neuron_states
    
    cond = layer_idx == last_layer
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

#region Forward Pass
@partial(jax.jit, static_argnames=['params', 'grad',])
def predict(params, key, weights, empty_neuron_states, batch_data: jnp.ndarray, grad=False):
    @jit
    def input_layer(args):
        neuron_states, x = args # x is shape (input_layer_size,)
        x_p = jnp.array(x)

        # @jit
        # def send_input(i, carry):
        #     timestep = carry
        #     data = x_p[i]
        #     @jit
        #     def send_data(t):
        #         # jax.debug.print("rank {} data {}", rank, data)
        #         t, _ = jax.lax.cond(mpi_config.model_part.contain(data),
        #                      lambda _: (t+1, forward_send(mpi_config, data)),
        #                      lambda _: (t, None), None)
        #         # send(data, dest=rank+process_per_layer, tag=0, comm=comm)
        #         return t
            
        #     timestep = jax.lax.cond(
        #         jnp.any(data != -2),
        #         send_data,
        #         lambda _: timestep,
        #         operand=timestep
        #     )
        #     return timestep

        # # Initial carry: (timestep=0)
        # iteration = jax.lax.fori_loop(0, x_p.shape[0], send_input, (0))
        #________________________________________________________________________________
        if params.shuffle_input:
            perm = jax.random.permutation(key, x_p.shape[0])
            x_p = x_p[perm]
            
        def send_input(i, carry):
            count = carry
            data = x_p[i]
            forward_send(mpi_config, data)
            # send(data, dest=rank+process_per_layer, tag=0, comm=comm)
            return i

        def first_not_minus2(row):
            return (row != -2)
        mask = jax.vmap(first_not_minus2)(x_p)
        loop_iterations = (jnp.count_nonzero(mask)/2).astype(int)
        # loop_iterations = x_p.shape[0]
        iteration = jax.lax.fori_loop(0, loop_iterations, send_input, (0))

        # Send end signal
        forward_send(mpi_config, jnp.array([-1.0, -1.0]))
        # send(jnp.array([-1.0, -1.0]), dest=rank+process_per_layer, tag=0, comm=comm)

        return jnp.zeros(()), neuron_states, iteration, jnp.zeros((BUFFER_SIZE, 2))
    @jit
    def other_layers(args):
        neuron_states, _ = args
        def cond(state): # end of input has been reached -> break the while loop
            _, _, finished, _, _= state            
            return jnp.any(finished > -1)
        @jit
        def forward_pass(state):
            layer_input, neuron_states, finished, iteration, buffer = state
            @jit
            def hidden_layers(args): # Send activation to the next layers
                loop_iterations, activated_output = args
                @jit
                def send_activation(i, _):
                    out_val = activated_output[i]
                    forward_send(mpi_config, out_val)
                    # send(out_val, dest=rank+process_per_layer, tag=0, comm=comm)
                    return None
                jax.lax.fori_loop(0, loop_iterations, send_activation, None)
                
                return None
            
            # Receive neuron values from previous layers and compute the activations
            # received_data = forward_recv(mpi_config, jnp.zeros((1,)).shape[0], 2)
            # (neuron_idx, layer_input) = recv(jnp.zeros((2,)), source=rank-process_per_layer, tag=0, comm=comm)
            # (neuron_idx, layer_input) = received[0]
            # loop_iterations, activated_output, new_neuron_states= layer_computation(params, key, neuron_idx.astype(int), layer_input, weights, neuron_states, iteration, grad)

            received_data = forward_recv(mpi_config, mpi_config.nb_previous, 2, finished)
            # neuron_idx, layer_input = received_data[0]
            # finished = finished.at[0].set(neuron_idx)

            # jax.debug.print("rank {}, finished {}", rank, finished)
            def layer_compute_and_send(carry, data_row):
                idx, neuron_states, iteration, finished = carry
                neuron_idx, layer_input = data_row[0], data_row[1]

                finished = jax.lax.cond(finished[idx] <= -1,
                                        lambda _: finished.at[idx].set(-2),
                                        lambda _: finished.at[idx].set(neuron_idx), None)
                # finished = finished.at[idx].set(neuron_idx)
                iteration = jax.lax.cond(neuron_idx < 0, lambda _: iteration, lambda _: iteration+1, operand=None)
                
                # if rank == 2:
                #     jax.debug.print("rank {}, neuron idx: {}, iterations {}", rank, neuron_idx, iteration)
                loop_iterations, activated_output, new_neuron_states = jax.lax.cond(finished[idx] == -2, # Don't fire if partial previous layer has finished sending its data
                                                                            lambda _: (0, jnp.zeros((neuron_states.values.shape[0], 2)), neuron_states),
                                                                            lambda _: layer_computation(params, key, neuron_idx.astype(int), layer_input, weights, 
                                                                                                        neuron_states, iteration, grad),
                                                                            None)
                # if rank == 2:
                #     jax.debug.print("rank {} index {}, iteration {}, output_iter: {}, finished {}", rank, neuron_idx, iteration, loop_iterations, finished[idx])
                jax.lax.cond(layer_idx == last_layer, lambda _: None, hidden_layers, (loop_iterations, activated_output))
                return (idx+1, new_neuron_states, iteration, finished), None

            init_state = (0, neuron_states, iteration, finished) # index = 0
            (_, new_neuron_states, iteration, finished), _ = jax.lax.scan(layer_compute_and_send, init_state, received_data)

            return layer_input, new_neuron_states, finished, iteration, buffer
        
        finished = jnp.zeros(max(mpi_config.nb_previous, 1)) # Prevent shape 0
        layer_input = jnp.zeros(())
        initial_state = (layer_input, neuron_states, finished, 0, jnp.zeros((BUFFER_SIZE, 2)))
        
        # Loop until the rank receives a -1 neuron_idx
        layer_input, neuron_states, finished, iteration, buffer = jax.lax.while_loop(cond, forward_pass, initial_state)
        
        # cond = jnp.logical_and(rank != -1, rank == 5)
        # jax.lax.cond(cond,
        #             lambda _: jax.debug.print("rank {} iteration {}", rank, iteration),
        #             lambda _: None, None)
        # Send -1 to the next rank when all incoming data has been processed
        jax.lax.cond(
            layer_idx != last_layer,
            lambda _: forward_send(mpi_config, jnp.array([-1.0, -1.0]), iteration),
            # lambda _: send(jnp.array([-1.0, -1.0]), dest=rank + process_per_layer, tag=0, comm=comm),
            lambda _: None,
            operand=None
        )
        # jax.debug.print("rank {}, last idx {} finished with iterations {}", rank, layer_idx, iteration)
        return layer_input, neuron_states, iteration-1, buffer

    # jax.debug.print("rank {} data has shape {}", rank, batch_data.shape)

    # Loop over batches, accumulate output values and return them
    @jit
    def loop_over_batches(_, x):
        neuron_states = empty_neuron_states  
        layer_input, new_neuron_states, iterations, buffer = jax.lax.cond(layer_idx==0, input_layer, other_layers, (neuron_states, x))
        
        return None, (new_neuron_states.values, iterations, new_neuron_states, buffer)
    
    _, (all_outputs, all_iterations, all_neuron_states, buffer) = jax.lax.scan(loop_over_batches, None, batch_data)    

    # Synchronize all ranks before starting the backward pass
    mpi4jax.barrier(comm=comm)

    return all_outputs, all_iterations, all_neuron_states, buffer

#region Training helpers
@partial(jax.jit, static_argnames=['params'])
def predict_bwd(params, key, weights, empty_neuron_states, batch_data):
    '''
    B: batch_size
    '''
    all_outputs, iterations, all_neuron_states, buffer = (predict)(params, key, weights, empty_neuron_states, batch_data, grad=True)
    # w_sum = l2_weight_regularization(mpi_config, weights)

    next_grad = backward_recv(mpi_config)
    next_grad = next_grad[:, mpi_config.model_part.start_idx:mpi_config.model_part.end_idx+1] # Shape: (B, layer_size)
    # next_grad = recv(jnp.zeros((batch_part_size, params.layer_sizes[layer_idx])), source=rank + process_per_layer, tag=2, comm=comm) # Shape: (B, 128)
    # jax.debug.print("Rank {} received next_grad shape: {}, next grad mean {}", rank, next_grad.shape, jnp.mean(next_grad))
    weight_grad, th_grad, weight_res = MLP_back_prop(params, all_neuron_states, next_grad, layer_idx)
    weight_grad += 2 * params.w_reg * weights

    if layer_idx > 1:
        send_grad = jnp.dot(next_grad, weights.T) # Shape: (B, 128) @ (128, 784) = (B, 784)
        # jax.debug.print("rank {} send grad hape {} wres shape {} mul shape {}", rank, send_grad.shape, weight_res.shape, (~jnp.all(weight_res == 0, axis=2)).shape )

        send_grad *= (~jnp.all(weight_res == 0, axis=2)) 

        backward_send(mpi_config, send_grad)
        # send(send_grad, dest=rank-process_per_layer, tag=2, comm=comm)
    
    # Sparsity loss gradients 
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
    
    scaling = jax.lax.cond(params.sparsity_impact[layer_idx] > 0,
                           lambda _: params.sparsity_impact[layer_idx] / (all_iterations * batch_part_size * mpi_config.get_process_per_batch),
                           lambda _: 0.0,
                           None)
    
    input_activity = jnp.sum(all_neuron_states.input_activity, axis=0) # Shape (784)
    layer_activity = jnp.sum(all_neuron_states.layer_activity, axis=0) # Shape (128)
    
    layer_activity = gather_batch(layer_activity, mpi_config, average=False) # Gather the weight gradients from all ranks in the split rank
    input_activity = gather_batch(input_activity, mpi_config, average=False)
    
    sparsity_residuals = scaling * layer_activity # Shape: (128,)
    # jax.debug.print("Rank {}, scaling mean: {}, sparsity_residuals mean: {}, sparsity_residuals sum: {}", rank, scaling, jnp.mean(sparsity_residuals), jnp.sum(sparsity_residuals))
    
    th_sparsity_grad = -sparsity_residuals
    weight_sparsity_grad = jnp.outer(input_activity, sparsity_residuals) # Shape: (784, 128)
    # jax.debug.print("Rank {}, th_sparsity_grad: {}, weight_sparsity_grad: {}", rank, jnp.mean(th_sparsity_grad), np.mean(weight_sparsity_grad))
    
    return all_outputs, iterations, all_neuron_states, (weight_grad, th_grad, weight_sparsity_grad, th_sparsity_grad) 

# Define the loss function
@partial(jax.jit, static_argnames=['params'])
def loss_fn(params, key, weights, empty_neuron_states, target, batch_data):
    all_outputs, iterations, all_neuron_states, buffer = (predict)(params, key, weights, empty_neuron_states, batch_data, grad=True)
    # w_sum = l2_weight_regularization(mpi_config, weights)

    full_outputs = gather_model_partition(mpi_config, all_outputs)

    # Compute Loss and loss gradient
    loss, loss_grad = jax.value_and_grad(mean_loss)(full_outputs, target)
    loss_grad /= mpi_config.get_process_per_batch # Shape (B, 10)
    loss_grad = loss_grad[:, mpi_config.model_part.start_idx:mpi_config.model_part.end_idx+1] # Shape (B, 5)
    # loss += params.w_reg * w_sum

    # jax.debug.print("Rank {}, loss grad shape: {}, mpi_config {}", rank, loss_grad.shape, mpi_config.model_part)

    # Compute output gradient and weight gradient
    out_grad, weight_grad = jax.vmap(loss_bpp, in_axes=(None, 0, 0))(weights, all_neuron_states, loss_grad) # Shape (B, 128), (B, 128, 10)
    mean_weight_grad = jnp.mean(weight_grad, axis=0) # Shape: (128, 10)
    mean_weight_grad += 2 * params.w_reg * weights
    mean_weight_grad = jnp.expand_dims(mean_weight_grad, axis=0)  # Shape: (1, 128, 10)
    # Send gradient to previous layers                
    backward_send(mpi_config, out_grad)
    # send(out_grad, dest=rank-process_per_layer, tag=2,comm=comm)
    
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)

    total_loss = loss + sparsity_L 

    acc_history, avg_rank = None, None
    if params.history_size > 0:
        # One-hot target → scalar class index
        target_labels = jnp.argmax(target, axis=-1)
        acc_history, avg_rank = process_history(all_neuron_states.values_history, all_neuron_states.history_index, target_labels)

    # Dummy return for testing purposes
    # jax.debug.print("rank {} all shapes:, loss {} out {} iter {} total loss {} acchist {} avgrank {} meanwgrad {} lossgrad {}", 
    #                 rank, loss, all_outputs.shape, iterations.shape, total_loss, acc_history, avg_rank, mean_weight_grad.shape, loss_grad.shape)
    # return (0, jnp.zeros((batch_part_size, 10)), jnp.zeros(batch_part_size), 0, (None, None)), (jnp.zeros((1, weights.shape[0], 10)), jnp.zeros((batch_part_size, 10)))

    return (loss, full_outputs, iterations, total_loss, (acc_history, avg_rank)), (mean_weight_grad, loss_grad)

def sparsity_loss(params, all_neuron_states, iterations):
    '''
    Compute the sparsity loss based on the input residuals and the weight residuals
    '''
    if all(x <= 0.0 for x in params.sparsity_impact):
        return 0, 1, 0
    
    # Gather all the activations at the last layer to compute the sparsity loss
    leader_rank = layer_idx * process_per_layer
    # jax.debug.print("Rank {}, activations shape: {}, iterations shape: {}, last layer neuron values: {}", rank, (activations.shape), jnp.mean(iterations.shape), jnp.sum(all_neuron_states.values))

    activations = gather_batch(all_neuron_states.input_residuals, mpi_config, average=False) # Gather the weight gradients from all ranks in the split rank
    iterations = gather_batch(iterations, mpi_config, average=True) # Gather the iterations from all ranks in the split rank
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
            all_activations = all_activations + (params.sparsity_impact[i] * act_sum[0]) # Sum of all activations in the hidden layers
            
            if i == 0: # Get iterations of input data
                it_mean = recv(jnp.zeros(1), source=i * process_per_layer, tag=6, comm=comm)
                all_iterations = it_mean[0]
        all_activations += params.sparsity_impact[layer_idx] * jnp.sum(activations) # Adding the activations of the last layer

        sparsity_L = all_activations /  (all_iterations * batch_part_size * process_per_layer)
        # jax.debug.print("Rank {}, sparsity L: {}, all iterations: {}, all activations: {}", rank, sparsity_L, all_iterations, all_activations)
    all_iterations = bcast(all_iterations, root=last_layer*process_per_layer, comm=comm)

    return all_activations, all_iterations, sparsity_L

# region TRAINING
def train(params: Params, key, total_batches, weights, empty_neuron_states, opti, trial=None, readInputJson=False):     
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
    tag 20: communications for gathering, sharing and averaging data across split ranks
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
        solver = optax.sgd(learning_rate=params.learning_rate, momentum=0.9)
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
    th_opt_state = th_solver.init(jax.scipy.special.logit(empty_neuron_states.thresholds))
    
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
            # threshold_grad = 0.0
            if layer_idx == 0: # Input layer
                if readInputJson: # Test with stored input
                    folder_add = "14_sorted_buf"
                    with open(f'pretrained_data/pretrained_data{folder_add}/{len(params.layer_sizes)}hidden_single_input.json') as f:
                        batch_x = np.expand_dims(np.array(json.load(f)).squeeze()[0], axis=0)
                    with open(f'pretrained_data/pretrained_data{folder_add}/{len(params.layer_sizes)}hidden_single_output.json') as f:
                        batch_y = np.expand_dims(np.array(json.load(f)["labels"]).squeeze()[0], axis=0)
                else:
                    batch_x, batch_y = split_batch(params, batch_iterator, mpi_config, 2) # Split the dataset to all the ranks of the input layer
                # print(f"rank {rank} data has shape {(batch_x.shape)}, {(batch_y.shape)}")

                send_labels(mpi_config, batch_y, mpi_config.batch_first_and_last_rank[0]) # Send to the labels to the output layer
                # send(batch_y, dest=mpi_config.batch_first_and_last_rank[1], tag=10,comm=comm) # Destination rank: last rank of the batch
                outputs, iterations, all_neuron_states, buffer = (predict)(params, subkey, weights, neuron_states, batch_data=jnp.array(batch_x))
                all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
            else:
                if mpi_config.is_last_layer: 
                    # Receive the labels from the input layer
                    # y = recv(jnp.zeros((batch_part_size,)), source=mpi_config.batch_first_and_last_rank[0], tag=10, comm=comm)  # Source rank: first rank of the batch
                    y = recv_labels(mpi_config)
                    y_encoded = jnp.array(one_hot_encode(y, num_classes=params.layer_sizes[-1]))

                    # Run the forward and backward pass for the output layer
                    (loss, outputs, iterations, total_loss, history), gradients = (loss_fn)(params, subkey, weights, neuron_states, y_encoded, jnp.zeros((batch_part_size, params.layer_sizes[0])))

                    epoch_loss.append(loss)
                    if params.history_size > 0:
                        all_history.append(history)

                    weight_grad = gradients[0]
                    
                    # Compute and store the accuracy
                    valid_y, batch_correct = accuracy(i, outputs, y, iterations, False)
                    
                    epoch_correct += batch_correct
                    epoch_total += valid_y.shape[0]
                    # weight_grad = gather_batch(weight_grad, mpi_config, average=True)
                    weight_grad = combine_batch_avg(weight_grad, mpi_config) # Gather the weight gradients from all ranks in the split rank
                else:
                    # Run the forward and backward pass for the hidden layers
                    outputs, iterations, all_neuron_states, grads = (predict_bwd)(params, subkey, weights, neuron_states, jnp.zeros((batch_part_size, params.layer_sizes[0])))
                    weight_grad, threshold_grad, weight_sparsity_grad, threshold_sparsity_grad = grads

                    threshold_grad = gather_batch(threshold_grad, mpi_config, average=True) # Gather the weight gradients from all ranks in the split rank
                    
                    # weight_grad = gather_batch(weight_grad, mpi_config, average=True)
                    weight_grad = combine_batch_avg(weight_grad, mpi_config) # Gather the weight gradients from all ranks in the split rank
                    
                    # Add sparsity loss' impact to the gradient if relevant
                    if jnp.any(jnp.array(params.sparsity_impact) > 0):
                        weight_grad = weight_grad + weight_sparsity_grad
                        threshold_grad = threshold_grad + threshold_sparsity_grad

                    # Update thresholds
                    if params.threshold_lr != 0:
                        # print(f"average threshold grad: {jnp.mean(threshold_grad)}")
                        th_updates, th_opt_state = solver.update(threshold_grad, th_opt_state, empty_neuron_states.thresholds)
                        new_thresholds = jax.nn.sigmoid(optax.apply_updates(
                                                            jax.scipy.special.logit(empty_neuron_states.thresholds), th_updates))
                                                                                 
                        empty_neuron_states = NeuronStates(
                                                values=empty_neuron_states.values, 
                                                thresholds=new_thresholds, 
                                                input_residuals=empty_neuron_states.input_residuals, 
                                                input_order=empty_neuron_states.input_order, 
                                                input_activity=empty_neuron_states.input_activity,
                                                layer_activity=empty_neuron_states.layer_activity,
                                                output_activity=empty_neuron_states.output_activity,
                                                last_sent_iteration=empty_neuron_states.last_sent_iteration,
                                                input_vector=empty_neuron_states.input_vector,
                                                output_vector=empty_neuron_states.output_vector,
                                                values_history=empty_neuron_states.values_history,
                                                history_index=empty_neuron_states.history_index)
                # Update weights
                if solver is not None and (params.async_layer < 0 or layer_idx == params.async_layer):
                    # Optax optimizer
                    updates, opt_state = solver.update(weight_grad, opt_state, weights)
                    weights = optax.apply_updates(weights, updates)
            # if i >= 0: # Run a single epoch for testing
            #     return
            epoch_iterations.append(iterations[iterations > 1])
        
        # Compute the average iterations for each layer
        epoch_iterations = jnp.concatenate(epoch_iterations)
        mean = jnp.mean(epoch_iterations)
        all_mean_iterations.append(mean)
        all_mean_iterations = gather_batch(all_mean_iterations, mpi_config)
        all_mean_iterations = all_mean_iterations.tolist()
        
        if layer_idx != 0 and trial is None:
            jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points and a mean threshold of {}", rank, mean, epoch_iterations.shape[0], jnp.mean(empty_neuron_states.thresholds))
        
        # Inference on the validation set
        val_accuracy, val_mean, _ = batch_predict(params, key, total_batches, weights, empty_neuron_states, dataset="val", save=False, debug=False)

        epoch_accuracy = 0.0
        if mpi_config.is_last_layer:
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
            if mpi_config.get_last_layer_batch_leader:
                jax.debug.print("Epoch {} , Training Accuracy: {:.2f}%, Validation Accuracy: {:.2f}%, mean loss: {}, mean val iterations: {}", epoch, all_epoch_accuracies[-1] * 100, val_accuracy * 100, mean_loss, val_mean)
                jax.debug.print("----------------------------\n")
        epoch_accuracy = bcast(epoch_accuracy, root=mpi_config.get_last_layer_batch_leader, comm=comm)
        if epoch_accuracy >= 0.9999:
            break
        
        if trial is not None: # If using Optuna Hyper-parameter tuner
            # Return values if the run is not promising and should be pruned  
            all_mean_it = combine_batch_avg(all_mean_iterations, mpi_config) # Gather the weight gradients from all ranks in the split rank
            all_mean_it = mpi4jax.allgather(all_mean_it, comm=comm)

            val_accuracy = bcast(val_accuracy, root=mpi_config.get_last_layer_batch_leader, comm=comm)
            # jax.debug.print("all mean it: {} {}", all_mean_it, jnp.max(all_mean_it[process_per_layer*2:])/all_mean_it[0])
            normalized_it = (jnp.max(all_mean_it[1:])/all_mean_it[0])
            combined_acc_act = val_accuracy*100 - normalized_it/10
            if jnp.any(all_mean_it==0):
                combined_acc_act = -10

            prune = 0
            if rank == 0:
                report_val = combined_acc_act
                # report_val = val_accuracy
                trial.report(report_val, epoch)
                prune = int(trial.should_prune())
                jax.debug.print("Should prune: {} with val {} (Acc: {}, max_it: {})", prune, report_val, val_accuracy, normalized_it)
            
            prune = bcast(jnp.array(prune), root=0, comm=comm)
            if prune:
                if rank == 0:
                    raise optuna.TrialPruned()
                else:
                    return val_accuracy, normalized_it
                
    # Inference on the test set
    test_accuracy, test_mean, _ = batch_predict(params, key, total_batches, weights, empty_neuron_states, dataset="test", save=False, debug=True)
    
    # Gather the weights and iteration values at the last layer
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_it_th(mpi_config, params, weights, jnp.array(all_mean_iterations), empty_neuron_states.thresholds)
    
    # Synchronize all MPI processes again
    mpi4jax.barrier(comm=comm)
    end_time = time.time()
    
    # Compute processing time and store all the results in a json file
    MAX_LEN = 256
    result_path = jnp.zeros(MAX_LEN, dtype=jnp.uint8)
    if mpi_config.is_last_layer_leader:
        # Execution time
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.6f} seconds")
        result_path_str = store_training_data(
                            size,
                            params, 
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
                            "MLP",
                            all_history,
                            total_batches[0])
        
        encoded = np.frombuffer(result_path_str.encode("utf-8"), dtype=np.uint8)
        if encoded.size > MAX_LEN:
            raise ValueError("result_path too long")
        padded = np.pad(encoded, (0, MAX_LEN - encoded.size), constant_values=0)
        result_path = jnp.array(padded)
    result_path = bcast(result_path, root=mpi_config.get_last_layer_batch_leader, comm=comm)
    result_path = bytes(result_path).decode("utf-8").rstrip("\x00")
    mpi4jax.barrier(comm=comm)

    if trial is not None:
        # If using the Optuna Hyper-parameter tuning return the score for ranking the trials 
        if mpi_config.is_batch_leader:
            all_iteration_mean = jnp.stack(all_iteration_mean)[:,-1]
        else:
            all_iteration_mean = jnp.zeros(mpi_config.get_process_per_batch) # Share iterations mean to the rank 0
        # print("init iteration mean", rank, val_accuracy, all_iteration_mean)

        all_iteration_mean = bcast(all_iteration_mean, root=mpi_config.get_batch_leader, comm=comm)
        
        val_accuracy = bcast(jnp.array(val_accuracy), root=mpi_config.get_batch_leader, comm=comm)
        # print(rank, val_accuracy, all_iteration_mean)
        return val_accuracy, jnp.max((all_iteration_mean[1:]))/all_iteration_mean[0]
    
    return val_accuracy, result_path
    
#region Initialization
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = jax.random.split(key)
    return scale * jax.random.normal(w_key, (n, m))#, scale * jax.random.normal(b_key, (n,))
    # return jnp.full((n, m), 0.1)

def init_params(key, batch_size, layer_sizes, load_file=False, best=False):
    # Initialize weights for each layer
    keys = jax.random.split(key, len(layer_sizes))
    
    if layer_idx != 0:
        if load_file:
            filename = f"tensor_data_{'_'.join(map(str, layer_sizes))}_batch{batch_size}.npz"
            print(f"Loading the weight file from {filename}...")

            if best:
                filename = "best_" + filename
            filepath = os.path.join("tensor_data/MLP/", filename)
            w_data = np.load(filepath)
            for i, k in enumerate(w_data.files):
                if i == layer_idx-1:
                    weights = jnp.array(w_data[k])
                    print(weights.shape)
                    return weights      
        
        # Random initialization of the weights       
        shape = (layer_sizes[layer_idx-1], layer_sizes[layer_idx])
        if len(shape) == 4:
            fan_in = shape[1] * shape[2] * shape[3]  # (out, in, kh, kw)
        elif len(shape) == 2:
            fan_in = shape[0]  # linear layer
        else:
            raise ValueError("Unsupported shape for Kaiming init")
        
        dtype=jnp.float32
        bound = 1.0/jnp.sqrt(fan_in)
        weights = random_layer_params(layer_sizes[layer_idx], layer_sizes[layer_idx-1], keys[layer_idx])
        # print(f"rank {rank} Weights shape: {weights.shape}")
        return weights
    else:
        weights = jnp.zeros((layer_sizes[-1], layer_sizes[0]))
        return weights

# region Inference
def batch_predict(params: Params, key, total_batches, weights, empty_neuron_states: NeuronStates, dataset:str="train", save=True, debug=True, readInputJson=False):    
    '''
    This function implements the forward pass of the neural network

    :param params: Params object holding all the network's parameters
    :param key: JAX random key object 
    :param total_batches: List of batches for train, val and test sets
    :param weights: Network's weights
    :param empty_neuron_states: Initial neuron states 
    :param dataset: Dataset to use (train, val or test)
    :param save: Save the results of the inference
    :param debug: Additionnal debug prints
    :param readInputJson: For testing single inputs
    '''
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
        
    if layer_idx == last_layer:
        epoch_correct = 0
        epoch_total = 0
        all_history = []
    
    epoch_iterations = []
    for i in tqdm(range(total_batches), disable=TQDM_DISABLE):
        neuron_states = empty_neuron_states
        
        if layer_idx == 0:                 
            if readInputJson: # Test with stored input
                folder_add = "14_sorted"
                with open(f'pretrained_data/pretrained_data{folder_add}/{len(params.layer_sizes)}hidden_single_input.json') as f:
                    batch_x = np.array(json.load(f)).squeeze() 
                with open(f'pretrained_data/pretrained_data{folder_add}/{len(params.layer_sizes)}hidden_single_output.json') as f:
                    batch_y = np.array(json.load(f)["labels"]).squeeze()
            else:
                batch_x, batch_y = split_batch(params, batch_iterator, mpi_config, 2)
            # store_data_to_json(f"{len(params.layer_sizes)}hidden_single_input.json", batch_x.tolist()) # Store for hardware usage

            # Run the forward pass
            outputs, iterations, all_neuron_states, buffer = (predict)(params, key, weights, neuron_states, jnp.array(batch_x))

            # Send label to the last layer
            send_labels(mpi_config, batch_y, mpi_config.batch_first_and_last_rank[0])
            # send(batch_y, dest=mpi_config.batch_first_and_last_rank[1], tag=10,comm=comm)
        else:
            # Run forward pass for hidden and output layers
            outputs, iterations, all_neuron_states, buffer = (predict)(params, key, weights, neuron_states, jnp.zeros((batch_part_size, params.layer_sizes[0]))) 
        
            if layer_idx == last_layer:
                # Receive the labels from the input layer and compute the accuracy
                y = recv_labels(mpi_config)
                # y = recv(jnp.zeros((batch_part_size,)), source=mpi_config.batch_first_and_last_rank[0], tag=10, comm=comm)   

                full_outputs = gather_model_partition(mpi_config, outputs)

                valid_y, batch_correct = accuracy(i, full_outputs, y, iterations, rank=rank, print=False)                 
                
                epoch_correct += batch_correct
                epoch_total += valid_y.shape[0]
                # store_data_to_json(f"{len(params.layer_sizes)}hidden_single_output.json", outputs.tolist(), y.tolist())

                if params.history_size > 0: # For history plots
                    # One-hot target → scalar class index
                    history = process_history(all_neuron_states.values_history, all_neuron_states.history_index, y)
                    all_history.append(history)

        # Store for hardware usage
        #     store_data_to_json(f"{len(params.layer_sizes)}hidden_intermediates_layer{rank}.json", outputs.tolist())
        #     store_data_to_json(f"{len(params.layer_sizes)}hidden_event_buffer_layer{rank}.json", buffer.tolist())
        # store_data_to_json(f"{len(params.layer_sizes)}hidden_iterations_layer{rank}.json", iterations.tolist())

        epoch_iterations.append(iterations[iterations > 1])
        # if i >= 0: # Run a single epoch for testing
        #     break
    
    # Compute the average iterations for each layer
    epoch_iterations = jnp.concatenate(epoch_iterations)
    mean = jnp.mean(epoch_iterations)
    mean = gather_batch(mean, mpi_config)
    
    if rank != 0 and debug:
        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, epoch_iterations.shape[0]*mpi_config.get_process_per_batch)
    
    epoch_accuracy = -1.0
    if mpi_config.is_last_layer:
        print(f"epoch correct {epoch_correct}, epoch total: {epoch_total}")
        epoch_accuracy = epoch_correct / epoch_total
        epoch_accuracy = gather_batch(epoch_accuracy, mpi_config)
        if debug and mpi_config.is_last_layer_leader:
            jax.debug.print("Epoch Accuracy: {:.10f}%", epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")
    
    # Gather the weights and iteration values at the last layer
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_it_th(mpi_config, params, weights, mean, empty_neuron_states.thresholds)
    
    # Synchronize all MPI processes again
    mpi4jax.barrier(comm=comm)
    end_time = time.time()

    # Compute processing time and store all the results in a json file if save is True
    if mpi_config.is_last_layer_leader:
        execution_time = end_time - start_time

        if debug:            
            print(f"Execution Time: {execution_time:.6f} seconds")
        if save:
            accuracies = {"train": [-1], "val": [-1], "test": [-1]}
            if dataset in accuracies:
                accuracies[dataset] = [epoch_accuracy]

            store_training_data(size,
                                params, 
                                "inference",
                                accuracies["train"], 
                                accuracies["val"], 
                                accuracies["test"][0],
                                execution_time,
                                all_iteration_mean,
                                weights_dict,
                                [],
                                thresholds_dict,
                                "",
                                "MLP",
                                all_history,
                                total_batches)
    return epoch_accuracy, mean, end_time - start_time

# region Main
def get_split_rank(batch_size, layer_sizes, trial=None):
    '''
    Define each MPI rank's split_rank 

    :param batch_size: Description
    :param layer_sizes: Description
    :param trial: Description
    '''
    global layer_idx 
    global process_per_layer
    global last_layer
    global batch_part_size
    global mpi_config

    mpi_config = data_split(rank, comm, size, batch_size, layer_sizes)
    # mpi_config = model_split(rank, comm, size, batch_size, layer_sizes)
    
    layer_idx = mpi_config.layer_idx
    last_layer = mpi_config.last_layer_idx
    batch_part_size = mpi_config.batch_part.get_size

    mpi_config.print()

def main(random_seed, key, rank_, size_, comm_, trial=None, trial_params=None):   
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

    dataset = 'mnist'
    # dataset = 'shd'
    # dataset = 'nmnist'
    # dataset = 'dvs'
    
    # Network structure and parameters
    all_layers = [] 
    
    match dataset:
        case "mnist":
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

            # all_layers.append((28*28, 128, 64, 32, 10))    
            # all_layers.append((28*28, 64, 64, 64, 10))
            # all_layers.append((28*28, 128, 128, 128, 10))

            # all_layers.append((14*14, 32, 32, 10))
            # all_layers.append((28*28, 32, 32, 10))
            # all_layers.append((28*28, 64, 64, 10))    
            # all_layers.append((28*28, 128, 128, 10))
            # all_layers.append((28*28, 256, 256, 10))

            # all_layers.append((14*14, 32, 10))
            # all_layers.append((28*28, 32, 10))
            # all_layers.append((28*28, 64, 10))    
            # all_layers.append((28*28, 128, 10))    
            all_layers.append((28*28, 256, 10))    
        case "shd":
            # all_layers.append((700, 128, 128, 128, 128, 20))    
            # all_layers.append((700, 256, 256, 256, 256, 20))                

            # all_layers.append((700, 256, 256, 256, 20))                
            # all_layers.append((700, 128, 64, 32, 20))    
            # all_layers.append((700, 64, 64, 64, 20))    
            # all_layers.append((700, 32, 32, 32, 20))  
              
            # all_layers.append((700, 256, 256, 20))                
            # all_layers.append((700, 128, 128, 20))    
            # all_layers.append((700, 64, 64, 20))    
            # all_layers.append((700, 32, 32, 20))    
            
            all_layers.append((700, 256, 20))    
            # all_layers.append((700, 128, 20))    
            # all_layers.append((700, 64, 20))    
            # all_layers.append((700, 32, 20))    
        case "nmnist":
            all_layers.append((34*34*2, 256, 10))   
        case "dvs":
            all_layers.append((128*128*2, 256, 11))
    
    layer_sizes = all_layers[0]
    best = False # Old variable/ not in use anymore
    
    load_file = False # Load pytorch pretrained weights
    batch_size = 36
    restrict = (1,) * len(layer_sizes) # Reset rate for each layer
    # restrict = (2, 2, 2, 2, 1, 1)

    init_thresholds = None
    if trial is not None: # Optuna Hyper-parameter tuning parameter
        dataset = trial_params.dataset
        layer_sizes = trial_params.layer_sizes
        batch_size = trial_params.batch_size
        restrict = trial_params.restrict
        init_thresholds = trial_params.init_thresholds
        
    # if size % len(layer_sizes) != 0:
    #     print(f"Error: layer_sizes ({len(layer_sizes)}) must match number of MPI ranks ({size})")
    #     sys.exit(1)
    
    get_split_rank(batch_size, layer_sizes, trial) # Compute the split rank for training/inference with multiple processes per batch

    for f_nb in [128]: # Loop for multiple experiments
        # Set rerun to a specific file to retrain or rerun inference and None for training from scratch
        rerun = "network_results/mnist/training/MLP/42_ep10_b36_784_128_10_acc0.974_adam_.json"
        rerun = "network_results/mnist/training/MLP/42_ep10_b36_784_128_10_acc0.972_adam_.json"
        rerun = "network_results/mnist/training/MLP/42_ep10_b36_784_128_10_acc0.862_adam_.json"
        # rerun = "network_results/mnist/training/MLP/42_ep10_b36_784_128_10_acc0.865_adam_.json"

        # rerun = 'network_results/shd/training/basic/42_ep10_batch36_700_128_128_20_acc0.612_adam_.json'
        rerun = "network_results/shd/training/MLP/sparsity_loss/42_ep10_batch36_700_128_20_acc0.613_adam_.json"
        
        # T03
        rerun ="network_results/mnist/training/MLP/42_ep40_b36_196_32_10_acc0.968_adam_.json"
        # rerun ="network_results/mnist/training/MLP/42_ep20_b36_196_32_32_10_acc0.959_adam_.json"

        # T0
        # rerun ="network_results/mnist/training/MLP/42_ep20_b36_196_32_10_acc0.965_adam_.json"
        # rerun ="network_results/mnist/training/MLP/42_ep20_b36_196_32_32_10_acc0.960_adam_.json"

        # Restrict
        rerun ="network_results/mnist/training/MLP/42_ep20_b36_196_32_32_10_acc0.952_adam_.json" # RESET2
        
        # Firing nb
        # rerun ="network_results/mnist/training/MLP/42_ep20_b36_196_32_32_10_acc0.948_adam_.json" # FNB 1
        # rerun ="network_results/mnist/training/MLP/42_ep20_b36_196_32_32_10_acc0.958_adam_.json" # FNB 2

        # Sync rate
        # rerun ="network_results/mnist/training/MLP/42_ep20_b36_196_32_32_10_acc0.965_adam_.json" # SYNC 2
        # rerun ="network_results/mnist/training/MLP/42_ep20_b36_196_32_32_10_acc0.966_adam_.json" # SYNC 4
        
        # rerun = "network_results/mnist/training/MLP/basic/load_false/42_ep20_batch36_784_128_10_acc0.976_adam_.json"
        # rerun = "network_results/mnist/training/MLP/basic/load_false/42_ep20_batch36_784_128_128_10_acc0.974_adam_.json"
        
        # Best MNIST
        rerun ="network_results/mnist/training/MLP/42_ep200_b36_784_256_256_10_acc0.981_adam_.json" 

        # rerun = "network_results/shd/training/MLP/w_reg/42_ep1_b36_700_256_20_acc0.431_adam_.json"
        # rerun = "network_results/shd/training/MLP/42_ep100_b36_700_256_256_256_20_acc0.662_adam_.json"
        # rerun = "network_results/shd/training/MLP/42_ep50_b36_700_256_20_acc0.677_adam_.json"
        rerun = None

        async_layer = -1 # Old parameter/ not in use anymore
        # async_layer = 1
        cont = True
        # for i in range(5):#[0.0001, 0.001, 0.01]:
        while cont: # Loop for multiple experiments
                # Initialize parameters (input data for rank 0 and weights for other ranks)
                key, subkey = jax.random.split(key) 
                total_train_batches, total_val_batches, total_test_batches, max_nonzero = 0, 0, 0, 0
                weights = init_params(subkey, batch_size, layer_sizes, load_file=load_file, best=best)
                if rank == 0: # Only the first rank loads the dataset
                    downsample = False
                    # Load the data 
                    match dataset:
                        case "mnist":
                            loader = mnist_loader_manual
                            if layer_sizes[0] == 14*14:
                                downsample = True
                        case "shd":
                            loader = torch_SHD_loader
                        case "nmnist":
                            loader = partial(torch_nmnist_loader, CNN_preprocess=False)
                        case "dvs":
                            loader = partial(torch_DVSGesture_loader, CNN_preprocess=False)                            
                        case "ncars":
                            if layer_sizes[0] == 60 * 50 * 2:
                                downsample = True
                            loader = partial(torch_NCARS_loader, CNN_preprocess=False)
                        case _:
                            raise ValueError(f"Unknown dataset: {dataset}")

                    # Load and unpack the dataloaders
                    train_data, val_data, test_data, max_nonzero = loader(
                                                                        batch_size=batch_size, 
                                                                        shuffle=False, 
                                                                        downsample=downsample
                                                                    )
                    training_generator, total_train_batches = train_data
                    validation_generator, total_val_batches = val_data
                    test_generator, total_test_batches = test_data

                # Broadcast the total number of batches to all other ranks
                total_train_batches, total_val_batches, total_test_batches = bcast(jnp.array([total_train_batches, total_val_batches, total_test_batches]), root=0 , comm=comm)
                max_nonzero = bcast(jnp.array([max_nonzero]), root=0 , comm=comm)
                max_nonzero = max_nonzero.tolist()[0]
                
                key, subkey = jax.random.split(key) 
                if init_thresholds is None:
                    init_thresholds = 0.0
                    # thresholds = jax.random.uniform(subkey, layer_sizes[layer_idx])
                    # thresholds = jax.nn.sigmoid(jax.random.normal(subkey, (layer_sizes[layer_idx])))*init_thresholds
                thresholds = jnp.full(layer_sizes[layer_idx], init_thresholds)

                params = Params(
                    dataset=dataset,
                    random_seed=random_seed,
                    layer_sizes=layer_sizes, 
                    init_thresholds=init_thresholds, 
                    num_epochs=3,
                    learning_rate=0.0001, 
                    batch_size=batch_size,
                    load_file=load_file,
                    shuffle_activations=False,
                    restrict=restrict,
                    firing_nb=1,
                    sync_rate=1,
                    max_nonzero=max_nonzero,
                    shuffle_input=False,
                    threshold_lr=0.0, 
                    sparsity_impact=tuple([0.000, 0.000, 0.000, 0.000, 0.0000]), # Beta sparse
                    w_reg=0.0,
                    rerun="",
                    async_layer=async_layer,
                    history_size=0
                )
                
                if trial is not None:
                    params = dataclasses.replace(trial_params, max_nonzero=max_nonzero)

                folder = "" 
                if rerun is not None:
                    new_epoch_number = 50 # Number of training epoch to run again                    
                    params, weights, thresholds = rerun_init(folder+rerun, 
                                                             mpi_config, 
                                                             params, 
                                                             new_epoch_number, 
                                                             threshold_lr=False, 
                                                             sparsity_impact=False, 
                                                             async_layer=False,
                                                             history_size=True,
                                                             shuffle_activations=False,
                                                             shuffle_input=False)
                
                if rank == 0:
                    print(f"Number of training batches: {total_train_batches}, validation batches: {total_val_batches}, test batches: {total_test_batches}")
                    print(params)
                
                empty_neuron_states = NeuronStates( values=jnp.zeros(layer_sizes[layer_idx]),
                                                    thresholds=thresholds,
                                                    input_residuals=np.zeros((layer_sizes[layer_idx-1],)),
                                                    input_order=jnp.full((layer_sizes[layer_idx-1],), -1, dtype=int), 
                                                    input_activity=jnp.full((layer_sizes[layer_idx-1],), 0, dtype=int),
                                                    layer_activity=jnp.zeros((layer_sizes[layer_idx],), dtype=int),
                                                    output_activity=jnp.zeros((layer_sizes[layer_idx-1], layer_sizes[layer_idx])),
                                                    last_sent_iteration=-1,
                                                    input_vector=jnp.zeros((layer_sizes[layer_idx-1]), dtype=int),
                                                    output_vector=jnp.zeros((layer_sizes[layer_idx]), dtype=int),
                                                    values_history=jnp.zeros((params.history_size, layer_sizes[layer_idx])),
                                                    history_index=jnp.array(0, dtype=jnp.int32))
                weights, empty_neuron_states = mpi_config.MPI_partition(weights, empty_neuron_states)
                total_batches = (total_train_batches, total_val_batches, total_test_batches)

                # batch_predict(params, key, total_batches, weights, empty_neuron_states, "test", save=True, debug=True) # To only run inference
                val_acc, result_path = train(params, key, total_batches, weights, empty_neuron_states, "adam", trial) # To run the full training pipeline

                # rerun = result_path
                # print(rerun)
                break
            
if __name__ == "__main__":
    random_seed = 42
    key = jax.random.key(random_seed)
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()      # Real rank
    size = comm.Get_size()

    main(random_seed, key, rank, size, comm)
