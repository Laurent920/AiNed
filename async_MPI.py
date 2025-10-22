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
from dataset_helpers.iris_species_helper import torch_iris_loader
from dataset_helpers.network_helper import one_hot_encode

from other_helpers.helpers import Params, NeuronStates
from other_helpers.helpers import accuracy, store_training_data, rerun_init
from other_helpers.helpers import update_history, process_history
from other_helpers.backpropagation import back_prop
from other_helpers.loss_functions import loss_bpp, mean_loss
from other_helpers.MPI_helpers import MPIConfig, combine_batch_avg, gather_batch, split_batch

jax.config.update("jax_debug_nans", True)

TQDM_DISABLE = False

# Initialize empty global MPI variables
comm = None
rank = None      
size = None

split_rank = None           # Rank corresponding to the layer
process_per_layer = None    # Number of processes for each layer
last_rank = None            # Rank of last layer
batch_part = None           # The size of the batch on each process
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
    @jit
    def last_layer_case(_):
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
                                            values_history=neuron_states.values_history,
                                            history_index=neuron_states.history_index)
        valid_elements = jnp.count_nonzero(activated_output)
        processed_output = process_activated_output(key, activated_output, params)

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

#region Forward Pass
@partial(jax.jit, static_argnames=['params'])
def predict(params, key, weights, empty_neuron_states, batch_data: jnp.ndarray):
    @jit
    def input_layer(args):
        neuron_states, x = args # x is shape (input_layer_size,)
        
        x_p = x
        @jit
        def send_input(i, carry):
            timestep = carry
            data = x_p[i]
            @jit
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
        #________________________________________________________________________________
        # x_p = jnp.array(x)
        # if params.shuffle_input:
        #     perm = jax.random.permutation(key, x_p.shape[0])
        #     x_p = x_p[perm]
            
        # def send_input(i, carry):
        #     count = carry
        #     data = x_p[i]
        #     send(data, dest=rank+process_per_layer, tag=0, comm=comm)
        #     return i

        # def first_not_minus2(row):
        #     return (row != -2)
        # mask = jax.vmap(first_not_minus2)(x_p)
        # loop_iterations = (jnp.count_nonzero(mask)/2).astype(int)
        # # loop_iterations = x_p.shape[0]
        # # jax.debug.print("input data type {}, {} ", (loop_iterations), len(x_p))
        # iteration = jax.lax.fori_loop(0, loop_iterations, send_input, (0))

        # Send end signal
        send(jnp.array([-1.0, -1.0]), dest=rank+process_per_layer, tag=0, comm=comm)

        return jnp.zeros(()), neuron_states, iteration
    @jit
    def other_layers(args):
        neuron_states, _ = args
        def cond(state): # end of input has been reached -> break the while loop
            _, _, neuron_idx, _= state            
            return neuron_idx != -1
        @jit
        def forward_pass(state):
            layer_input, neuron_states, neuron_idx, iteration = state
            @jit
            def hidden_layers(args): # Send activation to the next layers
                loop_iterations, activated_output = args
                # jax.debug.print("activated output shape: {}, {}", activated_output.shape, activated_output[:, 0])
                # loop_iterations = jnp.count_nonzero(input)
                # activated_output = process_activated_output(key, input, params)
                @jit
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
            lambda _: send(jnp.array([-1.0, -1.0]), dest=rank + process_per_layer, tag=0, comm=comm),
            lambda _: [],
            operand=None
        )
        return layer_input, neuron_states, iteration-1

    # jax.debug.print("rank {} data has shape {}", rank, batch_data.shape)

    # Loop over batches, accumulate output values and return them
    @jit
    def loop_over_batches(_, x):
        neuron_states = empty_neuron_states  
        layer_input, new_neuron_states, iterations = jax.lax.cond(split_rank==0, input_layer, other_layers, (neuron_states, x))
        
        return None, (new_neuron_states.values, iterations, new_neuron_states)
    
    _, (all_outputs, all_iterations, all_neuron_states) = jax.lax.scan(loop_over_batches, None, batch_data)    
    
    # Synchronize all ranks before starting the backward pass
    mpi4jax.barrier(comm=comm)

    return all_outputs, all_iterations, all_neuron_states

#region Training helpers
@partial(jax.jit, static_argnames=['params'])
def predict_bwd(params, key, weights, empty_neuron_states, batch_data):
    '''
    B: batch_size
    '''
    all_outputs, iterations, all_neuron_states = (predict)(params, key, weights, empty_neuron_states, batch_data)
    next_grad = recv(jnp.zeros((batch_part, params.layer_sizes[split_rank])), source=rank + process_per_layer, tag=2, comm=comm) # Shape: (B, 128)
    # jax.debug.print("Rank {} received next_grad shape: {}", rank, next_grad)

    # next_weight_res = jnp.ones((batch_part, params.layer_sizes[split_rank], params.layer_sizes[split_rank+1])) # Shape: (B, 128, 10)
    # # jax.debug.print("Rank {} received next_grad shape: {}, next_weight_res shape: {}", rank, next_grad.shape, next_weight_res.shape)
    # (next_weight_res) = jax.lax.cond(split_rank < last_rank - 1, 
    #                                lambda _: recv(next_weight_res, source=rank + process_per_layer, tag=3, comm=comm),
    #                                lambda _: (next_weight_res), None) 
    # jax.debug.print("Rank {} received next_grad shape: {}", rank, next_weight_res)

    weight_grad, th_grad, weight_res = back_prop(params, all_neuron_states, next_grad, split_rank)

    if split_rank > 1:
        send_grad = jnp.dot(next_grad, weights.T) # Shape: (B, 128) @ (128, 784) = (B, 784)
        # jax.debug.print("rank {} send grad hape {} wres shape {} mul shape {}", rank, send_grad.shape, weight_res.shape, (~jnp.all(weight_res == 0, axis=2)).shape )

        send_grad *= (~jnp.all(weight_res == 0, axis=2)) 
        send(send_grad, dest=rank-process_per_layer, tag=2, comm=comm)
        # send(weight_res, dest=rank-process_per_layer, tag=3, comm=comm)
    
    # Sparsity loss gradients 
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
    
    scaling = jax.lax.cond(params.sparsity_impact[split_rank] > 0,
                           lambda _: params.sparsity_impact[split_rank] / (all_iterations * batch_part * process_per_layer) ,
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
    all_outputs, iterations, all_neuron_states = (predict)(params, key, weights, empty_neuron_states, batch_data)

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

    acc_history, avg_rank = None, None
    if params.history_size > 0:
        # One-hot target → scalar class index
        target_labels = jnp.argmax(target, axis=-1)
        acc_history, avg_rank = process_history(all_neuron_states.values_history, all_neuron_states.history_index, target_labels)

    return (loss, all_outputs, iterations, total_loss, (acc_history, avg_rank)), (mean_weight_grad, loss_grad)

def sparsity_loss(params, all_neuron_states, iterations):
    '''
    Compute the sparsity loss based on the input residuals and the weight residuals
    '''
    if all(x <= 0.0 for x in params.sparsity_impact):
        return 0, 1, 0
    
    # Gather all the activations at the last layer to compute the sparsity loss
    leader_rank = split_rank * process_per_layer
    # jax.debug.print("Rank {}, activations shape: {}, iterations shape: {}, last layer neuron values: {}", rank, (activations.shape), jnp.mean(iterations.shape), jnp.sum(all_neuron_states.values))

    activations = gather_batch(all_neuron_states.input_residuals, mpi_config, average=False) # Gather the weight gradients from all ranks in the split rank
    iterations = gather_batch(iterations, mpi_config, average=True) # Gather the iterations from all ranks in the split rank
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
                it_mean = recv(jnp.zeros(1), source=i * process_per_layer, tag=6, comm=comm)
                all_iterations = it_mean[0]
        all_activations += params.sparsity_impact[split_rank] * jnp.sum(activations) # Adding the activations of the last layer

        sparsity_L = all_activations /  (all_iterations * batch_part * process_per_layer)
        # jax.debug.print("Rank {}, sparsity L: {}, all iterations: {}, all activations: {}", rank, sparsity_L, all_iterations, all_activations)
    all_iterations = bcast(all_iterations, root=last_rank*process_per_layer, comm=comm)

    return all_activations, all_iterations, sparsity_L

# region TRAINING
def train(params: Params, key, total_batches, weights, empty_neuron_states, opti, trial=None):     
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
        all_history = []
    all_mean_iterations = []
    
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
    th_opt_state = th_solver.init(jax.scipy.special.logit(empty_neuron_states.thresholds))
    
    mpi4jax.barrier(comm=comm)
    start_time = time.time()

    for epoch in tqdm(range(params.num_epochs), disable=TQDM_DISABLE):
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
        for i in tqdm(range(total_batches[0]), disable=TQDM_DISABLE):
            neuron_states = empty_neuron_states
            # threshold_grad = 0.0
            if split_rank == 0:
                # print(i)
                batch_x, batch_y = split_batch(params, batch_iterator, mpi_config, 2)
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
                    y_encoded = jnp.array(one_hot_encode(y, num_classes=params.layer_sizes[-1]))
                    # print("encoded y: ", y, y_encoded.shape, y_encoded)              
                    (loss, outputs, iterations, total_loss, history), gradients = (loss_fn)(params, subkey, weights, neuron_states, y_encoded, jnp.zeros((batch_part, params.layer_sizes[0])))

                    epoch_loss.append(loss)
                    if params.history_size > 0:
                        all_history.append(history)

                    weight_grad = gradients[0]
                                        
                    valid_y, batch_correct = accuracy(i, outputs, y, iterations, False)                 
                    
                    epoch_correct += batch_correct
                    epoch_total += valid_y.shape[0]
                    # weight_grad = gather_batch(weight_grad, mpi_config, average=True)
                    weight_grad = combine_batch_avg(weight_grad, mpi_config) # Gather the weight gradients from all ranks in the split rank
                else:
                    outputs, iterations, all_neuron_states, grads = (predict_bwd)(params, subkey, weights, neuron_states, jnp.zeros((batch_part, params.layer_sizes[0])))
                    weight_grad, threshold_grad, weight_sparsity_grad, threshold_sparsity_grad = grads
                    # print(f"rank {rank}, weight_res: {weight_res[0].tolist()}, shape: {weight_res.shape}")

                    # print(f"Rank {rank} finished predict_bwd for batch {i}, outputs shape: {outputs.shape}, iterations: {iterations.shape}, weight_grad shape: {weight_grad.shape}")
                
                    # jax.debug.print("rank {} thresholds grad before: {}", rank, threshold_grad.shape)
                    threshold_grad = gather_batch(threshold_grad, mpi_config, average=True) # Gather the weight gradients from all ranks in the split rank
                    
                    # jax.debug.print("rank {} thresholds grad after: {}", rank, threshold_grad.shape)
                    # weight_grad = gather_batch(weight_grad, mpi_config, average=True)
                    weight_grad = combine_batch_avg(weight_grad, mpi_config) # Gather the weight gradients from all ranks in the split rank
                    
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
                        # print(empty_neuron_states.thresholds)
                
                # print("Rank {}, batch {}, mean weight_grad: {}, max weight_grad: {}, min weight_grad: {}".format(rank, i, jnp.mean(weight_grad), jnp.max(weight_grad), jnp.min(weight_grad)))
                # Update weights
                if solver is not None and (params.async_layer < 0 or split_rank == params.async_layer):
                    # Optax optimizer
                    updates, opt_state = solver.update(weight_grad, opt_state, weights)
                    weights = optax.apply_updates(weights, updates)
            # if i > 10:
            #     break
            epoch_iterations.append(iterations)
        epoch_iterations = jnp.array(epoch_iterations).flatten()
        mean = jnp.mean(epoch_iterations)
        all_mean_iterations.append(mean)
        all_mean_iterations = gather_batch(all_mean_iterations, mpi_config)
        all_mean_iterations = all_mean_iterations.tolist()
        
        if split_rank != 0 and trial is None:
            jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points and a mean threshold of {}", rank, mean, epoch_iterations.shape[0], jnp.mean(empty_neuron_states.thresholds))
        
        # Inference on the validation set
        val_accuracy, val_mean, _ = batch_predict(params, key, total_batches, weights, empty_neuron_states, dataset="val", save=False, debug=False)
        # val_accuracy, val_mean = 0, 0
        epoch_accuracy = 0.0
        if split_rank == last_rank:
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
        
        if trial is not None:
            all_mean_it = combine_batch_avg(all_mean_iterations, mpi_config) # Gather the weight gradients from all ranks in the split rank
            all_mean_it = mpi4jax.allgather(all_mean_it, comm=comm)
            # jax.debug.print("all mean it: {} {}", all_mean_it, jnp.max(all_mean_it[process_per_layer*2:])/all_mean_it[0])
            combined_acc_act = val_accuracy - (jnp.max(all_mean_it[1:])/all_mean_it[0])
            if rank == 0:
                # report_val = combined_acc_act
                report_val = val_accuracy
                trial.report(report_val, epoch)
                print("Should prune: ", trial.should_prune())
                if trial.should_prune():
                    raise optuna.TrialPruned()
    # Inference on the test set
    test_accuracy, test_mean, _ = batch_predict(params, key, total_batches, weights, empty_neuron_states, dataset="test", save=False, debug=True)
    # test_accuracy = 0
    # Gather the weights and iteration values at the last layer
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_it_th(params, weights, jnp.array(all_mean_iterations), empty_neuron_states.thresholds)
    
    # Synchronize all MPI processes again
    mpi4jax.barrier(comm=comm)
    end_time = time.time()
    
    MAX_LEN = 256
    result_path = jnp.zeros(MAX_LEN, dtype=jnp.uint8)
    if rank == last_rank * process_per_layer:
        # Execution time
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.6f} seconds,{all_history}")
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
    result_path = bcast(result_path, root=last_rank*process_per_layer, comm=comm)
    result_path = bytes(result_path).decode("utf-8").rstrip("\x00")
    mpi4jax.barrier(comm=comm)

    if trial is not None:
        leader_rank = last_rank * process_per_layer
        if rank != leader_rank:
            all_iteration_mean = jnp.zeros(size//process_per_layer) # Share iterations mean to the rank 0
        else:
            all_iteration_mean = jnp.stack(all_iteration_mean)[:,-1]
        # print("init iteration mean", rank, val_accuracy, all_iteration_mean)

        all_iteration_mean = bcast(all_iteration_mean, root=leader_rank, comm=comm)
        
        val_accuracy = bcast(jnp.array(val_accuracy), root=leader_rank, comm=comm)
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
    
    if split_rank != 0:
        if load_file:
            filename = f"tensor_data_{'_'.join(map(str, layer_sizes))}_batch{batch_size}.npz"
            print(f"Loading the weight file from {filename}...")

            if best:
                filename = "best_" + filename
            filepath = os.path.join("tensor_data/MLP/", filename)
            w_data = np.load(filepath)
            for i, k in enumerate(w_data.files):
                if i == split_rank-1:
                    weights = jnp.array(w_data[k])
                    print(weights.shape)
                    return weights      
        
        # Random initializatoin of the weights       
        shape = (layer_sizes[split_rank], layer_sizes[split_rank-1])
        if len(shape) == 4:
            fan_in = shape[1] * shape[2] * shape[3]  # (out, in, kh, kw)
        elif len(shape) == 2:
            fan_in = shape[1]  # linear layer
        else:
            raise ValueError("Unsupported shape for Kaiming init")
        
        dtype=jnp.float32
        bound = jnp.sqrt(6.0 / fan_in)
        # return jax.random.uniform(jax.random.PRNGKey(0), shape, dtype, -bound, bound)
        weights = random_layer_params(layer_sizes[split_rank], layer_sizes[split_rank-1], keys[split_rank])        
        return weights
    else:
        weights = jnp.zeros((layer_sizes[-1], layer_sizes[0]))
        return weights


def gather_w_it_th(params, weights, mean_iterations, thresholds):
    # Gather all the weights and iteration values at the last layer to store them
    leader_rank = split_rank * process_per_layer

    weights_dict = {}
    all_iteration_mean = []
    thresholds_dict = {}
    
    # print(rank, thresholds.shape, mean_iterations)
    if split_rank != last_rank and rank == leader_rank:
        send(mean_iterations, dest=last_rank * process_per_layer, tag=5,comm=comm)
        if split_rank != 0:
            send(weights, dest=last_rank * process_per_layer, tag=5,comm=comm)
            send(thresholds, dest=last_rank * process_per_layer, tag=5,comm=comm)

    elif split_rank == last_rank and rank == leader_rank:
        for i in range(last_rank):
            # Storing mean iterations
            it_mean = recv(mean_iterations, source=i * process_per_layer, tag=5, comm=comm)
            all_iteration_mean.append(it_mean)
            if i==0: 
                continue

            # Storing the weights 
            w = recv(jnp.zeros((params.layer_sizes[i-1], params.layer_sizes[i])), source=i * process_per_layer, tag=5, comm=comm)   
            weights_dict[f"layer_{i}"] = w.tolist()
            
            # Storing the thresholds
            thr = recv(jnp.zeros(params.layer_sizes[i]), source=i * process_per_layer, tag=5, comm=comm)
            if i == 0: continue  # Skip the input layer thresholds
            thresholds_dict[f"thresholds_{i}"]= thr.tolist()
            
        all_iteration_mean.append(mean_iterations)  # Append the mean iterations of the last layer
        weights_dict[f"layer_{last_rank}"] = weights.tolist()

        print("all iteration mean: rank", rank, all_iteration_mean)

    return weights_dict, all_iteration_mean, thresholds_dict

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
    return
    
# region Inference
def batch_predict_time(params, key, total_batches, weights, empty_neuron_states, dataset:str="train", save=True, debug=True):    
    global training_generator
    global validation_generator
    global test_generator    

    mpi4jax.barrier(comm=comm)
    start_time = time.time()
    
    if dataset == "train":
        total_batches = total_batches[0]
        if split_rank == 0:
            batch_iterator = None
            if rank == 0:
                print(f"Inference on the training set...")
                batch_iterator = iter(training_generator)
    elif dataset == "val":
        total_batches = total_batches[1]
        if split_rank == 0:
            batch_iterator = None
            if rank == 0:
                print(f"Inference on the validation set...")
                batch_iterator = iter(validation_generator)
    elif dataset == "test":
        total_batches = total_batches[2]
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
        all_history = []
    
    all_runtimes = []
    all_activations = []
    epoch_iterations = []
    for i in tqdm(range(total_batches), disable=TQDM_DISABLE):
        neuron_states = empty_neuron_states
        
        if split_rank == 0:                 
            batch_x, batch_y = split_batch(params, batch_iterator, mpi_config, 2)
            # print(f"batch {i} has shape {batch_x.shape}, {batch_y.shape}")
        
        mpi4jax.barrier(comm=comm)
        start_predict_time = time.time()   
        if split_rank == 0:                 
            outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, jnp.array(batch_x))
            end_predict_time = time.time()

            # Send label to the last layer
            send(batch_y, dest=last_rank * process_per_layer + rank, tag=10,comm=comm)
        else:
            outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, jnp.zeros((batch_part, params.layer_sizes[0]))) 
            end_predict_time = time.time()
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

                if params.history_size > 0:
                    # One-hot target → scalar class index
                    history = process_history(all_neuron_states.values_history, all_neuron_states.history_index, y)
                    all_history.append(history)

        epoch_iterations.append(iterations)
        # jax.debug.print("Rank {}, iterations: {}", rank, iterations)
        # if i > 5:
        #     break
    
    # print(f"Shape iterations before flattening: {jnp.array(epoch_iterations).shape}")
    epoch_iterations = jnp.array(epoch_iterations).flatten()
    mean = jnp.mean(epoch_iterations)
    # print(f"Rank {rank} finished epoch with mean {mean} with {epoch_iterations.shape} iterations")

    if split_rank != 0:
        mean = gather_batch(mean, mpi_config)
    # jax.debug.print("Rank {}, all iterations shape: {}", rank, (epoch_iterations.shape[0]))
    
    if rank != 0 and debug:
        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, epoch_iterations.shape[0]*process_per_layer)
    
    epoch_accuracy = -1.0
    if split_rank == last_rank:
        epoch_accuracy = epoch_correct / epoch_total
        epoch_accuracy = gather_batch(epoch_accuracy, mpi_config)
        if debug:
            jax.debug.print("Epoch Accuracy: {:.2f}%", epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")
    
    compute_runtime_plot(all_runtimes, all_activations)
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_it_th(params, weights, mean, empty_neuron_states.thresholds)
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


def batch_predict(params, key, total_batches, weights, empty_neuron_states, dataset:str="train", save=True, debug=True):    
    global training_generator
    global validation_generator
    global test_generator    

    mpi4jax.barrier(comm=comm)
    start_time = time.time()
    
    if dataset == "train":
        total_batches = total_batches[0]
        if split_rank == 0:
            batch_iterator = None
            if rank == 0:
                print(f"Inference on the training set...")
                batch_iterator = iter(training_generator)
    elif dataset == "val":
        total_batches = total_batches[1]
        if split_rank == 0:
            batch_iterator = None
            if rank == 0:
                print(f"Inference on the validation set...")
                batch_iterator = iter(validation_generator)
    elif dataset == "test":
        total_batches = total_batches[2]
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
        all_history = []
    
    epoch_iterations = []
    for i in tqdm(range(total_batches), disable=TQDM_DISABLE):
        neuron_states = empty_neuron_states
        
        if split_rank == 0:                 
            batch_x, batch_y = split_batch(params, batch_iterator, mpi_config, 2)
            # print(f"batch {i} has shape {batch_x.shape}, {batch_y.shape}")                 
            outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, jnp.array(batch_x))

            # Send label to the last layer
            send(batch_y, dest=last_rank * process_per_layer + rank, tag=10,comm=comm)
        else:
            outputs, iterations, all_neuron_states = (predict)(params, key, weights, neuron_states, jnp.zeros((batch_part, params.layer_sizes[0]))) 
            # jax.debug.print("Rank {} All neuron states shape: {}, output shape : {}", rank, all_neuron_states.input_residuals.shape, outputs.shape)
        
        if split_rank != 0:
            if split_rank == last_rank:
                y = recv(jnp.zeros((batch_part,)), source=rank - (last_rank * process_per_layer), tag=10, comm=comm)   
                
                valid_y, batch_correct = accuracy(i, outputs, y, iterations, print=False)                 
                
                epoch_correct += batch_correct
                epoch_total += valid_y.shape[0]
                
                if params.history_size > 0:
                    # One-hot target → scalar class index
                    history = process_history(all_neuron_states.values_history, all_neuron_states.history_index, y)
                    all_history.append(history)

        epoch_iterations.append(iterations)
        # jax.debug.print("Rank {}, iterations: {}", rank, iterations)
        # if i > 10:
        #     break
    
    # print(f"Shape iterations before flattening: {jnp.array(epoch_iterations).shape}")
    epoch_iterations = jnp.array(epoch_iterations).flatten()
    mean = jnp.mean(epoch_iterations)
    # print(f"Rank {rank} finished epoch with mean {mean} with {epoch_iterations.shape} iterations")

    if split_rank != 0:
        mean = gather_batch(mean, mpi_config)
    # jax.debug.print("Rank {}, all iterations shape: {}", rank, (epoch_iterations.shape[0]))
    
    if rank != 0 and debug:
        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, epoch_iterations.shape[0]*process_per_layer)
    
    epoch_accuracy = -1.0
    if split_rank == last_rank:
        epoch_accuracy = epoch_correct / epoch_total
        epoch_accuracy = gather_batch(epoch_accuracy, mpi_config)
        if debug:
            jax.debug.print("Epoch Accuracy: {:.2f}%", epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")
    
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_it_th(params, weights, mean, empty_neuron_states.thresholds)
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
    global split_rank 
    global process_per_layer
    global last_rank
    global batch_part
    global mpi_config

    last_rank = len(layer_sizes)-1
    process_per_layer = size // (last_rank+1)
    split_rank = rank // process_per_layer
    batch_part = batch_size // process_per_layer

    mpi_config = MPIConfig(
        rank=rank,
        split_rank=split_rank,
        last_rank=last_rank,
        process_per_layer=process_per_layer,
        batch_part=batch_part,
        comm=comm
    )
    if trial is None:
        print(f"Rank {rank}, split rank: {split_rank}, batch part: {batch_part}, process per layer: {process_per_layer}, last rank: {last_rank}")

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

    # all_layers.append((28*28, 128, 64, 32, 10))    
    # all_layers.append((28*28, 64, 64, 64, 10))
    # all_layers.append((28*28, 128, 128, 128, 10))
    
    # all_layers.append((28*28, 32, 32, 10))
    # all_layers.append((28*28, 64, 64, 10))    
    all_layers.append((28*28, 128, 10))

    # all_layers.append((28*28, 32, 10))
    # all_layers.append((28*28, 64, 10))    
    # all_layers.append((28*28, 128, 10))    
    
    # SHD layers 
    # all_layers.append((700, 128, 128, 128, 128, 20))    

    # all_layers.append((700, 128, 64, 32, 20))    
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

    init_thresholds = None
    if trial is not None:
        dataset = trial_params.dataset
        layer_sizes = trial_params.layer_sizes
        batch_size = trial_params.batch_size
        restrict = trial_params.restrict
        init_thresholds = trial_params.init_thresholds
        
    if size % len(layer_sizes) != 0:
        print(f"Error: layer_sizes ({len(layer_sizes)}) must match number of MPI ranks ({size})")
        sys.exit(1)
    
    get_split_rank(batch_size, layer_sizes, trial) # Compute the split rank for training/inference with multiple processes per batch

    if batch_size % process_per_layer != 0:
        print(f"Error: one batch ({batch_size}) must be divisible by the number of processes per layer ({process_per_layer})")
        sys.exit(1)
    
    for f_nb in [128]:
        # restrict = (r,) * len(layer_sizes)    
        key, subkey = jax.random.split(key) 
        if init_thresholds is not None:
            init_thresholds = 0.0
            # thresholds = jax.random.uniform(subkey, layer_sizes[split_rank])
            # thresholds = jax.nn.sigmoid(jax.random.normal(subkey, (layer_sizes[split_rank])))*init_thresholds
        thresholds = jnp.full(layer_sizes[split_rank], init_thresholds)
        
        rerun = "network_results/mnist/training/constant_layer_activation_test_with_fnb2/42_ep2_batch36_784_128_10_acc0.936_adam_.json"
        # rerun = "network_results/mnist/training/constant_layer_activation_test_with_fnb2/42_ep2_batch36_784_128_128_10_acc0.920_adam_.json"
        # rerun = "network_results/mnist/training/constant_layer_activation_test_with_fnb2/42_ep2_batch36_784_128_128_128_10_acc0.955_adam_.json"
        # rerun = "network_results/mnist/training/constant_layer_activation_test_with_fnb2/42_ep2_batch36_784_128_128_128_128_10_acc0.948_adam_.json"
        # rerun = "network_results/mnist/training/constant_layer_activation_test_with_fnb2/42_ep2_batch36_784_128_128_128_128_128_10_acc0.940_adam_.json"
        # rerun = "network_results/mnist/training/constant_layer_activation_test_with_fnb2/42_ep2_batch36_784_128_128_128_128_128_128_10_acc0.933_adam_.json"
        # rerun = "network_results/mnist/training/constant_layer_activation_test_with_fnb2/42_ep2_batch36_784_128_128_128_128_128_128_128_10_acc0.918_adam_.json"
        # rerun = "network_results/shd/training/ReLu_threshold_trained/42_ep20_batch36_700_128_128_20_acc0.580_adam_.json"
        rerun = "network_results/mnist/training/MLP/basic/load_false/42_ep20_batch36_784_128_10_acc0.976_adam_.json"
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
                total_train_batches, total_val_batches, total_test_batches, max_nonzero = 0, 0, 0, 0
                weights = init_params(subkey, batch_size, layer_sizes, load_file=load_file, best=best)
                if rank == 0:
                    # Load the data 
                    match dataset:
                        case "mnist":
                            loader = mnist_loader_manual
                        case "shd":
                            loader = torch_SHD_loader
                        case _:
                            raise ValueError(f"Unknown dataset: {dataset}")
        
                    (training_generator, total_train_batches), (validation_generator, total_val_batches), (test_generator, total_test_batches), max_nonzero = loader(batch_size, shuffle=False)
                
                # Broadcast total_batches to all other ranks
                total_train_batches, total_val_batches, total_test_batches = bcast(jnp.array([total_train_batches, total_val_batches, total_test_batches]), root=0 , comm=comm)
                max_nonzero = bcast(jnp.array([max_nonzero]), root=0 , comm=comm)
                max_nonzero = max_nonzero.tolist()[0]
                    
                params = Params(
                    dataset=dataset,
                    random_seed=random_seed,
                    layer_sizes=layer_sizes, 
                    init_thresholds=init_thresholds, 
                    num_epochs=1, 
                    learning_rate=0.0001, 
                    batch_size=batch_size,
                    load_file=load_file,
                    shuffle_activations=False,
                    restrict=restrict,
                    firing_nb=f_nb,
                    sync_rate=1000,
                    max_nonzero=max_nonzero,
                    shuffle_input=False,
                    threshold_lr=0.01, 
                    sparsity_impact=tuple([0.0000, 0.0000, 0.0000, 0.0000, 0.0000]), # Beta sparse
                    rerun="",
                    async_layer=async_layer,
                    history_size=0
                )
                
                if trial is not None:
                    params = dataclasses.replace(trial_params, max_nonzero=max_nonzero)

                folder = "" #"network_results/training/"
                # rerun = "42_ep20_batch36_784_128_64_10_acc0.967_adam_.json"
                # rerun = "42_ep20_batch36_784_128_64_10_acc0.973_adam_.json"
                # rerun = "42_ep1_batch36_784_128_64_10_acc0.799_adam_.json"
                # rerun = None
                if rerun is not None:
                    new_epoch_number = 10 # Number of training epoch to run again
                    th_lr, beta = 1, 0.0
                    
                    params, weights, thresholds = rerun_init(folder+rerun, 
                                                             mpi_config, 
                                                             params, 
                                                             new_epoch_number, 
                                                             threshold_lr=True, 
                                                             sparsity_impact=True, 
                                                             async_layer=True)
                    if len(layer_sizes) != len(params.layer_sizes):
                        print(f"Error: rerun file {rerun} has different layer sizes than the current network structure {layer_sizes}.")
                        sys.exit(1)
                
                if rank == 0:
                    print(f"Number of training batches: {total_train_batches}, validation batches: {total_val_batches}, test batches: {total_test_batches}")
                    print(params)
                
                # print(rank, layer_sizes, thresholds.shape)
                empty_neuron_states = NeuronStates( values=jnp.zeros(layer_sizes[split_rank]),
                                                    thresholds=thresholds,
                                                    input_residuals=np.zeros((layer_sizes[split_rank-1],)),
                                                    input_order=jnp.full((layer_sizes[split_rank-1],), -1, dtype=int), 
                                                    input_activity=jnp.full((layer_sizes[split_rank-1],), 0, dtype=int),
                                                    layer_activity=jnp.zeros((layer_sizes[split_rank],), dtype=int),
                                                    output_activity=jnp.zeros((layer_sizes[split_rank-1], layer_sizes[split_rank])),
                                                    last_sent_iteration=0,
                                                    input_vector=jnp.zeros((layer_sizes[split_rank-1]), dtype=int),
                                                    output_vector=jnp.zeros((layer_sizes[split_rank]), dtype=int),
                                                    values_history=jnp.zeros((params.history_size, layer_sizes[split_rank])),
                                                    history_index=jnp.array(0, dtype=jnp.int32))
                t = 2
                all_time = 0
                total_batches = (total_train_batches, total_val_batches, total_test_batches)

                # for i in range(t):
                #     _, _, ex_time = batch_predict(params, key, total_batches, weights, empty_neuron_states, "test", save=False, debug=True)
                #     all_time += ex_time
                # print("average execution time : {}", all_time/t)
                # batch_predict(params, key, total_batches, weights, empty_neuron_states, "test", save=True, debug=True)
                result_path = train(params, key, total_batches, weights, empty_neuron_states, "adam", trial)
                if trial is not None:
                    return result_path
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