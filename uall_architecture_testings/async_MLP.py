import os
os.environ["JAX_PLATFORMS"] = "cpu"

from mpi4py import MPI
# os.environ["JAX_TRACEBACK_FILTERING"] = "on"
os.environ.pop("JAX_TRACEBACK_FILTERING", None)

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import optax

import dataclasses
import time
import json
import sys
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import optuna

import mpi4jax
from mpi4jax import send, recv, bcast

from dataset_helpers.mnist_helper import mnist_loader_manual
from dataset_helpers.cifar10_helper import cifar10_loader_manual
from dataset_helpers.shd_helper import torch_SHD_loader
from dataset_helpers.nmnist_helper import torch_nmnist_loader
from dataset_helpers.dvs_helper import torch_DVSGesture_loader
from dataset_helpers.ncars_helper import torch_NCARS_loader
from dataset_helpers.iris_species_helper import torch_iris_loader
from dataset_helpers.network_helper import one_hot_encode

from other_helpers.helpers import BaseParams, NeuronStates
from other_helpers.helpers import accuracy, store_training_data, rerun_init, store_data_to_json
from other_helpers.helpers import activation_func, keep_top_k, output_vector_to_event
from other_helpers.helpers import update_history, process_history, load_config_with_defaults, parse_unknown_args_and_overrides_config
from forward_backward_pass.backpropagation import MLP_back_prop, RNN_back_prop
from forward_backward_pass.loss_functions import loss_bpp, loss_func
from other_helpers.MPI_helpers import MPIConfig, combine_batch_avg, gather_batch, split_batch, l2_weight_regularization
from other_helpers.init_weights import init_params

jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)

TQDM_DISABLE = False
STORE_EACH_EPOCH = False
BUFFER_SIZE = 0
END_SIGNAL = jnp.array([-1.0, -1.0], dtype=jnp.float32)

# Initialize empty global MPI variables
comm = None
rank = None      
size = None

layer_idx = None           # Rank corresponding to the layer
process_per_layer = None    # Number of processes for each layer
last_layer = None            # Rank of last layer
batch_part = None           # The size of the batch on each process
mpi_config = None

training_generator = None
validation_generator = None
test_generator = None

# region PARAMS DEFINITIONS
@dataclasses.dataclass(frozen=True)
class MLPParams(BaseParams):
    """Classification MLP parameters (MNIST, CIFAR10, DVS, etc.)"""
    exploration_rate: float = 0.0
    trace_event_timing: bool = False


@dataclasses.dataclass(frozen=True)
class NeuralDecodingParams(BaseParams):
    """Regression neural decoding parameters (primate reaching, neural_decoding)"""
    dataset_file: str | None = None
    collapse_units: bool = True
    preserve_exact_times: bool = False

# endregion

# region INFERENCE
@partial(jax.jit, static_argnames=['params', 'grad'])
def layer_computation(params, key, neuron_idx, layer_input, weights, neuron_states, iteration=0, grad=False): 
    # jax.debug.print("rank {}, neuron idx {} weight array {}", rank, neuron_idx, weights[neuron_idx].shape)   
    # jax.debug.print("rank {}, neuron idx {} layer input {}", rank, neuron_idx, layer_input)   
    # Compute the new values of the neuron states
    filtered_weights = keep_top_k(weights[neuron_idx], params.top_weights, apply_abs=True)
    # filtered_weights = weights[neuron_idx]

    # jax.debug.print("Original weights {}, filtered wights {}", weights[neuron_idx], filtered_weights)
    invalid_idx = neuron_idx < 0
    activations = jax.lax.cond(
        invalid_idx,
        lambda _: neuron_states.values,
        lambda _: layer_input * filtered_weights + neuron_states.values + neuron_states.bias/params.max_nonzero,
        None
    )
    
    # activations = jnp.tanh(activations) # Shape: (128,)
    #TODO: being able to compute multiple incoming index neurons
    #TODO: store the weight residuals of last layer and neuron state of input layer in sparse matrix representation to reduce space utilization because unused
    
    # jax.lax.cond(neuron_idx == -1,
    #                 lambda _: jax.debug.print("rank {}, iteration: {}, neuron idx: {}", rank, iteration, neuron_idx),
    #                 lambda _: None,
    #                 None)

    if grad:
        new_input_residuals = jax.lax.cond(invalid_idx,
                                lambda _: neuron_states.input_residuals,
                                lambda _: neuron_states.input_residuals.at[neuron_idx].add(layer_input),
                                None
                                )
        new_input_activity = jax.lax.cond(invalid_idx,
                                lambda _: neuron_states.input_activity,
                                lambda _: neuron_states.input_activity.at[neuron_idx].add(1),
                                None
                                )
    else:
        new_input_residuals = neuron_states.input_residuals
        new_input_activity = neuron_states.input_activity

    def last_layer_case(): # No need for additional computation at the output layer
        new_values_history, new_history_index = neuron_states.values_history, neuron_states.history_index

        decayed_activations = neuron_states.values + (activations - neuron_states.values) * (params.output_decay ** iteration)

        if params.history_size > 0:
            new_values_history, new_history_index = update_history(new_values_history, new_history_index, decayed_activations)

        dummy_activations = jnp.zeros((activations.shape[0], 2))
        return jnp.array(0), dummy_activations, neuron_states.replace(  values=decayed_activations,
                                                                        input_residuals=new_input_residuals,
                                                                        input_activity=new_input_activity,
                                                                        values_history=new_values_history,
                                                                        history_index=new_history_index,), key
    
    def hidden_layer_case():
        # APPLY THE SYNC RATE
        sync_fire = (iteration - neuron_states.last_sent_iteration >= neuron_states.sync_rate_vector).astype(jnp.int32)
        sync_fire = jax.lax.cond(invalid_idx, lambda _: jnp.ones(sync_fire.shape, dtype=jnp.int32), lambda _: sync_fire, None)
        activated_output = activations * sync_fire # Mask out the neurons that don't meet the sync rate condition
        # jax.debug.print("rank {}, sync_fire: {}, iteration {}, sync rate vector {}, sync rate {}", rank, sync_fire.shape, iteration.shape, neuron_states.last_sent_iteration.shape,  neuron_states.sync_rate_vector.shape)

        # APPLY ACTIVATION FUNCTION
        activated_output = activation_func(neuron_states.thresholds, activated_output)

        # APPLY THE FIRING NUMBER
        f_nb = params.firing_nb
        k = f_nb if isinstance(f_nb, int) else f_nb[layer_idx]
        pre_topk = activated_output  # Save pre-top-k activations for exploration
        activated_output = keep_top_k(activated_output, k) # Get the top k activations

        # EXPLORATION: randomly replace a top-k fired neuron with a non-top-k non-zero neuron
        # Only applies when firing_nb is actually restricting (k < number of non-zero values)
        new_key = key
        if params.exploration_rate > 0.0:
            new_key, exploration_key = jax.random.split(key)

            num_nonzero = jnp.count_nonzero(pre_topk)
            should_explore = jnp.logical_and(
                jax.random.uniform(key) < params.exploration_rate,
                num_nonzero > k  # Only explore when top-k is actually restricting
            )
            def do_exploration(_):
                key1, key2 = jax.random.split(exploration_key)
                topk_mask = activated_output > 0              # mask of fired neurons
                nonzero_mask = pre_topk > 0                   # mask of all non-zero neurons
                non_topk_nonzero_mask = jnp.logical_and(nonzero_mask, ~topk_mask)  # non-zero but not in top-k

                # Pick a random fired neuron to remove
                topk_indices = jnp.nonzero(topk_mask, size=k, fill_value=-1)[0]
                remove_choice = jax.random.randint(key1, shape=(), minval=0, maxval=k)
                remove_idx = topk_indices[remove_choice]

                # Pick a random non-top-k non-zero neuron to add
                n = activated_output.shape[0]
                candidate_indices = jnp.nonzero(non_topk_nonzero_mask, size=n, fill_value=-1)[0]
                num_candidates = jnp.count_nonzero(non_topk_nonzero_mask)
                add_choice = jax.random.randint(key2, shape=(), minval=0, maxval=jnp.maximum(num_candidates, 1))
                add_idx = candidate_indices[add_choice]

                # Apply the swap: zero out the removed neuron, add the new one
                swapped = activated_output.at[remove_idx].set(0.0)
                swapped = swapped.at[add_idx].set(pre_topk[add_idx])
                return swapped

            activated_output = jax.lax.cond(should_explore, do_exploration, lambda _: activated_output, None)

        # APPLY THE RESTRICTION
        reset = params.restrict
        if not isinstance(reset, int) and not isinstance(reset, float):
            reset = reset[layer_idx]
        penalty = activated_output * reset if reset > 0 else activated_output

        
        active_mask = (activated_output > 0)
        fire = jnp.logical_and(sync_fire, active_mask)  # If sync condition is met and the neuron activated => update the neuron's last sent iteration
        new_last_sent_iteration = jnp.where(fire, iteration, neuron_states.last_sent_iteration)
        if grad:
            active_indexes = active_mask.astype(neuron_states.layer_activity.dtype)  # Update the layer activity by adding the neurons that activated
            last_neuron_idx = jnp.argmax(neuron_states.input_order) # Last neuron index in the input order
            new_neuron_idx = jax.lax.cond(invalid_idx, lambda _: last_neuron_idx, lambda _: neuron_idx, None)

            new_neuron_states = neuron_states.replace(
                values=activations - penalty,
                input_residuals=new_input_residuals,
                input_activity=new_input_activity,
                layer_activity=neuron_states.layer_activity + active_indexes,
                input_order=neuron_states.input_order.at[new_neuron_idx].set(iteration),                    # Update the input activity by setting the input neuron to the iteration number 
                output_activity=neuron_states.output_activity.at[new_neuron_idx].add(active_indexes),
                input_vector=neuron_states.input_vector.at[neuron_idx].set(iteration + 1),                  # Set the input neuron to the iteration at which the input was received (# Added +1 so that we can differentiate between never activated (0) and activated at iteration 0 (1))
                output_vector=jnp.where(active_mask, iteration + 1, neuron_states.output_vector),  # Set the output neuron to the last iteration at which it activated     (Same as above for +1)
                last_sent_iteration=new_last_sent_iteration,)
        else:
            new_neuron_states = neuron_states.replace(
                values=activations - penalty,
                input_residuals=new_input_residuals,
                input_activity=new_input_activity,
                last_sent_iteration=new_last_sent_iteration)

        valid_elements = jnp.count_nonzero(activated_output)
        processed_output = output_vector_to_event(key, activated_output, params, params.layer_sizes[layer_idx])

        return valid_elements, processed_output, new_neuron_states, new_key
    
    
    if layer_idx == last_layer:
        return last_layer_case()
    else:
        return hidden_layer_case()
    
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
    '''
    MLP inference, each layer sends each event separately in the format: (index, value)
    -1 means end of data from previous layer
    -2 means placeholder data in the input layer 
    '''
    def input_layer(x):
        # neuron_states, x = args # x is shape (input_layer_size,)
        x_p = jnp.array(x)

        # 2 Ways to compute and send the inputs (depending on the dataset one or the other is more efficient) 
        # TODO: Determine when one is better than the other
        # @jit
        # def send_input(i, carry):
        #     timestep = carry
        #     data = x_p[i]
        #     @jit
        #     def send_data(t):
        #         # combined = jnp.stack([data[3], data[0], data[1], 1.0]) # Sending format (c, x, y, v)
        #         combined = data

        #         # jax.debug.print("rank {} sending data {}", rank, combined)
        #         send(combined, dest=rank+process_per_layer, tag=0, comm=comm)
        #         return t+1
            
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
            send(data, dest=rank+process_per_layer, tag=0, comm=comm)
            # jax.lax.cond(i < 10,
            #              lambda _: jax.debug.print("rank {} sending data {}", rank, data),
            #              lambda _: None, None)
            # jax.debug.print("rank {} sending data {}", rank, data)
            return i

        mask = (x_p != -2)
        loop_iterations = (jnp.count_nonzero(mask)/2).astype(int)

        iteration = jax.lax.fori_loop(0, loop_iterations, send_input, (0))

        # Send end signal
        send(END_SIGNAL, dest=rank+process_per_layer, tag=0, comm=comm)
        
        buffer = jnp.zeros((BUFFER_SIZE, 2))
        return iteration, buffer, key
    
    def other_layers(neuron_states):
        def cond(state): # end of input has been reached -> break the while loop
            _, _, neuron_idx, _, _, _= state            
            return neuron_idx != -1
        def forward_pass(state):
            layer_input, neuron_states, neuron_idx, iteration, buffer, key = state
            def hidden_layers(loop_iterations, activated_output): # Send activation to the next layers
                def send_activation(i, _):
                    out_val = activated_output[i]
                    send(out_val, dest=rank+process_per_layer, tag=0, comm=comm)
                    return None
                jax.lax.fori_loop(0, loop_iterations, send_activation, None)
                return None
            
            # Receive neuron values from previous layers and compute the activations 
            # (neuron_idx, layer_input) = recv(jnp.zeros((2,)), source=rank-process_per_layer, tag=0, comm=comm)
            (neuron_idx, layer_input) = recv(jnp.zeros((2,)), source=MPI.ANY_SOURCE, tag=0, comm=comm)
            loop_iterations, activated_output, new_neuron_states, new_key = layer_computation(params, key, neuron_idx.astype(int), layer_input, weights, neuron_states, iteration, grad)

            # buffer = jax.lax.cond(
            #     iteration < BUFFER_SIZE,
            #     lambda: buffer.at[iteration].set(jnp.array([(neuron_idx, layer_input)]).flatten()),
            #     lambda: buffer,  # don't update if already have 100
            # )
            neuron_states = new_neuron_states
            
            if layer_idx != last_layer:
                hidden_layers(loop_iterations, activated_output)

            return layer_input, neuron_states, neuron_idx, iteration+1, buffer, new_key
        
        neuron_idx = 0
        layer_input = jnp.zeros(())
        initial_state = (layer_input, neuron_states, neuron_idx, 0, jnp.zeros((BUFFER_SIZE, 2)), key)
        
        # Loop until the rank receives a -1 neuron_idx
        layer_input, neuron_states, neuron_idx, iteration, buffer, new_key = jax.lax.while_loop(cond, forward_pass, initial_state)

        # Send -1 to the next rank when all incoming data has been processed
        if layer_idx != last_layer:
            send(END_SIGNAL, dest=rank + process_per_layer, tag=0, comm=comm)

        return layer_input, neuron_states, iteration-1, buffer, new_key

    # Loop over batches, accumulate output values and return them
    def loop_over_batches(_, x):
        neuron_states = empty_neuron_states
        if layer_idx==0:
            iterations, buffer, new_key = input_layer(x)
            layer_input, new_neuron_states = jnp.zeros(()), neuron_states
        else:
            layer_input, new_neuron_states, iterations, buffer, new_key = other_layers(neuron_states)        
        return None, (new_neuron_states.values, iterations, new_neuron_states, buffer, new_key)
    
    _, (all_outputs, all_iterations, all_neuron_states, buffer, new_key) = jax.lax.scan(loop_over_batches, None, batch_data)
    
    # Synchronize all ranks before starting the backward pass
    mpi4jax.barrier(comm=comm)

    return all_outputs, all_iterations, all_neuron_states, buffer, new_key

#region Training helpers
@partial(jax.jit, static_argnames=['params'])
def predict_bwd(params, key, weights, empty_neuron_states, batch_data):
    '''
    B: batch_size
    '''
    all_outputs, iterations, all_neuron_states, buffer, new_key = (predict)(params, key, weights, empty_neuron_states, batch_data, grad=True)
    # jax.debug.print("rank {}, layer activity: {} max: {}, ending values: {}", rank, all_neuron_states.layer_activity[0], jax.vmap(jnp.max)(all_neuron_states.layer_activity), all_neuron_states.values[0])

    w_sum = l2_weight_regularization(mpi_config, weights)

    # Receive the gradients from the later layers
    next_grad = recv(jnp.zeros((batch_part, params.layer_sizes[layer_idx])), source=rank + process_per_layer, tag=2, comm=comm) # Shape: (B, 128)

    # Compute input's gradient and weight gradient
    weight_grad, th_grad, weight_res, bias_grad = MLP_back_prop(params, all_neuron_states, next_grad, layer_idx)
    weight_grad += 2 * params.w_reg * weights
    # bias_grad = jnp.zeros(empty_neuron_states.bias.shape)

    if layer_idx > 1:
        cur_relu_mask = (all_neuron_states.output_vector > 0).astype(next_grad.dtype)

        # Send gradient to the previous layer
        send_grad = jnp.dot(next_grad * cur_relu_mask, weights.T) # Shape: (B, 128) @ (128, 784) = (B, 784)
        send(send_grad, dest=rank-process_per_layer, tag=2, comm=comm)
    
    # Sparsity loss gradients 
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
    scaling = 0.0
    if params.sparsity_impact[layer_idx] > 0:
        scaling = params.sparsity_impact[layer_idx] / (all_iterations * batch_part * process_per_layer)

    
    input_activity = jnp.sum(all_neuron_states.input_activity, axis=0) # Shape (784)
    layer_activity = jnp.sum(all_neuron_states.layer_activity, axis=0) # Shape (128)
    
    layer_activity = gather_batch(layer_activity, mpi_config, average=False) # Gather the weight gradients from all ranks in the same layer
    input_activity = gather_batch(input_activity, mpi_config, average=False)
    
    sparsity_residuals = scaling * layer_activity # Shape: (128,)
    # jax.debug.print("Rank {}, scaling mean: {}, sparsity_residuals mean: {}, sparsity_residuals sum: {}", rank, scaling, jnp.mean(sparsity_residuals), jnp.sum(sparsity_residuals))
    
    th_sparsity_grad = -sparsity_residuals
    weight_sparsity_grad = jnp.outer(input_activity, sparsity_residuals) # Shape: (784, 128)
    # jax.debug.print("Rank {}, th_sparsity_grad: {}, weight_sparsity_grad: {}", rank, jnp.mean(th_sparsity_grad), np.mean(weight_sparsity_grad))
    
    return all_outputs, iterations, all_neuron_states, (weight_grad, th_grad, weight_sparsity_grad, th_sparsity_grad, bias_grad) 

# Define the loss function
@partial(jax.jit, static_argnames=['params'])
def loss_fn(params, key, weights, empty_neuron_states, target, batch_data):
    all_outputs, iterations, all_neuron_states, buffer, new_key = (predict)(params, key, weights, empty_neuron_states, batch_data, grad=True)
    w_sum = l2_weight_regularization(mpi_config, weights)

    # Compute Loss and loss gradient
    loss, loss_grad = jax.value_and_grad(loss_func)(all_outputs, target)
    loss_grad /= process_per_layer # Shape (B, 10)
    loss += params.w_reg * w_sum

    # Compute output gradient and weight gradient
    out_grad, weight_grad = jax.vmap(loss_bpp, in_axes=(None, 0, 0))(weights, all_neuron_states, loss_grad) # Shape (B, 128), (B, 128, 10)
    # jax.debug.print("Rank {}, loss: {}, loss grad mean: {}, weight grad mean: {}", rank, loss, (loss_grad.shape), (weight_grad.shape))
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
    Compute the sparsity loss based on the input residuals and the weight residuals
    '''
    if all(x <= 0.0 for x in params.sparsity_impact):
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
            all_activations = all_activations + (params.sparsity_impact[i] * act_sum[0]) # Sum of all activations in the hidden layers
            
            if i == 0: # Get iterations of input data
                it_mean = recv(jnp.zeros(1), source=i * process_per_layer, tag=6, comm=comm)
                all_iterations = it_mean[0]
        all_activations += params.sparsity_impact[layer_idx] * jnp.sum(activations) # Adding the activations of the last layer

        sparsity_L = all_activations /  (all_iterations * batch_part * process_per_layer)
        # jax.debug.print("Rank {}, sparsity L: {}, all iterations: {}, all activations: {}", rank, sparsity_L, all_iterations, all_activations)
    all_iterations = bcast(all_iterations, root=last_layer*process_per_layer, comm=comm)

    return all_activations, all_iterations, sparsity_L

def _mlp_extra_fields(params) -> dict:
    """MLP-specific fields written to the result JSON on top of the base fields."""
    return {k: getattr(params, k, None) for k in ("use_tanh", "exact_rtrl", "recurrence")}

# region TRAINING
def train(params: BaseParams, key, total_batches, weights, empty_neuron_states, opti, trial=None, readInputJson=False):     
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
        bias_opt_state = solver.init(empty_neuron_states.bias)

    th_solver = optax.adam(learning_rate=params.threshold_lr)
    th_opt_state = th_solver.init(jax.scipy.special.logit(empty_neuron_states.thresholds))
    
    # Synchronize all ranks and start timer
    mpi4jax.barrier(comm=comm)
    start_time = time.time()

    for epoch in tqdm(range(params.num_epochs), disable=TQDM_DISABLE):
        key, subkey = jax.random.split(key)

        if layer_idx == last_layer:
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
                if readInputJson: # Test with stored input
                    folder_add = "14_sorted_buf"
                    with open(f'pretrained_data/pretrained_data{folder_add}/{len(params.layer_sizes)}hidden_single_input.json') as f:
                        batch_x = np.expand_dims(np.array(json.load(f)).squeeze()[0], axis=0)
                    with open(f'pretrained_data/pretrained_data{folder_add}/{len(params.layer_sizes)}hidden_single_output.json') as f:
                        batch_y = np.expand_dims(np.array(json.load(f)["labels"]).squeeze()[0], axis=0)
                else:
                    batch_x, batch_y = split_batch(params, batch_iterator, mpi_config, 2) # Split the dataset to all the ranks of the input layer
                # print(f"rank {rank} data has shape {(batch_x.shape)}, {(batch_y.shape)}")

                # Send labels to the output layer via plain mpi4py to avoid mpi4jax cache pollution
                comm.Send(np.ascontiguousarray(np.asarray(batch_y, dtype=np.float32)), dest=last_layer * process_per_layer + rank, tag=10)

                # Run the forward pass
                outputs, iterations, all_neuron_states, buffer, new_key = (predict)(params, subkey, weights, neuron_states, batch_data=jnp.array(batch_x))
                all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
            else:
                if layer_idx==last_layer: # Output layer
                    # Receive the labels from the input layer via plain mpi4py
                    y_buf = np.empty((batch_part,), dtype=np.float32)
                    comm.Recv(y_buf, source=rank - (last_layer * process_per_layer), tag=10)
                    y = y_buf
                    y_encoded = jnp.array(one_hot_encode(y, num_classes=params.layer_sizes[-1]))

                    # Run the forward and backward pass for the output layer
                    (loss, outputs, iterations, total_loss, history), gradients = (loss_fn)(params, subkey, weights, neuron_states, y_encoded, jnp.zeros((batch_part, params.layer_sizes[0])))
                    
                    weight_grad = gradients[0]
                    # weight_grad = gather_batch(weight_grad, mpi_config, average=True)
                    weight_grad = combine_batch_avg(weight_grad, mpi_config) # Gather the weight gradients from all ranks in the same layer

                    # Store the accuracy, loss and history                    
                    valid_y, batch_correct = accuracy(i, outputs, y, iterations, False)           
                    # print(f"Batch {i}, Accuracy: {batch_correct}/{valid_y.shape[0]} ")         
                    epoch_correct += int(batch_correct)
                    epoch_total += valid_y.shape[0]

                    epoch_loss.append(float(loss))
                    if params.history_size > 0:
                        all_history.append(history)
                else:
                    # Run the forward and backward pass for the hidden layers
                    outputs, iterations, all_neuron_states, grads = (predict_bwd)(params, subkey, weights, neuron_states, jnp.zeros((batch_part, params.layer_sizes[0])))
                    weight_grad, threshold_grad, weight_sparsity_grad, threshold_sparsity_grad, bias_grad = grads

                    threshold_grad = gather_batch(threshold_grad, mpi_config, average=True) # Gather the thresholds' gradients from all ranks in the same layer
                    
                    # weight_grad = gather_batch(weight_grad, mpi_config, average=True)
                    weight_grad = combine_batch_avg(weight_grad, mpi_config) # Gather the weight gradients from all ranks in the same layer
                    
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
                    else:
                        new_thresholds = empty_neuron_states.thresholds    

                    b = empty_neuron_states.bias
                    if params.use_bias:
                        bias_updates, bias_opt_state = solver.update(bias_grad, bias_opt_state, b)
                        new_bias = optax.apply_updates(b, bias_updates)
                    else:
                        new_bias = b 

                        
                    empty_neuron_states = empty_neuron_states.replace(
                                            bias=new_bias,
                                            thresholds=new_thresholds,)
                # Update weights
                if solver is not None:
                    # Optax optimizer
                    updates, opt_state = solver.update(weight_grad, opt_state, weights)
                    weights = optax.apply_updates(weights, updates)
                else:
                    # Basic GD
                    weights -= params.learning_rate * weight_grad 
            # if i >= 100: # Run a few epochs for testing
            #     break
            #     return 0, 0
            valid_mask = iterations > 0
            epoch_iter_sum += float(jnp.sum(jnp.where(valid_mask, iterations, 0.0)))
            epoch_iter_count += int(jnp.sum(valid_mask))

        # Compute the average iterations for each layer
        mean = epoch_iter_sum / epoch_iter_count if epoch_iter_count > 0 else 0.0
        all_mean_iterations.append(mean)
        all_mean_iterations = gather_batch(jnp.array(all_mean_iterations), mpi_config)
        all_mean_iterations = all_mean_iterations.tolist()

        if layer_idx != 0 and trial is None:
            jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points and a mean threshold of {}", rank, mean, epoch_iter_count, jnp.mean(empty_neuron_states.thresholds))
        
        # Inference on the validation set
        val_accuracy, val_mean, _ = batch_predict(params, key, total_batches, weights, empty_neuron_states, dataset="val", save=False, debug=False)

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
            weights_dict, all_iteration_mean, thresholds_dict = gather_w_it_th(params, weights, jnp.array(all_mean_iterations), empty_neuron_states.thresholds)
            if rank == last_layer * process_per_layer:
                result_path_str = store_training_data(
                            size,
                            params,
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
                            "MLP_temp",
                            all_history,
                            total_batches[0],
                            extra_fields=_mlp_extra_fields(params))
            
        if trial is not None: # If using Optuna Hyper-parameter tuner
            # Return values if the run is not promising and should be pruned  
            all_mean_it = combine_batch_avg(all_mean_iterations, mpi_config) # Gather the weight gradients from all ranks in the same layer
            all_mean_it = mpi4jax.allgather(all_mean_it, comm=comm)

            val_accuracy = bcast(val_accuracy, root=last_layer * process_per_layer, comm=comm)
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
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_it_th(params, weights, jnp.array(all_mean_iterations), empty_neuron_states.thresholds)
    
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
                            total_batches[0],
                            extra_fields=_mlp_extra_fields(params))
        
        encoded = np.frombuffer(result_path_str.encode("utf-8"), dtype=np.uint8)
        if encoded.size > MAX_LEN:
            raise ValueError("result_path too long")
        padded = np.pad(encoded, (0, MAX_LEN - encoded.size), constant_values=0)
        result_path = jnp.array(padded)
    result_path = bcast(result_path, root=last_layer*process_per_layer, comm=comm)
    result_path = bytes(result_path).decode("utf-8").rstrip("\x00")
    mpi4jax.barrier(comm=comm)

    if trial is not None:
        # If using the Optuna Hyper-parameter tuning return the score for ranking the trials 
        leader_rank = last_layer * process_per_layer
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
            fan_out = shape[1]
        else:
            raise ValueError("Unsupported shape for Kaiming init")
        
        std = jnp.sqrt(2/(fan_in))
        # std = jnp.sqrt(2/(fan_in + fan_out))
        # std=1e-2
        print("rank std: ", rank, std)
        weights = random_layer_params(layer_sizes[layer_idx], layer_sizes[layer_idx-1], keys[layer_idx], scale=std)
        # print(f"rank {rank} Weights shape: {weights.shape}")
        return weights
    else:
        weights = jnp.zeros((layer_sizes[-1], layer_sizes[0]))
        return weights


def gather_w_it_th(params, weights, mean_iterations, thresholds):
    # Gather all the weights and iteration values at the last layer to store them
    leader_rank = layer_idx * process_per_layer

    weights_dict = {}
    all_iteration_mean = []
    thresholds_dict = {}
    
    # print(rank, thresholds.shape, mean_iterations)
    if layer_idx != last_layer and rank == leader_rank:
        send(mean_iterations, dest=last_layer * process_per_layer, tag=5,comm=comm)
        if layer_idx != 0:
            send(weights, dest=last_layer * process_per_layer, tag=5,comm=comm)
            send(thresholds, dest=last_layer * process_per_layer, tag=5,comm=comm)

    elif layer_idx == last_layer and rank == leader_rank:
        for i in range(last_layer):
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
        weights_dict[f"layer_{last_layer}"] = weights.tolist()

        print("all iteration mean: rank", rank, all_iteration_mean)

    return weights_dict, all_iteration_mean, thresholds_dict

def compute_runtime_plot(all_runtimes, all_activations):
    global rank
    leader_rank = layer_idx * process_per_layer

    runtimes_dict = {}
    activations_dict = {}
    if layer_idx != last_layer and rank == leader_rank:
        send(jnp.array(all_runtimes), dest=last_layer * process_per_layer, tag=5,comm=comm)
        send(jnp.array(all_activations), dest=last_layer * process_per_layer, tag=5,comm=comm)
    elif layer_idx == last_layer and rank == leader_rank:
        for i in range(last_layer):
            runtimes = recv(jnp.array(all_runtimes), source=i * process_per_layer, tag=5, comm=comm)
            activations = recv(jnp.array(all_activations), source=i * process_per_layer, tag=5, comm=comm)

            runtimes_dict[f"rank_{i}"] = runtimes.tolist()
            activations_dict[f"rank_{i}"] = activations.tolist()
            print(i, runtimes.shape)
    runtimes_dict[f"rank_{last_layer}"] = all_runtimes
    activations_dict[f"rank_{last_layer}"] = all_activations
    
    if rank == last_layer:
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
    
# def batch_predict_time(params, key, total_batches, weights, empty_neuron_states, dataset:str="train", save=True, debug=True):
#     '''
#     Duplicate of batch_predict for getting each layer's individual runtime
#     '''    
#     global training_generator
#     global validation_generator
#     global test_generator    

#     mpi4jax.barrier(comm=comm)
#     start_time = time.time()
    
#     if dataset == "train":
#         total_batches = total_batches[0]
#         if layer_idx == 0:
#             batch_iterator = None
#             if rank == 0:
#                 print(f"Inference on the training set...")
#                 batch_iterator = iter(training_generator)
#     elif dataset == "val":
#         total_batches = total_batches[1]
#         if layer_idx == 0:
#             batch_iterator = None
#             if rank == 0:
#                 print(f"Inference on the validation set...")
#                 batch_iterator = iter(validation_generator)
#     elif dataset == "test":
#         total_batches = total_batches[2]
#         if layer_idx == 0:
#             batch_iterator = None
#             if rank == 0:
#                 print(f"Inference on the test set...")
#                 batch_iterator = iter(test_generator)
#     else:
#         print("INVALID DATASET")
#         return
        
#     if layer_idx == last_layer:
#         epoch_correct = 0
#         epoch_total = 0
#         all_history = []
    
#     all_runtimes = []
#     all_activations = []
#     epoch_iterations = []
#     for i in tqdm(range(total_batches), disable=TQDM_DISABLE):
#         neuron_states = empty_neuron_states
        
#         if layer_idx == 0:                 
#             batch_x, batch_y = split_batch(params, batch_iterator, mpi_config, 2)
#             # print(f"batch {i} has shape {batch_x.shape}, {batch_y.shape}")
        
#         mpi4jax.barrier(comm=comm)
#         start_predict_time = time.time()   
#         if layer_idx == 0:                 
#             outputs, iterations, all_neuron_states, buffer = (predict)(params, key, weights, neuron_states, jnp.array(batch_x))
#             end_predict_time = time.time()

#             # Send label to the last layer
#             send(batch_y, dest=last_layer * process_per_layer + rank, tag=10,comm=comm)
#         else:
#             outputs, iterations, all_neuron_states, buffer = (predict)(params, key, weights, neuron_states, jnp.zeros((batch_part, params.layer_sizes[0]))) 
#             end_predict_time = time.time()
#             # jax.debug.print("Rank {} All neuron states shape: {}, output shape : {}", rank, all_neuron_states.input_residuals.shape, outputs.shape)

#         all_runtimes.append(end_predict_time - start_predict_time)
#         all_activations.append((iterations.item()))
#         # print(f"rank {rank} finished computing in: {end_predict_time - start_predict_time} seconds (start: {start_predict_time}, end: {end_predict_time})")
#         mpi4jax.barrier(comm=comm)
        
#         if layer_idx != 0:
#             if layer_idx == last_layer:
#                 y = recv(jnp.zeros((batch_part,)), source=rank - (last_layer * process_per_layer), tag=10, comm=comm)   
                
#                 valid_y, batch_correct = accuracy(i, outputs, y, iterations, print=False)                 
                
#                 epoch_correct += batch_correct
#                 epoch_total += valid_y.shape[0]

#                 if params.history_size > 0:
#                     # One-hot target → scalar class index
#                     history = process_history(all_neuron_states.values_history, all_neuron_states.history_index, y)
#                     all_history.append(history)

#         epoch_iterations.append(iterations[iterations > 1])
#         # jax.debug.print("Rank {}, iterations: {}", rank, iterations)
#         # if i > 5:
#         #     break
    
#     # print(f"Shape iterations before flattening: {jnp.array(epoch_iterations).shape}")
#     epoch_iterations = jnp.concatenate(epoch_iterations)
#     mean = jnp.mean(epoch_iterations)
#     # print(f"Rank {rank} finished epoch with mean {mean} with {epoch_iterations.shape} iterations")

#     if layer_idx != 0:
#         mean = gather_batch(mean, mpi_config)
#     # jax.debug.print("Rank {}, all iterations shape: {}", rank, (epoch_iterations.shape[0]))
    
#     if rank != 0 and debug:
#         jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, epoch_iterations.shape[0]*process_per_layer)
    
#     epoch_accuracy = -1.0
#     if layer_idx == last_layer:
#         epoch_accuracy = epoch_correct / epoch_total
#         epoch_accuracy = gather_batch(epoch_accuracy, mpi_config)
#         if debug:
#             jax.debug.print("Epoch Accuracy: {:.10f}%", epoch_accuracy * 100)
#             jax.debug.print("----------------------------\n")
    
#     compute_runtime_plot(all_runtimes, all_activations)
#     weights_dict, all_iteration_mean, thresholds_dict = gather_w_it_th(params, weights, mean, empty_neuron_states.thresholds)
#     # jax.debug.print("rank {} all iterations mean: {}, shape: {}", rank, all_iteration_mean, (all_iteration_mean.shape))
    
#     # Synchronize all MPI processes again
#     mpi4jax.barrier(comm=comm)
#     end_time = time.time()

#     if rank == last_layer * process_per_layer:
#         execution_time = end_time - start_time

#         if debug:            
#             print(f"Execution Time: {execution_time:.6f} seconds")
#         if save:
#             accuracies = {"train": [-1], "val": [-1], "test": [-1]}
#             if dataset in accuracies:
#                 accuracies[dataset] = [epoch_accuracy]

#             store_training_data(size,
#                                 params, 
#                                 "inference",
#                                 accuracies["train"], 
#                                 accuracies["val"], 
#                                 accuracies["test"][0],
#                                 execution_time,
#                                 all_iteration_mean,
#                                 weights_dict,
#                                 [],
#                                 thresholds_dict,
#                                 "",
#                                 "MLP",
#                                 all_history,
#                                 total_batches)
#     return epoch_accuracy, mean, end_time - start_time
            
# region Inference
#TODO: training and inference functions are very similar, we could merge them and avoid code duplication
def batch_predict(params: BaseParams, key, total_batches, weights, empty_neuron_states: NeuronStates, dataset:str="train", save=True, debug=True, readInputJson=False):
    '''
    This function implements the forward pass of the neural network

    :param params: params object holding all the network's parameters
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
    if total_batches == 0:
        return -0.01, -1.0, -1.0 # arbitrary code for empty dataset
    
    if layer_idx == last_layer:
        epoch_correct = 0
        epoch_total = 0
        all_history = []
    
    epoch_iter_sum = 0.0
    epoch_iter_count = 0
    for i in tqdm(range(total_batches), disable=TQDM_DISABLE):
        neuron_states = empty_neuron_states
        
        if layer_idx == 0: # Input layer  
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
            outputs, iterations, all_neuron_states, buffer, new_key = (predict)(params, key, weights, neuron_states, jnp.array(batch_x))

            # Send label to the last layer via plain mpi4py
            comm.Send(np.ascontiguousarray(np.asarray(batch_y, dtype=np.float32)), dest=last_layer * process_per_layer + rank, tag=10)
        else:
            # Run forward pass for hidden and output layers
            outputs, iterations, all_neuron_states, buffer, new_key = (predict)(params, key, weights, neuron_states, jnp.zeros((batch_part, params.layer_sizes[0]))) 
        
            if layer_idx == last_layer: # Output layer
                # Receive the labels from the input layer via plain mpi4py
                y_buf = np.empty((batch_part,), dtype=np.float32)
                comm.Recv(y_buf, source=rank - (last_layer * process_per_layer), tag=10)
                y = y_buf
                
                valid_y, batch_correct = accuracy(i, outputs, y, iterations, print=False)

                epoch_correct += int(batch_correct)
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

        valid_mask = iterations > 1
        epoch_iter_sum += float(jnp.sum(jnp.where(valid_mask, iterations, 0.0)))
        epoch_iter_count += int(jnp.sum(valid_mask))
        # if i >= 100: # Run a single epoch for testing
        #     break

    # Compute the average iterations for each layer
    mean = epoch_iter_sum / epoch_iter_count if epoch_iter_count > 0 else 0.0
    mean = gather_batch(jnp.array(mean), mpi_config)

    if rank != 0 and debug:
        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, epoch_iter_count*process_per_layer)
    
    epoch_accuracy = -1.0
    if layer_idx == last_layer: # Output layer
        print(f"epoch correct {epoch_correct}, epoch total: {epoch_total}")
        epoch_accuracy = epoch_correct / epoch_total
        epoch_accuracy = gather_batch(epoch_accuracy, mpi_config)
        if debug:
            jax.debug.print("Epoch Accuracy: {:.10f}%", epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")
    
    # Gather the weights and iteration values at the last layer
    weights_dict, all_iteration_mean, thresholds_dict = gather_w_it_th(params, weights, mean, empty_neuron_states.thresholds)
    
    # Synchronize all MPI processes again
    mpi4jax.barrier(comm=comm)
    end_time = time.time()

    # Compute processing time and store all the results in a json file if save is True
    if rank == last_layer * process_per_layer:
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
def get_layer_idx(batch_size, layer_sizes, trial=None):
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
    if trial is None:
        print(f"Rank {rank}, layer idx: {layer_idx}, batch part: {batch_part}, process per layer: {process_per_layer}, last rank: {last_layer}")

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
    restrict = config['restrict']
    init_thresholds = config['init_thresholds']
    load_file = config['load_file']
    best = config['best']
    rerun = config['rerun']
    mode = config.get('mode', 'training')


    if trial is not None: # Optuna Hyper-parameter tuning parameter
        dataset = trial_params.dataset
        layer_sizes = trial_params.layer_sizes
        batch_size = trial_params.batch_size
        restrict = trial_params.restrict
        init_thresholds = trial_params.init_thresholds

    if size % len(layer_sizes) != 0:
        print(f"Error: layer_sizes ({len(layer_sizes)}) must match number of MPI ranks ({size})")
        sys.exit(1)
    
    get_layer_idx(batch_size, layer_sizes, trial) # Compute the layer index for training/inference with multiple processes per batch

    if batch_size % process_per_layer != 0:
        print(f"Error: one batch ({batch_size}) must be divisible by the number of processes per layer ({process_per_layer})")
        sys.exit(1)
    
    # for reset in [0.1, 0.3, 0.5, 0.7, 0.9]: # Loop for multiple experiments
    # for lr in [0.00001, 0.00005, 0.0001, 0.0005, 0.001]: # Loop for multiple experiments
    for f_nb in [1]: # Loop for multiple experiments
        # Initialize parameters (input data for rank 0 and weights for other ranks)
        key, subkey = jax.random.split(key) 
        total_train_batches, total_val_batches, total_test_batches, max_nonzero = 0, 0, 0, 0
        weights = init_params(subkey, batch_size, layer_sizes, load_file=load_file, best=best)
        if rank == 0: # Only the first rank loads the dataset
            downsample = False
            # Load the data 
            match dataset:
                case "mnist" | "smnist" | "psmnist":
                    sequential = dataset in ("smnist", "psmnist")
                    permuted = dataset == "psmnist"
                    if layer_sizes[0] == 14*14:
                        downsample = True
                    loader = partial(mnist_loader_manual, 
                                        sequential=sequential, 
                                        permuted=permuted)
                case "shd":
                    loader = torch_SHD_loader
                case "nmnist":
                    loader = torch_nmnist_loader
                case "dvs":
                    if layer_sizes[0] == 64*64*2:
                        downsample = True 
                    loader = torch_DVSGesture_loader                            
                case "ncars":
                    if layer_sizes[0] == 60 * 50 * 2:
                        downsample = True
                    loader = torch_NCARS_loader
                case "cifar10":
                    loader = cifar10_loader_manual
                case _:
                    raise ValueError(f"Unknown dataset: {dataset}")

            # Load and unpack the dataloaders
            train_data, val_data, test_data, max_nonzero = loader(  batch_size=batch_size, 
                                                                    shuffle=False, 
                                                                    CNN_preprocess=False, 
                                                                    downsample=downsample,
                                                                    data_dir=data_dir)
            training_generator, total_train_batches = train_data
            validation_generator, total_val_batches = val_data
            test_generator, total_test_batches = test_data
            print("max nonzero: ", max_nonzero)

        # Broadcast the total number of batches to all other ranks
        total_train_batches, total_val_batches, total_test_batches = bcast(jnp.array([total_train_batches, total_val_batches, total_test_batches]), root=0 , comm=comm)
        max_nonzero = bcast(jnp.array([max_nonzero]), root=0, comm=comm)
        max_nonzero = max_nonzero.tolist()[0]

        thresholds = jnp.full(layer_sizes[layer_idx], init_thresholds)

        params = MLPParams(
            dataset=dataset,
            random_seed=random_seed,
            layer_sizes=layer_sizes, 
            init_thresholds=init_thresholds, 
            num_epochs=config['num_epochs'],
            # learning_rate=lr, 
            learning_rate=config['learning_rate'], 
            batch_size=batch_size,
            load_file=load_file,
            shuffle_activations=config['shuffle_activations'],
            # restrict=reset,
            restrict=config['restrict'],
            # firing_nb=(1, f_nb,1,1,1,1,1,1,1),
            firing_nb=config['firing_nb'],
            sync_rate=config['sync_rate'],
            max_nonzero=max_nonzero,
            shuffle_input=config['shuffle_input'],
            threshold_lr=config['threshold_lr'],
            sparsity_impact=tuple(config['sparsity_impact']),
            w_reg=config['w_reg'],
            rerun=None,
            top_weights=config['top_weights'],
            history_size=config['history_size'],
            output_decay=config.get('output_decay', 1.0),
            use_bias=config['use_bias'],
            exploration_rate=config['exploration_rate'],
            trace_event_timing=config['trace_event_timing'],
        )
        if trial is not None:
            params = dataclasses.replace(trial_params, max_nonzero=max_nonzero)

        if rerun is not None:
            override_list = config.get('override_params', None)
            params, weights, thresholds = rerun_init(
                rerun,
                mpi_config,
                params,
                override_params=override_list
            )

        if rank == 0:
            print(f"Number of training batches: {total_train_batches}, validation batches: {total_val_batches}, test batches: {total_test_batches}")
            print(params)
        
        # Instantiate the neuron states with the correct shapes and initial values
        prev_size, cur_size = layer_sizes[layer_idx-1], layer_sizes[layer_idx] 

        # layer_key = jax.random.fold_in(key, layer_idx)
        # sync_rate_vector = jax.random.randint(layer_key, shape=(layer_sizes[layer_idx],), minval=1, maxval=params.sync_rate)
        sync_rate_vector = jnp.full(shape=(cur_size,), fill_value=params.sync_rate)

        empty_neuron_states = NeuronStates( 
            values=jnp.zeros(cur_size),
            bias=jnp.zeros(cur_size),
            thresholds=thresholds,
            input_residuals=np.zeros((prev_size,)),
            input_order=jnp.full((prev_size,), -1, dtype=int),
            input_activity=jnp.full((prev_size,), 0, dtype=int),
            layer_activity=jnp.zeros((cur_size,), dtype=int),
            output_activity=jnp.zeros((prev_size, cur_size)),
            last_sent_iteration=jnp.full(shape=(cur_size,), fill_value=-1),
            input_vector=jnp.zeros((prev_size), dtype=int),
            output_vector=jnp.zeros((cur_size), dtype=int),
            sync_rate_vector=sync_rate_vector,
            values_history=jnp.zeros((params.history_size, cur_size)),
            history_index=jnp.array(0, dtype=jnp.int32))
        # print(f"rank {rank} sync rates: {sync_rate_vector}")
        total_batches = (total_train_batches, total_val_batches, total_test_batches)

        if mode == 'inference':
            # To only run inference
            batch_predict(params, key, total_batches, weights, empty_neuron_states, "test", save=True, debug=True)
        elif mode == 'training':
            # To run the full training pipeline
            val_acc, result_path = train(params, key, total_batches, weights, empty_neuron_states, config['optimizer'], trial)
            # val_acc, result_path = train(params, key, total_batches, weights, empty_neuron_states, "adam", trial)
        else:
            print(f"Unknown mode in config file, choose either 'training' or 'inference', got {mode}")
            sys.exit(1)

if __name__ == "__main__":
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
JAX_PLATFORMS=cpu mpirun -n 4 python async_MLP.py --config "configs/MLP_config.yaml"
'''