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
from other_helpers.helpers import process_history, load_config_with_defaults, parse_unknown_args_and_overrides_config
from forward_backward_pass.backpropagation import MLP_back_prop
from forward_backward_pass.loss_functions import loss_bpp, loss_func

from other_helpers.general_MPI_helper_input_split import data_split, model_split_custom, model_split
from other_helpers.init_weights import init_params
from forward_backward_pass.inference_input_split import predict, layer_computation

jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)

TQDM_DISABLE = False
STORE_EACH_EPOCH = False
BUFFER_SIZE = 0
END_SIGNAL = jnp.array([-1.0, -1.0], dtype=jnp.float32)

# Hidden→hidden backward gradient. DEFAULT: old unmasked dot(next_grad, W.T) (trains deep
# nets much faster). Set AINED_LEGACY_BWD=0 to restore the (output_vector>0) ReLU-derivative mask.
# (This script already uses a constant LR / plain Adam.)
_LEGACY_BWD_GRAD = os.environ.get("AINED_LEGACY_BWD", "1") == "1"

# Initialize empty global MPI variables
comm = None
rank = None      
size = None

layer_idx = None           # Rank corresponding to the layer
processes_per_layer_global = None    # Number of processes for each layer
input_splits_per_layer_global = None # Input splits for each layer (2D block parallelism)
result_extra_fields_global = None    # Extra fields persisted into result JSONs (set in main)
last_layer = None            # Rank of last layer
batch_part_size = None           # The size of the batch on each process
mpi_config = None

training_generator = None
validation_generator = None
test_generator = None

@dataclasses.dataclass(frozen=True)
class Params(BaseParams):
    pass

#region Training helpers
@partial(jax.jit, static_argnames=['params'])
def predict_bwd(params, key, weights, empty_neuron_states, batch_data):
    '''
    B: batch_size
    '''
    all_outputs, iterations, all_neuron_states, buffer = (predict)(params, 
                                                                   mpi_config, 
                                                                   key, 
                                                                   weights, 
                                                                   empty_neuron_states, 
                                                                   layer_computation,
                                                                   batch_data, 
                                                                   grad=True, 
                                                                   END_SIGNAL=END_SIGNAL, 
                                                                   BUFFER_SIZE=BUFFER_SIZE)
    # w_sum = l2_weight_regularization(mpi_config, weights)

    next_grad = mpi_config.backward_recv()   # Shape: (B, layer_size)
    # jax.debug.print("Rank {} received next_grad shape: {}, next grad mean {}", rank, next_grad.shape, jnp.mean(next_grad))
    weight_grad, th_grad, weight_res, bias_grad = MLP_back_prop(params, all_neuron_states, next_grad, layer_idx)
    weight_grad += 2 * params.w_reg * weights

    if layer_idx > 1:
        if _LEGACY_BWD_GRAD:
            # Old form: unmasked gradient propagation (no ReLU-derivative gate on next_grad).
            send_grad = jnp.dot(next_grad, weights.T)
        else:
            cur_relu_mask = (all_neuron_states.output_vector > 0).astype(next_grad.dtype)
            # Send gradient to the previous layer
            send_grad = jnp.dot(next_grad * cur_relu_mask, weights.T) # Shape: (B, 128) @ (128, 784) = (B, 784)
        mpi_config.backward_send(send_grad)
    
    # Sparsity loss gradients 
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)

    scaling = 0.0
    if params.sparsity_impact[layer_idx] > 0:
        scaling = params.sparsity_impact[layer_idx] / (all_iterations * batch_part_size * mpi_config.get_process_per_batch)

    
    input_activity = jnp.sum(all_neuron_states.input_activity, axis=0) # Shape (784)
    layer_activity = jnp.sum(all_neuron_states.layer_activity, axis=0) # Shape (128)
    
    layer_activity = mpi_config.gather_batch(layer_activity, average=False) # Gather the weight gradients from all ranks in the split rank
    input_activity = mpi_config.gather_batch(input_activity, average=False)
    
    sparsity_residuals = scaling * layer_activity # Shape: (128,)
    # jax.debug.print("Rank {}, scaling mean: {}, sparsity_residuals mean: {}, sparsity_residuals sum: {}", rank, scaling, jnp.mean(sparsity_residuals), jnp.sum(sparsity_residuals))
    
    th_sparsity_grad = -sparsity_residuals
    weight_sparsity_grad = jnp.outer(input_activity, sparsity_residuals) # Shape: (784, 128)
    # jax.debug.print("Rank {}, th_sparsity_grad: {}, weight_sparsity_grad: {}", rank, jnp.mean(th_sparsity_grad), np.mean(weight_sparsity_grad))
    
    return all_outputs, iterations, all_neuron_states, (weight_grad, th_grad, weight_sparsity_grad, th_sparsity_grad) 

# Define the loss function
@partial(jax.jit, static_argnames=['params'])
def loss_fn(params, key, weights, empty_neuron_states, target, batch_data):
    all_outputs, iterations, all_neuron_states, buffer = (predict)(params, 
                                                                   mpi_config, 
                                                                   key, 
                                                                   weights, 
                                                                   empty_neuron_states, 
                                                                   layer_computation,
                                                                   batch_data, 
                                                                   grad=True,
                                                                   END_SIGNAL=END_SIGNAL, 
                                                                   BUFFER_SIZE = BUFFER_SIZE)
    # w_sum = l2_weight_regularization(mpi_config, weights)

    full_outputs = mpi_config.gather_model_partition(all_outputs)

    # jax.debug.print("{} ", full_outputs)
    # Compute Loss and loss gradient
    loss, loss_grad = jax.value_and_grad(loss_func)(full_outputs, target)
    loss_grad /= mpi_config.get_process_per_batch # Shape (B, 10)
    loss_grad = loss_grad[:, mpi_config.model_part.start_idx:mpi_config.model_part.end_idx+1] # Shape (B, 5)
    # loss += params.w_reg * w_sum

    # jax.debug.print("Rank {}, loss grad shape: {}, weights shape {}, mpi_config {}", rank, loss_grad.shape, weights.shape, mpi_config.model_part)

    # Compute output gradient and weight gradient
    out_grad, weight_grad = jax.vmap(loss_bpp, in_axes=(None, 0, 0))(weights, all_neuron_states, loss_grad) # Shape (B, 128), (B, 128, 10)
    mean_weight_grad = jnp.mean(weight_grad, axis=0) # Shape: (128, 10)
    mean_weight_grad += 2 * params.w_reg * weights
    mean_weight_grad = jnp.expand_dims(mean_weight_grad, axis=0)  # Shape: (1, 128, 10)
    # Send gradient to previous layers                
    mpi_config.backward_send(out_grad)

    # jax.debug.print("Rank {}, out grad shape {}", rank, out_grad)    
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
    leader_rank = layer_idx * processes_per_layer_global
    # jax.debug.print("Rank {}, activations shape: {}, iterations shape: {}, last layer neuron values: {}", rank, (activations.shape), jnp.mean(iterations.shape), jnp.sum(all_neuron_states.values))

    activations = mpi_config.gather_batch(all_neuron_states.input_residuals, average=False) # Gather the weight gradients from all ranks in the split rank
    iterations = mpi_config.gather_batch(iterations, average=True) # Gather the iterations from all ranks in the split rank
    # jax.debug.print("Rank {}, activations shape: {}, iterations shape: {}, last layer neuron values: {}", rank, (activations.shape), jnp.mean(iterations.shape), jnp.sum(all_neuron_states.values))

    all_iterations = 0.0
    all_activations = 0.0
    sparsity_L = 0.0
    if layer_idx != last_layer and rank == leader_rank:
        # jax.debug.print("Rank {}, sending activations {} and iterations {} to the last rank", rank, jnp.sum(activations), jnp.mean(iterations))
        send(jnp.sum(activations), dest=last_layer * processes_per_layer_global, tag=6,comm=comm)
        if rank == 0:
            send(jnp.mean(iterations), dest=last_layer * processes_per_layer_global, tag=6,comm=comm)
    elif layer_idx == last_layer and rank == leader_rank:
        for i in range(last_layer):
            # Storing the thresholds
            act_sum = recv(jnp.zeros(1), source=i * processes_per_layer_global, tag=6, comm=comm)
            all_activations = all_activations + (params.sparsity_impact[i] * act_sum[0]) # Sum of all activations in the hidden layers
            
            if i == 0: # Get iterations of input data
                it_mean = recv(jnp.zeros(1), source=i * processes_per_layer_global, tag=6, comm=comm)
                all_iterations = it_mean[0]
        all_activations += params.sparsity_impact[layer_idx] * jnp.sum(activations) # Adding the activations of the last layer

        sparsity_L = all_activations /  (all_iterations * batch_part_size * processes_per_layer_global)
        # jax.debug.print("Rank {}, sparsity L: {}, all iterations: {}, all activations: {}", rank, sparsity_L, all_iterations, all_activations)
    all_iterations = bcast(all_iterations, root=last_layer*processes_per_layer_global, comm=comm)

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
    if params.init_thresholds != 0:
        th_opt_state = th_solver.init(jax.scipy.special.logit(empty_neuron_states.thresholds))
    else:
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

        epoch_iter_sum = 0.0
        epoch_iter_count = 0
        if layer_idx == 0:
            batch_iterator = None
            if rank == 0:
                batch_iterator = iter(training_generator) # Make the dataloader iterable

        for i in tqdm(range(total_batches[0]), miniters=total_batches[0]//10, maxinterval=float('inf'), disable=TQDM_DISABLE):
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
                    batch_x, batch_y = mpi_config.split_batch(params, batch_iterator, 2) # Split the dataset to all the ranks of the input layer
                # print(f"rank {rank} data has shape {(batch_x.shape)}, {(batch_y.shape)}")

                mpi_config.send_labels(batch_y, mpi_config.batch_first_and_last_rank[0]) # Send to the labels to the output layer
                outputs, iterations, all_neuron_states, buffer = (predict)(params, mpi_config, subkey, weights, neuron_states, layer_computation, batch_data=jnp.array(batch_x), 
                                                                           END_SIGNAL=END_SIGNAL, BUFFER_SIZE=BUFFER_SIZE)
                all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
            else:
                if mpi_config.is_last_layer: 
                    # Receive the labels from the input layer
                    y = mpi_config.recv_labels()
                    y_encoded = jnp.array(one_hot_encode(y, num_classes=params.layer_sizes[-1]))

                    # Run the forward and backward pass for the output layer
                    (loss, outputs, iterations, total_loss, history), gradients = (loss_fn)(params, subkey, weights, neuron_states, y_encoded, jnp.zeros((batch_part_size, params.layer_sizes[0])))
                    # jax.debug.print("Rank {}, with {}", rank, outputs)

                    weight_grad = gradients[0]
                    # weight_grad = gather_batch(weight_grad, mpi_config, average=True)
                    # jax.debug.print("last layer before combine batch avg, weight grad shape {}", weight_grad.shape)
                    weight_grad = mpi_config.combine_batch_avg(weight_grad) # Gather the weight gradients from all ranks in the split rank
                    # jax.debug.print("last layer after combine batch avg, weight grad shape {}", weight_grad.shape)
                
                    # Store the accuracy, loss and history                    
                    valid_y, batch_correct = accuracy(i, outputs, y, iterations, False)

                    epoch_correct += int(batch_correct)
                    epoch_total += valid_y.shape[0]

                    epoch_loss.append(float(loss))
                    if params.history_size > 0:
                        all_history.append(history)
                else:
                    # Run the forward and backward pass for the hidden layers
                    outputs, iterations, all_neuron_states, grads = (predict_bwd)(params, subkey, weights, neuron_states, jnp.zeros((batch_part_size, params.layer_sizes[0])))
                    weight_grad, threshold_grad, weight_sparsity_grad, threshold_sparsity_grad = grads

                    # jax.debug.print("hidden layer before gather batch, threshold grad shape {}", threshold_grad.shape)
                    threshold_grad = mpi_config.gather_batch(threshold_grad, average=True) # Gather the weight gradients from all ranks in the split rank
                    # jax.debug.print("hidden layer after gather batch, threshold grad shape {}", threshold_grad.shape)

                    # weight_grad = mpi_config.gather_batch(weight_grad, average=True)
                    weight_grad = mpi_config.combine_batch_avg(weight_grad) # Gather the weight gradients from all ranks in the split rank
                    
                    # Add sparsity loss' impact to the gradient if relevant
                    if jnp.any(jnp.array(params.sparsity_impact) > 0):
                        weight_grad = weight_grad + weight_sparsity_grad
                        threshold_grad = threshold_grad + threshold_sparsity_grad

                    # Update thresholds
                    if params.threshold_lr != 0:
                        th_updates, th_opt_state = th_solver.update(threshold_grad, th_opt_state, empty_neuron_states.thresholds)
                        if params.init_thresholds != 0:
                            new_thresholds = jax.nn.sigmoid(optax.apply_updates(
                                                jax.scipy.special.logit(empty_neuron_states.thresholds), th_updates))
                        else:
                            new_thresholds = optax.apply_updates(empty_neuron_states.thresholds, th_updates)
                        empty_neuron_states = empty_neuron_states.replace(thresholds=new_thresholds)
                # Update weights
                if solver is not None:
                    # Optax optimizer
                    updates, opt_state = solver.update(weight_grad, opt_state, weights)
                    weights = optax.apply_updates(weights, updates)
            # if i >= 0: # Run a single epoch for testing
            #     return
            valid_mask = iterations > 0
            epoch_iter_sum += float(jnp.sum(jnp.where(valid_mask, iterations, 0.0)))
            epoch_iter_count += int(jnp.sum(valid_mask))

        # Compute the average iterations for each layer
        mean = epoch_iter_sum / epoch_iter_count if epoch_iter_count > 0 else 0.0
        all_mean_iterations.append(mean)
        all_mean_iterations = mpi_config.gather_batch(jnp.array(all_mean_iterations))
        all_mean_iterations = all_mean_iterations.tolist()

        if layer_idx != 0 and trial is None:
            jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points and a mean threshold of {}", rank, mean, epoch_iter_count, jnp.mean(empty_neuron_states.thresholds))
        
        # Inference on the validation set
        val_accuracy, val_mean, _ = batch_predict(params, key, total_batches, weights, empty_neuron_states, dataset="val", save=False, debug=False)

        epoch_accuracy = 0.0
        if mpi_config.is_last_layer:
            # Store loss values
            mean_loss = jnp.mean(jnp.array(epoch_loss))
            all_loss.append(mean_loss)
            mean_loss = mpi_config.gather_batch(mean_loss)

            # Store training and validation accuracies
            epoch_accuracy = epoch_correct / epoch_total
            all_epoch_accuracies.append(epoch_accuracy)
            all_validation_accuracies.append(val_accuracy)
            all_epoch_accuracies = mpi_config.gather_batch(all_epoch_accuracies)
            all_validation_accuracies = mpi_config.gather_batch(all_validation_accuracies)
            all_epoch_accuracies, all_validation_accuracies = all_epoch_accuracies.tolist(), all_validation_accuracies.tolist()
            if mpi_config.get_last_layer_batch_leader:
                jax.debug.print("Epoch {} , Training Accuracy: {:.2f}%, Validation Accuracy: {:.2f}%, mean loss: {}, mean val iterations: {}", epoch, all_epoch_accuracies[-1] * 100, val_accuracy * 100, mean_loss, val_mean)
                jax.debug.print("----------------------------\n")
        epoch_accuracy = bcast(epoch_accuracy, root=mpi_config.get_last_layer_batch_leader, comm=comm)
        if epoch_accuracy >= 0.9999:
            break

        if STORE_EACH_EPOCH:
            # Gather the weights and iteration values at the last layer
            weights_dict, all_iteration_mean, thresholds_dict = mpi_config.gather_w_it_th(params, weights, jnp.array(all_mean_iterations), empty_neuron_states.thresholds)
            if mpi_config.is_last_layer_leader:
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
                            extra_fields=result_extra_fields_global)

        if trial is not None: # If using Optuna Hyper-parameter tuner
            # Return values if the run is not promising and should be pruned  
            all_mean_it = mpi_config.combine_batch_avg(all_mean_iterations) # Gather the weight gradients from all ranks in the split rank
            all_mean_it = mpi4jax.allgather(all_mean_it, comm=comm)

            val_accuracy = bcast(val_accuracy, root=mpi_config.get_last_layer_batch_leader, comm=comm)
            # jax.debug.print("all mean it: {} {}", all_mean_it, jnp.max(all_mean_it[processes_per_layer_global*2:])/all_mean_it[0])
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
    weights_dict, all_iteration_mean, thresholds_dict = mpi_config.gather_w_it_th(params, weights, jnp.array(all_mean_iterations), empty_neuron_states.thresholds)
    
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
                            total_batches[0],
                            extra_fields=result_extra_fields_global)

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

    if total_batches == 0:
        return -0.01, -1.0, -1.0 # arbitrary code for empty dataset
    
    if layer_idx == last_layer:
        epoch_correct = 0
        epoch_total = 0
        all_history = []
    
    epoch_iter_sum = 0.0
    epoch_iter_count = 0
    for i in tqdm(range(total_batches), miniters=int(total_batches)//10, maxinterval=float('inf'), disable=TQDM_DISABLE):
        neuron_states = empty_neuron_states
        
        if layer_idx == 0:                 
            if readInputJson: # Test with stored input
                folder_add = "14_sorted"
                with open(f'pretrained_data/pretrained_data{folder_add}/{len(params.layer_sizes)}hidden_single_input.json') as f:
                    batch_x = np.array(json.load(f)).squeeze() 
                with open(f'pretrained_data/pretrained_data{folder_add}/{len(params.layer_sizes)}hidden_single_output.json') as f:
                    batch_y = np.array(json.load(f)["labels"]).squeeze()
            else:
                batch_x, batch_y = mpi_config.split_batch(params, batch_iterator, 2)
            # store_data_to_json(f"{len(params.layer_sizes)}hidden_single_input.json", batch_x.tolist()) # Store for hardware usage

            # Run the forward pass
            outputs, iterations, all_neuron_states, buffer = (predict)(params, mpi_config, key, weights, neuron_states, layer_computation, jnp.array(batch_x), 
                                                                       END_SIGNAL=END_SIGNAL, BUFFER_SIZE=BUFFER_SIZE)

            # Send label to the last layer
            mpi_config.send_labels(batch_y, mpi_config.batch_first_and_last_rank[0])
            # send(batch_y, dest=mpi_config.batch_first_and_last_rank[1], tag=10,comm=comm)
        else:
            # Run forward pass for hidden and output layers
            outputs, iterations, all_neuron_states, buffer = (predict)(params, mpi_config, key, weights, neuron_states, layer_computation, jnp.zeros((batch_part_size, params.layer_sizes[0])), 
                                                                       END_SIGNAL=END_SIGNAL, BUFFER_SIZE=BUFFER_SIZE) 
        
            if layer_idx == last_layer:
                # Receive the labels from the input layer and compute the accuracy
                y = mpi_config.recv_labels()
                # y = recv(jnp.zeros((batch_part_size,)), source=mpi_config.batch_first_and_last_rank[0], tag=10, comm=comm)   

                full_outputs = mpi_config.gather_model_partition(outputs)

                valid_y, batch_correct = accuracy(i, full_outputs, y, iterations, rank=rank, print=False)

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
        # if i >= 0: # Run a single epoch for testing
        #     break

    # Compute the average iterations for each layer
    mean = epoch_iter_sum / epoch_iter_count if epoch_iter_count > 0 else 0.0
    mean = mpi_config.gather_batch(jnp.array(mean))

    if rank != 0 and debug:
        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, epoch_iter_count*mpi_config.get_process_per_batch)
    
    epoch_accuracy = -1.0
    if mpi_config.is_last_layer:
        print(f"epoch correct {epoch_correct}, epoch total: {epoch_total}")
        epoch_accuracy = epoch_correct / epoch_total
        epoch_accuracy = mpi_config.gather_batch(epoch_accuracy)
        if debug and mpi_config.is_last_layer_leader:
            jax.debug.print("Epoch Accuracy: {:.10f}%", epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")
    
    # Gather the weights and iteration values at the last layer
    weights_dict, all_iteration_mean, thresholds_dict = mpi_config.gather_w_it_th(params, weights, mean, empty_neuron_states.thresholds)
    
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
                                total_batches,
                                extra_fields=result_extra_fields_global)
    return epoch_accuracy, mean, end_time - start_time

# region Main
def get_layer_idx(batch_size, layer_sizes, processes_per_layer=None, input_splits_per_layer=None, trial=None):
    '''
    Define each MPI rank's split_rank.
    If processes_per_layer is given (tuple with one int per layer), uses custom data split.
    Otherwise falls back to uniform data split (requires size % nb_layers == 0).
    '''
    global layer_idx
    global last_layer
    global batch_part_size
    global mpi_config
    global processes_per_layer_global
    global input_splits_per_layer_global

    input_splits_per_layer_global = list(input_splits_per_layer) if input_splits_per_layer is not None else None
    if processes_per_layer is not None:
        isl = tuple(input_splits_per_layer) if input_splits_per_layer is not None else None
        mpi_config = model_split_custom(rank, comm, size, batch_size, layer_sizes, tuple(processes_per_layer), isl)
        processes_per_layer_global = list(processes_per_layer)
    else:
        mpi_config = model_split(rank, comm, size, batch_size, layer_sizes)
        processes_per_layer_global = None

    layer_idx = mpi_config.layer_idx
    last_layer = mpi_config.last_layer_idx
    batch_part_size = mpi_config.batch_part.get_size

    mpi_config.print()

def main(random_seed, key, rank_, size_, comm_, trial=None, trial_params=None, config_path=None, data_dir=""):
    global training_generator
    global validation_generator
    global test_generator
    global rank
    global size
    global comm
    global TQDM_DISABLE
    global result_extra_fields_global

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
    processes_per_layer = config.get('processes_per_layer', None)
    if processes_per_layer is not None:
        processes_per_layer = tuple(processes_per_layer)
    input_splits_per_layer = config.get('input_splits_per_layer', None)
    if input_splits_per_layer is not None:
        input_splits_per_layer = tuple(input_splits_per_layer)

    if trial is not None:
        dataset = trial_params.dataset
        layer_sizes = trial_params.layer_sizes
        batch_size = trial_params.batch_size
        restrict = trial_params.restrict
        init_thresholds = trial_params.init_thresholds

    if processes_per_layer is not None:
        if len(processes_per_layer) != len(layer_sizes):
            print(f"Error: processes_per_layer length ({len(processes_per_layer)}) must match number of layers ({len(layer_sizes)})")
            sys.exit(1)
        if input_splits_per_layer is None:
            if sum(processes_per_layer) != size:
                print(f"Error: sum of processes_per_layer ({sum(processes_per_layer)}) must equal MPI size ({size})")
                sys.exit(1)
    else:
        if size % len(layer_sizes) != 0:
            print(f"Error: MPI size ({size}) must be a multiple of number of layers ({len(layer_sizes)}). Use processes_per_layer in config for custom distribution.")
            sys.exit(1)

    if input_splits_per_layer is not None:
        if processes_per_layer is None:
            print("Error: input_splits_per_layer requires processes_per_layer to be set")
            sys.exit(1)
        if len(input_splits_per_layer) != len(layer_sizes):
            print(f"Error: input_splits_per_layer length ({len(input_splits_per_layer)}) must match number of layers ({len(layer_sizes)})")
            sys.exit(1)
        if input_splits_per_layer[0] != 1:
            print("Error: input_splits_per_layer[0] must be 1 (first layer has no incoming input to split)")
            sys.exit(1)
        if input_splits_per_layer[-1] != 1:
            print("Error: input_splits_per_layer[-1] must be 1 (last layer must be unsplit)")
            sys.exit(1)
        if not all(s >= 1 for s in input_splits_per_layer):
            print("Error: all input_splits_per_layer entries must be >= 1")
            sys.exit(1)
        expected_size = sum(p * s for p, s in zip(processes_per_layer, input_splits_per_layer))
        if expected_size != size:
            print(f"Error: sum(processes_per_layer * input_splits_per_layer) = {expected_size} must equal MPI size ({size})")
            sys.exit(1)
        sparsity = config.get('sparsity_impact', (0,) * len(layer_sizes))
        if any(s > 1 for s in input_splits_per_layer) and any(v > 0 for v in sparsity):
            print("Error: sparsity_impact must be all 0 when any input split > 1")
            sys.exit(1)

    get_layer_idx(batch_size, layer_sizes, processes_per_layer, input_splits_per_layer, trial)

    # Fields persisted into the result JSON alongside the Params (set once, after
    # get_layer_idx has populated the *_global vars).
    result_extra_fields_global = {"processes_per_layer": processes_per_layer_global,
                                  "input_splits_per_layer": input_splits_per_layer_global}
    if dataset == "nmnist":
        result_extra_fields_global["first_saccade_only"] = config['first_saccade_only']

    if batch_size % mpi_config.get_process_per_batch != 0:
        print(f"Error: batch_size ({batch_size}) must be divisible by processes per layer ({mpi_config.get_process_per_batch})")
        sys.exit(1)

    key, subkey = jax.random.split(key)
    total_train_batches, total_val_batches, total_test_batches, max_nonzero = 0, 0, 0, 0
    weights = init_params(subkey, batch_size, layer_sizes, load_file=load_file, best=best)

    if rank == 0:
        downsample = False
        match dataset:
            case "mnist" | "smnist" | "psmnist":
                sequential = dataset in ("smnist", "psmnist")
                permuted = dataset == "psmnist"
                if layer_sizes[0] == 14*14:
                    downsample = True
                loader = partial(mnist_loader_manual, sequential=sequential, permuted=permuted)
            case "shd":
                loader = torch_SHD_loader
            case "nmnist":
                loader = partial(torch_nmnist_loader, first_saccade_only=config['first_saccade_only'])
            case "dvs":
                if layer_sizes[0] == 64*64*2:
                    downsample = True
                loader = partial(torch_DVSGesture_loader)
            case "ncars":
                if layer_sizes[0] == 60 * 50 * 2:
                    downsample = True
                loader = partial(torch_NCARS_loader)
            case "cifar10":
                if layer_sizes[0] == 16 * 16 * 3:
                    downsample = True
                loader = cifar10_loader_manual
            case _:
                raise ValueError(f"Unknown dataset: {dataset}")

        train_data, val_data, test_data, max_nonzero = loader(
            batch_size=batch_size,
            shuffle=False,
            CNN_preprocess=False, 
            downsample=downsample,
            data_dir=data_dir,
        )
        training_generator, total_train_batches = train_data
        validation_generator, total_val_batches = val_data
        test_generator, total_test_batches = test_data

    total_train_batches, total_val_batches, total_test_batches = bcast(jnp.array([total_train_batches, total_val_batches, total_test_batches]), root=0, comm=comm)
    max_nonzero = bcast(jnp.array([max_nonzero]), root=0, comm=comm)
    max_nonzero = max_nonzero.tolist()[0]

    thresholds = jnp.full(layer_sizes[layer_idx], init_thresholds)

    params = Params(
        dataset=dataset,
        random_seed=random_seed,
        layer_sizes=layer_sizes,
        init_thresholds=init_thresholds,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        batch_size=batch_size,
        load_file=load_file,
        shuffle_activations=config['shuffle_activations'],
        restrict=config['restrict'],
        firing_nb=config['firing_nb'] if isinstance(config['firing_nb'], int) else tuple(config['firing_nb']),
        sync_rate=config['sync_rate'] if isinstance(config['sync_rate'], int) else tuple(config['sync_rate']),
        max_nonzero=max_nonzero,
        shuffle_input=config['shuffle_input'],
        threshold_lr=config['threshold_lr'],
        sparsity_impact=tuple(config['sparsity_impact']),
        w_reg=config['w_reg'],
        rerun=None,
        top_weights=config['top_weights'],
        history_size=config['history_size'],
        use_bias=config['use_bias'],
        output_decay=config.get('output_decay', 1.0),
    )

    if trial is not None:
        params = dataclasses.replace(trial_params, max_nonzero=max_nonzero)

    if rerun is not None:
        override_list = config.get('override_params', None)
        params, weights, thresholds = rerun_init(
            rerun,
            mpi_config,
            params,
            override_params=override_list,
        )

    if rank == 0:
        print(f"Number of training batches: {total_train_batches}, validation batches: {total_val_batches}, test batches: {total_test_batches}")
        print(params)

    prev_size, cur_size = layer_sizes[layer_idx-1], layer_sizes[layer_idx]
    sr = params.sync_rate if isinstance(params.sync_rate, int) else params.sync_rate[layer_idx]
    sync_rate_vector = jnp.full(shape=(cur_size,), fill_value=sr)
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
        input_vector=jnp.zeros((prev_size,), dtype=int),
        output_vector=jnp.zeros((cur_size,), dtype=int),
        sync_rate_vector=sync_rate_vector,
        # Only the output layer records history; other layers keep an empty buffer.
        values_history=jnp.zeros((params.history_size if layer_idx == last_layer else 0, cur_size)),
        history_index=jnp.array(0, dtype=jnp.int32),
    )
    weights, empty_neuron_states = mpi_config.MPI_partition(weights, empty_neuron_states)
    total_batches = (total_train_batches, total_val_batches, total_test_batches)

    if mode == 'inference':
        batch_predict(params, key, total_batches, weights, empty_neuron_states, "test", save=True, debug=True)
    elif mode == 'training':
        val_acc, result_path = train(params, key, total_batches, weights, empty_neuron_states, config['optimizer'], trial)
    else:
        print(f"Unknown mode in config file, choose either 'training' or 'inference', got {mode}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train async neural network (multi-process per layer)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--data_dir', type=str, default="",
                       help='Directory for storing and reading the datasets')
    args, unknown = parser.parse_known_args()

    random_seed = args.seed
    key = jax.random.key(random_seed)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    main(random_seed, key, rank, size, comm, config_path=args.config, data_dir=args.data_dir)
'''
JAX_PLATFORMS=cpu mpirun -n 6 python async_MLP_general.py --config "configs/MLP_general_config.yaml"
'''