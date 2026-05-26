import os
os.environ["JAX_PLATFORMS"] = "cpu"
import torch

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

from other_helpers.helpers import BaseParams, NeuronStates
from other_helpers.helpers import accuracy, store_training_data, rerun_init, store_data_to_json
from other_helpers.helpers import activation_func, keep_top_k, output_vector_to_event
from other_helpers.helpers import update_history, process_history, load_config_with_defaults, parse_unknown_args_and_overrides_config
from forward_backward_pass.backpropagation import MLP_back_prop, RNN_back_prop
from forward_backward_pass.loss_functions import loss_bpp, loss_func
from other_helpers.general_MPI_helper import model_split

jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


@dataclasses.dataclass(frozen=True)
class Params(BaseParams):
    use_tanh: bool = False
    exact_rtrl: bool = False
    recurrence: tuple | None = None
    cell_type: str = "aed"
    fptt_parts: int = 1
    fptt_alpha: float = 0.1
    fptt_beta: float = 0.5
    fptt_lambda: float = 2.0
    fptt_rho: float = 0.0
    fptt_clip: float = 1.0
    fptt_warm_epochs: int = 1
    fptt_accumulate_logits: bool = True
    fptt_avg_logits: bool = False
    fptt_relu_output: bool = False


TQDM_DISABLE = False
BUFFER_SIZE = 0
END_SIGNAL = jnp.array([-1.0, -1.0], dtype=jnp.float32)

# Initialize empty global MPI variables
comm = None
rank = None      
size = None

layer_idx = None                   # Rank corresponding to the layer
processes_per_layer_global = None  # Number of processes for each layer
last_layer = None                  # Rank of last layer
batch_part_size = None             # The size of the batch on each process
mpi_config = None

training_generator = None
validation_generator = None
test_generator = None

# region INFERENCE
@partial(jax.jit, static_argnames=['params', 'grad'])
def layer_computation(params, key, neuron_idx, layer_input, weights, neuron_states, iteration=0, grad=False): 
    # jax.debug.print("rank {}, neuron idx {} weight array {}", rank, neuron_idx, weights[neuron_idx].shape)   
    # jax.debug.print("rank {}, layer input {}", rank, layer_input)   
    # Compute the new values of the neuron states
    filtered_weights = keep_top_k(weights[neuron_idx], params.top_weights, apply_abs=True)
    # filtered_weights = weights[neuron_idx]
    
    # APPLY THE RECURRENCE
    if params.recurrence[layer_idx] is not None:
        recurrent_activation = jnp.dot(neuron_states.prev_activated_output, neuron_states.recurrent_weight) # Shape (128,)
    else:
        recurrent_activation = jnp.zeros(neuron_states.values.shape) # Shape (128,)

    # jax.debug.print("Original weights {}, filtered wights {}", weights[neuron_idx], filtered_weights)
    invalid_idx = neuron_idx < 0
    activations = jax.lax.cond(
        invalid_idx,
        lambda _: neuron_states.values,
        lambda _: layer_input * filtered_weights + neuron_states.values + neuron_states.bias/params.max_nonzero + recurrent_activation,
        None
    )
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

    @jit
    def last_layer_case(_): # No need for additional computation at the output layer
        new_values_history, new_history_index = neuron_states.values_history, neuron_states.history_index
        if params.history_size > 0:
            new_values_history, new_history_index = update_history(new_values_history, new_history_index, activations)

        dummy_activations = jnp.zeros((activations.shape[0], 2))
        return jnp.array(0), dummy_activations, neuron_states.replace(  values=activations,
                                                                        input_residuals=new_input_residuals,
                                                                        input_activity=new_input_activity,
                                                                        values_history=new_values_history,
                                                                        history_index=new_history_index)
    
    @jit
    def hidden_layer_case(_):
        # APPLY THE SYNC RATE
        sync_fire = (iteration - neuron_states.last_sent_iteration >= neuron_states.sync_rate_vector).astype(jnp.int32)
        activated_output = activations * sync_fire # Mask out the neurons that don't meet the sync rate condition
        # jax.debug.print("rank {}, sync_fire: {}, iteration {}, sync rate vector {}, sync rate {}", rank, sync_fire.shape, iteration.shape, neuron_states.last_sent_iteration.shape,  neuron_states.sync_rate_vector.shape)

        if params.use_tanh:
            # Optionally wrap the state update with tanh (controlled by params.use_tanh)
            # Without tanh: z^t = activations - penalty + recurrent_activation
            # With tanh:    z^t = tanh(activations - penalty + recurrent_activation)
            #               O^t = ReLU(z^t)
            # With tanh the recurrent input must use the previous step's send_output (= relu(tanh(z_{t-1})))
            # so the W_hh scale stays in (0,1) and matches the gradient-check forward pass.
            tanh_out = jnp.tanh(activated_output)
            activated_output = tanh_out
            # Store raw tanh(z) as the hidden state (needed for correct tanh' = 1 - z^2 in backprop).
            # Emit O_t = topk(sync(ReLU(tanh(z_t)))) to match the checked PyTorch dynamics.
            # new_values = tanh_out
            # send_output = jax.nn.relu(tanh_out)
            # send_output = send_output * sync_fire
            # send_output = keep_top_k(send_output, k)
        # else:
            # tanh_out = inner  # alias for consistency in tanh_deriv_curr below
            # new_values = inner
            # send_output = activated_output
            
        # APPLY ACTIVATION FUNCTION
        activated_output = activation_func(neuron_states.thresholds, activated_output)
        # jax.debug.print("rank {}, iteration {}, sync_fire: {}, activations {}, activated_output {}", rank, iteration, sync_fire, activations, activated_output)

        # APPLY THE FIRING NUMBER
        f_nb = params.firing_nb
        k = f_nb if isinstance(f_nb, int) else f_nb[layer_idx]
        activated_output = keep_top_k(activated_output, k) # Get the top k activations

        new_last_sent_iteration = neuron_states.last_sent_iteration# jax.lax.cond(fire, lambda _: iteration, lambda _: neuron_states.last_sent_iteration, None)

        # APPLY THE RESTRICTION
        if params.restrict[layer_idx] <= 0:
            penalty = activated_output
        else:
            penalty = activated_output * params.restrict[layer_idx]

        send_output = activated_output
        new_values = activations - penalty
        if grad:
            # Keep recurrence gradient bookkeeping aligned with the selected events
            active_indexes = jnp.where(send_output != 0, 1, 0)  # Update the layer activity by adding the neurons that activated

            # Tanh derivatives for backprop
            # tanh_deriv_curr: derivative of relu(tanh(activations)) w.r.t. activations
            #   = (tanh_out > 0) * (1 - tanh_out^2), i.e. tanh'(act) where tanh > 0, else 0
            # active_tanh = m_i * tanh_deriv_curr (effective derivative for firing neurons)
            if params.use_tanh:
                tanh_deriv_curr = (tanh_out > 0).astype(tanh_out.dtype) * (1.0 - tanh_out**2)
                active_tanh = active_indexes * tanh_deriv_curr    # m_i * T_i
            else:
                tanh_deriv_curr = jnp.ones_like(activations)
                active_tanh = active_indexes

            # --- A matrix: exact Jacobian dactivations_t / dactivations_{t-1} ---
            # A[k,j] = (1 - m_eff_{t-1}[k]) * delta_{kj} + m_eff_{t-1}[k] * W_hh[k,j]
            # where m_eff_{t-1} = prev_active * prev_tanh_deriv
            prev_active_f = neuron_states.prev_active.astype(float)  # (n_hidden,)
            W_hh = neuron_states.recurrent_weight  # (n_hidden, n_hidden)
            prev_td = neuron_states.prev_tanh_deriv if neuron_states.prev_tanh_deriv is not None else jnp.ones_like(prev_active_f)
            prev_m_eff = prev_active_f * prev_td  # (n_hidden,)
            A_T = (1.0 - prev_m_eff)[:, None] * jnp.eye(W_hh.shape[0]) + prev_m_eff[:, None] * W_hh

            # --- Update rnn_running_sum ---
            A_T_diag = (1.0 - prev_m_eff) + prev_m_eff * jnp.diag(W_hh)  # (n_hidden,)
            new_running_sum = neuron_states.rnn_running_sum * A_T_diag[None, :]  # (n_input, n_hidden)
            new_running_sum = jax.lax.cond(
                neuron_idx >= 0,
                lambda rs: rs.at[neuron_idx, :].add(layer_input),
                lambda rs: rs,
                new_running_sum
            )

            # --- Update rnn_total_sum (accumulate when neuron fires) ---
            # With tanh: accumulate with m_i * T_i instead of m_i
            new_total_sum = (neuron_states.rnn_total_sum
                            + new_running_sum * active_tanh[None, :])  # (n_input, n_hidden)

            # --- Bias trace: same diagonal propagation as W_ih, but input is always 1 ---
            # bias_running_sum propagates like rnn_running_sum but the source term is 1
            # bias_running_sum_new = bias_running_sum * A_T_diag + 1
            # bias_total_sum accumulates bias_running_sum * m_t_eff at each firing step
            new_bias_running_sum = neuron_states.bias_running_sum * A_T_diag + 1.0  # (n_hidden,)
            new_bias_total_sum = (neuron_states.bias_total_sum
                                  + new_bias_running_sum * active_tanh)  # (n_hidden,)

            # --- Compact recurrent accumulator for W_hh (diagonal approx) ---
            # source[:, n] = R_{i-1} for every target column n.
            # Recurrence: U_i = source(R_{i-1}) + U_{i-1} @ A_{i-1}
            # Total: S += U_i * (ReLU'(z_i) * T_i)[None, :]  (column-wise scale)
            recurrent_source = jnp.broadcast_to(
                neuron_states.prev_activated_output[:, None],
                neuron_states.rnn_running_product.shape,
            )
            recurrent_running_sum = (
                neuron_states.rnn_running_product @ A_T
                + recurrent_source
            )
            new_total_product_sum = (
                neuron_states.rnn_total_product_sum
                + recurrent_running_sum * active_tanh[None, :]
            )

            # --- Exact RTRL traces (when params.exact_rtrl is True) ---
            # These use the full A matrix for propagation and full einsum for gradient extraction
            # Never record exact traces for input/output layers (even if recurrence is enabled)
            use_exact_rtrl = params.exact_rtrl and (layer_idx != 0) and (layer_idx != last_layer)
            if use_exact_rtrl:
                H = W_hh.shape[0]
                eye_h = jnp.eye(H)

                # Exact W_hh trace: P_hh (H, H, H)
                # P_hh[m, n, j] = dactivations[j] / dW_hh[m, n]
                # Propagation: P_new[m,:,:] = P_old[m,:,:] @ A for each m
                new_exact_hh_running = jnp.einsum(
                    "mnk,kj->mnj", neuron_states.exact_hh_running, A_T
                )
                # Source: P[m, n, j] += o_prev[m] * I[n, j]
                new_exact_hh_running = new_exact_hh_running + (
                    neuron_states.prev_activated_output[:, None, None] * eye_h[None, :, :]
                )
                new_exact_hh_total = (
                    neuron_states.exact_hh_total
                    + new_exact_hh_running * active_tanh[None, None, :]
                )

                # Exact bias trace: Q_bias (H, H)
                # Q_bias[n, j] = dactivations[j] / dbias[n]
                # Propagation: Q_new = Q_old @ A + I
                new_exact_bias_running = (
                    neuron_states.exact_bias_running @ A_T + eye_h
                )
                new_exact_bias_total = (
                    neuron_states.exact_bias_total
                    + new_exact_bias_running * active_tanh[None, :]
                )

            last_neuron_idx = jnp.argmax(neuron_states.input_order) # Last neuron index in the input order
            new_neuron_idx = jax.lax.cond(neuron_idx < 0, lambda _: last_neuron_idx, lambda _: neuron_idx, None)

            replace_fields = dict(
                values=new_values,
                input_residuals=new_input_residuals,
                output_residuals=neuron_states.output_residuals + send_output,
                input_activity=new_input_activity,
                layer_activity=neuron_states.layer_activity + active_indexes,
                input_order=neuron_states.input_order.at[new_neuron_idx].set(iteration),                # Update the input activity by setting the input neuron to the iteration number
                output_activity=neuron_states.output_activity.at[new_neuron_idx].add(active_indexes),
                input_vector=neuron_states.input_vector.at[neuron_idx].set(iteration + 1),              # Set the input neuron to the iteration at which the input was received (# Added +1 so that we can differentiate between never activated (0) and activated at iteration 0 (1))
                output_vector=jnp.where(send_output > 0, iteration + 1, neuron_states.output_vector),   # Set the output neuron to the last iteration at which it activated     (Same as above for +1)
                last_sent_iteration=new_last_sent_iteration,
                # Diagonal RNN gradient fields:
                rnn_running_sum=new_running_sum,
                rnn_total_sum=new_total_sum,
                rnn_running_product=recurrent_running_sum,
                rnn_total_product_sum=new_total_product_sum,
                bias_running_sum=new_bias_running_sum,
                bias_total_sum=new_bias_total_sum,
                prev_active=active_indexes,
                prev_activated_output=send_output,
                prev_tanh_deriv=tanh_deriv_curr,
            )
            if use_exact_rtrl:
                replace_fields.update(
                    exact_hh_running=new_exact_hh_running,
                    exact_hh_total=new_exact_hh_total,
                    exact_bias_running=new_exact_bias_running,
                    exact_bias_total=new_exact_bias_total,
                )
            new_neuron_states = neuron_states.replace(**replace_fields)
        else:
            new_neuron_states = neuron_states.replace(
                values=new_values,
                input_residuals=new_input_residuals,
                input_activity=new_input_activity,
                last_sent_iteration=new_last_sent_iteration,
                prev_activated_output=send_output)

        valid_elements = jnp.count_nonzero(send_output)
        processed_output = output_vector_to_event(key, send_output, params, params.layer_sizes[layer_idx])

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
@partial(jax.jit, static_argnames=['params', 'grad'])
def predict(params, key, weights, empty_neuron_states, batch_data: jnp.ndarray, grad=False):
    '''
    MLP inference, each layer sends each event separately in the format: (index, value)
    -1 means end of data from previous layer
    -2 means placeholder data in the input layer
    '''
    def input_layer(x):
        x_p = jnp.array(x)

        if params.shuffle_input:
            perm = jax.random.permutation(key, x_p.shape[0])
            x_p = x_p[perm]

        def send_input(i, carry):
            mpi_config.forward_send(x_p[i])
            return i

        mask = (x_p != -2)
        loop_iterations = (jnp.count_nonzero(mask) / 2).astype(int)
        iteration = jax.lax.fori_loop(0, loop_iterations, send_input, 0)

        mpi_config.forward_send(END_SIGNAL)
        return iteration, jnp.zeros((BUFFER_SIZE, 2))

    def other_layers(neuron_states):
        def cond(state):
            _, _, finished, _, _ = state
            return finished < mpi_config.nb_previous

        def forward_pass(state):
            layer_input, neuron_states, finished, iteration, buffer = state

            def send_activations(loop_iterations, activated_output):
                def send_one(i, _):
                    mpi_config.forward_send(activated_output[i])
                    return None
                jax.lax.fori_loop(0, loop_iterations, send_one, None)

            received = mpi_config.forward_recv(2)
            neuron_idx, layer_input = received[0], received[1]
            neuron_idx = neuron_idx.astype(jnp.int32)

            neg_idx = neuron_idx == -1
            finished = jax.lax.cond(neg_idx, lambda _: finished + 1, lambda _: finished, operand=None)
            iteration = jax.lax.cond(neg_idx, lambda _: iteration, lambda _: iteration + 1, operand=None)

            loop_iterations, activated_output, new_neuron_states = layer_computation(
                params, key, neuron_idx, layer_input, weights, neuron_states, iteration, grad)

            if layer_idx != last_layer:
                send_activations(loop_iterations, activated_output)

            return layer_input, new_neuron_states, finished, iteration, buffer

        finished = jnp.array(0)
        layer_input = jnp.zeros(())
        initial_state = (layer_input, neuron_states, finished, 0, jnp.zeros((BUFFER_SIZE, 2)))
        layer_input, neuron_states, finished, iteration, buffer = jax.lax.while_loop(cond, forward_pass, initial_state)

        if layer_idx != last_layer:
            mpi_config.forward_send(END_SIGNAL, iteration)

        return layer_input, neuron_states, iteration - mpi_config.nb_previous, buffer

    @jit
    def loop_over_batches(_, x):
        neuron_states = empty_neuron_states
        if layer_idx == 0:
            iterations, buffer = input_layer(x)
            new_neuron_states = neuron_states
        else:
            _, new_neuron_states, iterations, buffer = other_layers(neuron_states)
        mpi4jax.barrier(comm=mpi_config.comm)
        return None, (new_neuron_states.values, iterations, new_neuron_states, buffer)

    _, (all_outputs, all_iterations, all_neuron_states, buffer) = jax.lax.scan(loop_over_batches, None, batch_data)

    mpi4jax.barrier(comm=mpi_config.comm)
    return all_outputs, all_iterations, all_neuron_states, buffer

#region Training helpers
@partial(jax.jit, static_argnames=['params'])
def predict_bwd(params, key, weights, empty_neuron_states, batch_data):
    '''
    B: batch_size
    '''
    all_outputs, iterations, all_neuron_states, buffer = (predict)(params, key, weights, empty_neuron_states, batch_data, grad=True)
    # jax.debug.print("rank {}, layer activity: {} max: {}, ending values: {}", rank, all_neuron_states.layer_activity[0], jax.vmap(jnp.max)(all_neuron_states.layer_activity), all_neuron_states.values[0])

    # Receive the gradients from the later layers
    next_grad = mpi_config.backward_recv()  # Shape: (B, layer_size)

    # Compute input's gradient and weight gradient
    weight_grad, th_grad, weight_res, _ = MLP_back_prop(params, all_neuron_states, next_grad, layer_idx)
    # weight_grad += 2 * params.w_reg * weights

    weight_grad, recurrent_weight_grad, weight_res, bias_grad = RNN_back_prop(params, all_neuron_states, next_grad, layer_idx)

    if layer_idx > 1:
        cur_relu_mask = (all_neuron_states.output_vector > 0).astype(next_grad.dtype)

        # Send gradient to the previous layer
        # With tanh: dL/d(input) = W^T @ (dL/dO * ReLU' * tanh'), where tanh' = 1 - z^2
        if params.use_tanh:
            # values = tanh(z); derivative of relu(tanh(z)) = (tanh(z) > 0) * (1 - tanh(z)^2)
            cur_tanh_mask = ((all_neuron_states.values > 0) * (1.0 - all_neuron_states.values**2)).astype(next_grad.dtype)
            send_grad = jnp.dot(next_grad * cur_relu_mask * cur_tanh_mask, weights.T) # Shape: (B, 128) @ (128, 784) = (B, 784)
        else:
            send_grad = jnp.dot(next_grad * cur_relu_mask, weights.T) # Shape: (B, 128) @ (128, 784) = (B, 784)
        mpi_config.backward_send(send_grad)

    # Sparsity loss gradients
    all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)

    scaling = jax.lax.cond(params.sparsity_impact[layer_idx] > 0,
                           lambda _: params.sparsity_impact[layer_idx] / (all_iterations * batch_part_size * processes_per_layer_global) ,
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

    return all_outputs, iterations, all_neuron_states, (weight_grad, th_grad, weight_sparsity_grad, th_sparsity_grad, recurrent_weight_grad, jnp.mean(next_grad, axis=0), bias_grad)

# Define the loss function
@partial(jax.jit, static_argnames=['params'])
def loss_fn(params, key, weights, empty_neuron_states, target, batch_data):
    all_outputs, iterations, all_neuron_states, buffer = (predict)(params, key, weights, empty_neuron_states, batch_data, grad=True)

    # Compute L2 weight regularization sum across layers
    leader_rank = layer_idx * processes_per_layer_global
    w_sum = jnp.sum(weights**2)
    if layer_idx != last_layer and rank == leader_rank:
        if layer_idx != 0:
            send(w_sum, dest=last_layer * processes_per_layer_global, tag=7, comm=comm)
    elif layer_idx == last_layer and rank == leader_rank:
        for _i in range(1, last_layer):
            _s = recv(w_sum, source=_i * processes_per_layer_global, tag=7, comm=comm)
            w_sum += _s

    # Compute Loss and loss gradient
    loss, loss_grad = jax.value_and_grad(loss_func)(all_outputs, target)
    loss_grad /= mpi_config.get_process_per_batch # Shape (B, 10)
    loss += params.w_reg * w_sum

    # Compute output gradient and weight gradient
    out_grad, weight_grad = jax.vmap(loss_bpp, in_axes=(None, 0, 0))(weights, all_neuron_states, loss_grad) # Shape (B, 128), (B, 128, 10)
    # jax.debug.print("Rank {}, loss: {}, loss grad mean: {}, weight grad mean: {}", rank, loss, (loss_grad.shape), (weight_grad.shape))
    mean_weight_grad = jnp.mean(weight_grad, axis=0) # Shape: (128, 10)
    mean_weight_grad += 2 * params.w_reg * weights
    mean_weight_grad = jnp.expand_dims(mean_weight_grad, axis=0)  # Shape: (1, 128, 10)

    # Send gradient to previous layers
    mpi_config.backward_send(out_grad)
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
    leader_rank = layer_idx * processes_per_layer_global
    # jax.debug.print("Rank {}, activations shape: {}, iterations shape: {}, last layer neuron values: {}", rank, (activations.shape), jnp.mean(iterations.shape), jnp.sum(all_neuron_states.values))

    activations = mpi_config.gather_batch(all_neuron_states.input_residuals, average=False) # Gather the weight gradients from all ranks in the same layer
    iterations = mpi_config.gather_batch(iterations, average=True) # Gather the iterations from all ranks in the same layer
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

        sparsity_L = all_activations / (all_iterations * batch_part_size * processes_per_layer_global)
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
    if params.recurrence[layer_idx] is not None:
        recurrent_opt_state = solver.init(empty_neuron_states.recurrent_weight)

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
            
        epoch_iterations = []
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
                    batch_x, batch_y = mpi_config.split_batch(params, batch_iterator, 2) # Split the dataset to all the ranks of the input layer
                # print(f"rank {rank} data has shape {(batch_x.shape)}, {(batch_y.shape)}")

                mpi_config.send_labels(batch_y, mpi_config.batch_first_and_last_rank[0]) # Send the labels to the output layer

                # Run the forward pass
                outputs, iterations, all_neuron_states, buffer = (predict)(params, subkey, weights, neuron_states, batch_data=jnp.array(batch_x))
                all_activations, all_iterations, sparsity_L = sparsity_loss(params, all_neuron_states, iterations)
            else:
                if layer_idx==last_layer: # Output layer
                    # Receive the labels from the input layer
                    y = mpi_config.recv_labels()
                    y_encoded = jnp.array(one_hot_encode(y, num_classes=params.layer_sizes[-1]))

                    # Run the forward and backward pass for the output layer
                    (loss, outputs, iterations, total_loss, history), gradients = (loss_fn)(params, subkey, weights, neuron_states, y_encoded, jnp.zeros((batch_part_size, params.layer_sizes[0])))

                    weight_grad = gradients[0]
                    weight_grad = mpi_config.combine_batch_avg(weight_grad) # Gather the weight gradients from all ranks in the same layer

                    # Store the accuracy, loss and history                    
                    valid_y, batch_correct = accuracy(i, outputs, y, iterations, False)           
                    # print(f"Batch {i}, Accuracy: {batch_correct}/{valid_y.shape[0]} ")         
                    epoch_correct += batch_correct
                    epoch_total += valid_y.shape[0]
                    
                    epoch_loss.append(loss)                    
                    if params.history_size > 0:
                        all_history.append(history)
                else: 
                    # Run the forward and backward pass for the hidden layers
                    outputs, iterations, all_neuron_states, grads = (predict_bwd)(params, subkey, weights, neuron_states, jnp.zeros((batch_part_size, params.layer_sizes[0])))
                    weight_grad, threshold_grad, weight_sparsity_grad, threshold_sparsity_grad, recurrent_weight_grad, next_grad, bias_grad = grads

                    threshold_grad = mpi_config.gather_batch(threshold_grad, average=True) # Gather the thresholds' gradients from all ranks in the same layer

                    weight_grad = mpi_config.combine_batch_avg(weight_grad) # Gather the weight gradients from all ranks in the same layer
                    
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

                    # print(f"rank {rank}, recurrence: {params.recurrence[layer_idx]}")
                    if params.recurrence[layer_idx] is not None:
                        rw = neuron_states.recurrent_weight
                        # print(f"updating recurrent weights of shape {rw.shape} with grad of shape {recurrent_weight_grad.shape}")

                        recurrent_updates, recurrent_opt_state = solver.update(recurrent_weight_grad, recurrent_opt_state, rw)
                        new_recurrence_weights = optax.apply_updates(rw, recurrent_updates)
                    else:
                        new_recurrence_weights = neuron_states.recurrent_weight
                    
                    b = empty_neuron_states.bias
                    if params.use_bias:
                        bias_updates, bias_opt_state = solver.update(bias_grad, bias_opt_state, b)
                        new_bias = optax.apply_updates(b, bias_updates)
                    else:
                        new_bias = b 

           
                    empty_neuron_states = empty_neuron_states.replace(
                                            bias=new_bias,
                                            thresholds=new_thresholds,
                                            recurrent_weight=new_recurrence_weights,)
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
            epoch_iterations.append(iterations[iterations > 0])
        
        # Compute the average iterations for each layer
        epoch_iterations = jnp.concatenate(epoch_iterations)
        mean = 0.0
        if epoch_iterations.size > 0:
            mean = jnp.mean(epoch_iterations)        
        all_mean_iterations.append(mean)
        all_mean_iterations = mpi_config.gather_batch(jnp.array(all_mean_iterations))
        all_mean_iterations = all_mean_iterations.tolist()
        
        if layer_idx != 0 and trial is None:
            jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points and a mean threshold of {}", rank, mean, epoch_iterations.shape[0], jnp.mean(empty_neuron_states.thresholds))
        
        # print("rank {} bias {}".format(rank, empty_neuron_states.bias))
        # Inference on the validation set
        val_accuracy, val_mean, _ = batch_predict(params, key, total_batches, weights, empty_neuron_states, dataset="val", save=False, debug=False)
        
        epoch_accuracy = 0.0
        if layer_idx == last_layer:
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
        
        if trial is not None: # If using Optuna Hyper-parameter tuner
            # Return values if the run is not promising and should be pruned  
            all_mean_it = mpi_config.combine_batch_avg(jnp.array(all_mean_iterations)) # Gather the weight gradients from all ranks in the same layer
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
                            "RNN",
                            all_history,
                            total_batches[0],
                            extra_fields={
                                "cell_type": params.cell_type,
                                "use_tanh": params.use_tanh,
                                "fptt_parts": params.fptt_parts,
                                "fptt_alpha": params.fptt_alpha,
                                "fptt_beta": params.fptt_beta,
                                "fptt_lambda": params.fptt_lambda,
                                "fptt_rho": params.fptt_rho,
                                "fptt_clip": params.fptt_clip,
                                "fptt_warm_epochs": params.fptt_warm_epochs,
                                "fptt_accumulate_logits": params.fptt_accumulate_logits,
                                "fptt_avg_logits": params.fptt_avg_logits,
                                "fptt_relu_output": params.fptt_relu_output,
                            })
        
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
        leader_rank = mpi_config.get_last_layer_batch_leader
        if not mpi_config.is_last_layer_leader:
            all_iteration_mean = jnp.zeros(mpi_config.get_process_per_batch) # Share iterations mean to the rank 0
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
        
        # Random initializatoin of the weights       
        shape = (layer_sizes[layer_idx-1], layer_sizes[layer_idx])
        if len(shape) == 4:
            fan_in = shape[1] * shape[2] * shape[3]  # (out, in, kh, kw)
        elif len(shape) == 2:
            fan_in = shape[0]  # linear layer
        else:
            raise ValueError("Unsupported shape for Kaiming init")
        
        std = jnp.sqrt(2/fan_in)
        print("std: ", std)
        weights = random_layer_params(layer_sizes[layer_idx], layer_sizes[layer_idx-1], keys[layer_idx], scale=std)
        # print(f"rank {rank} Weights shape: {weights.shape}")
        return weights
    else:
        weights = jnp.zeros((layer_sizes[-1], layer_sizes[0]))
        return weights

def init_recurrent(key, N, gain=1.0):
    """
    Initialize recurrent weights ~ N(0, gain^2 / N)
    """
    W = jax.random.normal(key, shape=(N, N))
    W = W * (gain / jnp.sqrt(N))
    # W = jnp.eye(N)
    print(f"rank {rank} Recurrent Weights shape: {W.shape}")
    return W

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
    
    epoch_iterations = []
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
                batch_x, batch_y = mpi_config.split_batch(params, batch_iterator, 2)
            # store_data_to_json(f"{len(params.layer_sizes)}hidden_single_input.json", batch_x.tolist()) # Store for hardware usage

            # Run the forward pass
            outputs, iterations, all_neuron_states, buffer = (predict)(params, key, weights, neuron_states, jnp.array(batch_x))

            # Send label to the last layer
            mpi_config.send_labels(batch_y, mpi_config.batch_first_and_last_rank[0])
        else:
            # Run forward pass for hidden and output layers
            outputs, iterations, all_neuron_states, buffer = (predict)(params, key, weights, neuron_states, jnp.zeros((batch_part_size, params.layer_sizes[0])))

            if layer_idx == last_layer: # Output layer
                # Receive the labels from the input layer and compute the accuracy
                y = mpi_config.recv_labels()
                
                valid_y, batch_correct = accuracy(i, outputs, y, iterations, print=False)                 
                
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
    mean = 0.0
    if epoch_iterations.size > 0:
        mean = jnp.mean(epoch_iterations)     
    mean = mpi_config.gather_batch(jnp.array(mean))

    if rank != 0 and debug:
        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, mean, epoch_iterations.shape[0]*processes_per_layer_global)

    epoch_accuracy = -1.0
    if layer_idx == last_layer: # Output layer
        print(f"epoch correct {epoch_correct}, epoch total: {epoch_total}")
        epoch_accuracy = epoch_correct / epoch_total
        epoch_accuracy = mpi_config.gather_batch(epoch_accuracy)
        if debug:
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
                                "RNN",
                                all_history,
                                total_batches,
                                extra_fields={
                                    "cell_type": params.cell_type,
                                    "use_tanh": params.use_tanh,
                                    "fptt_parts": params.fptt_parts,
                                    "fptt_alpha": params.fptt_alpha,
                                    "fptt_beta": params.fptt_beta,
                                    "fptt_lambda": params.fptt_lambda,
                                    "fptt_rho": params.fptt_rho,
                                    "fptt_clip": params.fptt_clip,
                                    "fptt_warm_epochs": params.fptt_warm_epochs,
                                    "fptt_accumulate_logits": params.fptt_accumulate_logits,
                                    "fptt_avg_logits": params.fptt_avg_logits,
                                    "fptt_relu_output": params.fptt_relu_output,
                                })
    return epoch_accuracy, mean, end_time - start_time

# region Main
def get_layer_idx(batch_size, layer_sizes, trial=None):
    '''
    Define for each MPI rank:
    - layer_idx:                   Which layer it belongs to
    - processes_per_layer_global:  How many MPI processes there are per layer
    - last_layer:                  The index of the last layer
    - batch_part_size:             The size of the batch each rank has to process
    '''
    global layer_idx
    global processes_per_layer_global
    global last_layer
    global batch_part_size
    global mpi_config

    mpi_config = model_split(rank, comm, size, batch_size, layer_sizes)

    layer_idx = mpi_config.layer_idx
    last_layer = mpi_config.last_layer_idx
    batch_part_size = mpi_config.batch_part.get_size
    processes_per_layer_global = mpi_config.get_process_per_batch

    if trial is None:
        mpi_config.print()

# ===========================================================================
#region FPTT with MinimalRNN — cell, predict_chunk, backward, training loop
# ===========================================================================

# ---------------------------------------------------------------------------
# MinimalRNN layer computation (replaces layer_computation for minimalrnn)
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=['params', 'grad'])
def minimalrnn_layer_computation(params, key, neuron_idx, layer_input,
                                  rnn_weights, neuron_states, iteration=0, grad=False):
    """
    SEED MinimalRNN cell update for a single incoming event (hidden layers only).

    Equations:
        phi_pre_t = value * W_phi[neuron_idx, :] + b_phi
        z_t       = tanh(relu(phi_pre_t))                   # relu before tanh for sparsity
        th        = neuron_states.threshold                  # per-neuron threshold (init 0)
        s_prev    = H(h_prev - th)                          # spike mask (Heaviside)
        y_prev    = h_prev * s_prev                         # sparse previous output
        u_t       = sigmoid(W_gate @ [y_prev, z_t] + b_gate)
        h_t       = u_t * h_prev + (1 - u_t) * z_t - s_prev * th
        s_t       = H(h_t - th)                             # current spike mask
        y_t       = h_t * s_t                               # output (sent to next layer)

    With th=0: s = H(h), y = relu(h), reduces to relu-gated MinimalRNN.
    Straight-through estimator used for Heaviside in backward pass.
    """
    W_phi = rnn_weights['W_phi']    # (input_dim, H)
    b_phi = rnn_weights['b_phi']    # (H,)
    W_gate = rnn_weights['W_gate']  # (2*H, H)
    b_gate = rnn_weights['b_gate']  # (H,)
    h_prev = neuron_states.h_state  # (H,)
    H = h_prev.shape[0]

    invalid_idx = neuron_idx < 0

    # z_t = tanh(value * W_phi[idx] + b_phi), zeroed on END_SIGNAL
    safe_idx = jnp.where(invalid_idx, 0, neuron_idx)
    phi_pre_unclamped = layer_input * W_phi[safe_idx] + b_phi
    phi_pre = jnp.where(invalid_idx, jnp.zeros(H), phi_pre_unclamped)
    z_t = jnp.tanh(phi_pre)

    # Gate: u = sigmoid([h_prev, z_t] @ W_gate + b_gate)  — matches FPTTMinimalRNNAED
    cat_hz = jnp.concatenate([h_prev, z_t])        # (2*H,)
    u_t = jax.nn.sigmoid(jnp.dot(cat_hz, W_gate) + b_gate)

    # State update, pass-through on END_SIGNAL
    h_new_unclamped = u_t * h_prev + (1.0 - u_t) * z_t
    h_new = jnp.where(invalid_idx, h_prev, h_new_unclamped)

    # Output: relu(h_new) — matches FPTTMinimalRNNAED
    y_t = jax.nn.relu(h_new)

    # Apply sync_rate
    sync_fire = (iteration - neuron_states.last_sent_iteration >= neuron_states.sync_rate_vector).astype(jnp.int32)
    output = y_t * sync_fire

    # Apply firing number (top-k)
    f_nb = params.firing_nb
    k = f_nb if isinstance(f_nb, int) else f_nb[layer_idx]
    output = keep_top_k(output, k)

    send_output = output
    new_values = h_new

    # Gradient trace storage — skip END_SIGNAL events (neuron_idx < 0).
    # Use unconditional masked stores (no jax.lax.cond) — avoids branch overhead in while_loop.
    if grad:
        trace_idx = neuron_states.trace_index
        valid_event = ~invalid_idx                   # True for real events, False for END_SIGNAL
        valid_f = valid_event.astype(jnp.float32)   # 1.0 for real events, 0.0 for END_SIGNAL
        valid_i = valid_event.astype(jnp.int32)

        def masked_set(buf, val):
            """Write val at trace_idx only if valid_event, else write existing value (no-op)."""
            return buf.at[trace_idx].set(jnp.where(valid_event, val, buf[trace_idx]))

        new_all_z          = masked_set(neuron_states.all_z,          z_t)
        new_all_u          = masked_set(neuron_states.all_u,          u_t)
        new_all_h_prev     = masked_set(neuron_states.all_h_prev,     h_prev)   # stores h_prev
        new_all_neuron_idx = masked_set(neuron_states.all_neuron_idx, jnp.where(valid_event, neuron_idx, neuron_states.all_neuron_idx[trace_idx]))
        new_all_value      = masked_set(neuron_states.all_value,      layer_input)
        # Store h_new directly so backward can reconstruct s_t = (h_new > 0)
        new_all_h_new      = masked_set(neuron_states.all_h_new,      h_new)
        new_trace_index    = trace_idx + valid_i

        active_indexes = jnp.where(send_output != 0, 1, 0)
        new_ns = neuron_states.replace(
            values=new_values,
            h_state=h_new,
            layer_activity=neuron_states.layer_activity + active_indexes,
            prev_activated_output=send_output,
            all_z=new_all_z,
            all_u=new_all_u,
            all_h_prev=new_all_h_prev,
            all_h_new=new_all_h_new,
            all_neuron_idx=new_all_neuron_idx,
            all_value=new_all_value,
            trace_index=new_trace_index,
        )
    else:
        new_ns = neuron_states.replace(
            values=new_values,
            h_state=h_new,
            prev_activated_output=send_output,
        )

    valid_elements = jnp.count_nonzero(send_output)
    processed_output = output_vector_to_event(key, send_output, params, params.layer_sizes[layer_idx])
    return valid_elements, processed_output, new_ns


# ---------------------------------------------------------------------------
# predict_chunk: forward pass for one FPTT chunk
# ---------------------------------------------------------------------------
@jit
def input_layer_one_sample(neuron_states, x, key):
    """Send one sample's chunk events (input layer). JIT-compiled, called in Python loop."""
    x_p = jnp.array(x)  # (chunk_len, 2)
    def send_input(i, carry):
        data = x_p[i]
        send(data, dest=rank + processes_per_layer_global, tag=0, comm=comm)
        return carry
    def first_not_minus2(row):
        return (row != -2)
    mask = jax.vmap(first_not_minus2)(x_p)
    loop_iterations = (jnp.count_nonzero(mask) / 2).astype(int)
    jax.lax.fori_loop(0, loop_iterations, send_input, None)
    send(END_SIGNAL, dest=rank + processes_per_layer_global, tag=0, comm=comm)
    return neuron_states, loop_iterations


@partial(jax.jit, static_argnames=['params', 'grad'])
def hidden_layer_one_sample(params, key, rnn_weights, neuron_states, grad=False):
    """Process one sample's chunk events (hidden layer). JIT-compiled, called in Python loop."""
    H_size = neuron_states.h_state.shape[0]

    # State: (layer_input, neuron_states, neuron_idx, iteration, relu_sum)
    # relu_sum is only meaningful for the last hidden layer dense-output branch.
    def cond(state):
        _, _, neuron_idx, _, _ = state
        return neuron_idx != -1

    if layer_idx < last_layer - 1:
        # Intermediate hidden layer: forward top-k events to next hidden layer
        def forward_pass(state):
            layer_input, neuron_states, neuron_idx, iteration, relu_sum = state
            (neuron_idx, layer_input) = recv(
                jnp.zeros((2,)), source=rank - processes_per_layer_global, tag=0, comm=comm)
            loop_iterations, activated_output, new_neuron_states = minimalrnn_layer_computation(
                params, key, neuron_idx.astype(int), layer_input,
                rnn_weights, neuron_states, iteration, grad)
            def send_one(i, _carry):
                send(activated_output[i], dest=rank + processes_per_layer_global, tag=0, comm=comm)
                return _carry
            jax.lax.fori_loop(0, loop_iterations, send_one, None)
            return layer_input, new_neuron_states, neuron_idx, iteration + 1, relu_sum

    elif params.fptt_relu_output:
        # Last hidden layer, event-based output: send relu top-k events to output layer
        def forward_pass(state):
            layer_input, neuron_states, neuron_idx, iteration, relu_sum = state
            (neuron_idx, layer_input) = recv(
                jnp.zeros((2,)), source=rank - processes_per_layer_global, tag=0, comm=comm)
            loop_iterations, activated_output, new_neuron_states = minimalrnn_layer_computation(
                params, key, neuron_idx.astype(int), layer_input,
                rnn_weights, neuron_states, iteration, grad)
            def send_one(i, _carry):
                send(activated_output[i], dest=rank + processes_per_layer_global, tag=0, comm=comm)
                return _carry
            jax.lax.fori_loop(0, loop_iterations, send_one, None)
            return layer_input, new_neuron_states, neuron_idx, iteration + 1, relu_sum

    else:
        # Last hidden layer, dense output: accumulate keep_top_k(relu(h_t), firing_nb) per
        # timestep. Matches FPTTMinimalRNNAED.forward_chunk:
        #   o_last_total += keep_top_k(relu(h_t), firing_nb)  (not the full relu vector).
        # For dense checkpoints (firing_nb > H) keep_top_k is a no-op.
        _k_out = params.firing_nb if isinstance(params.firing_nb, int) else params.firing_nb[layer_idx]
        def forward_pass(state):
            layer_input, neuron_states, neuron_idx, iteration, relu_sum = state
            (neuron_idx, layer_input) = recv(
                jnp.zeros((2,)), source=rank - processes_per_layer_global, tag=0, comm=comm)
            loop_iterations, activated_output, new_neuron_states = minimalrnn_layer_computation(
                params, key, neuron_idx.astype(int), layer_input,
                rnn_weights, neuron_states, iteration, grad)
            valid = (neuron_idx >= 0).astype(jnp.float32)
            relu_sum = relu_sum + keep_top_k(jax.nn.relu(new_neuron_states.h_state), _k_out) * valid
            return layer_input, new_neuron_states, neuron_idx, iteration + 1, relu_sum

    layer_input = jnp.zeros(())
    relu_sum_init = jnp.zeros(H_size)
    initial_state = (layer_input, neuron_states, jnp.array(0.0, dtype=jnp.float32), 0, relu_sum_init)

    layer_input, neuron_states, neuron_idx, iteration, relu_sum = jax.lax.while_loop(
        cond, forward_pass, initial_state)

    # Send chunk-end signal after while_loop
    if params.fptt_relu_output and layer_idx == last_layer - 1:
        send(END_SIGNAL, dest=rank + processes_per_layer_global, tag=0, comm=comm)
    elif layer_idx < last_layer - 1:
        send(END_SIGNAL, dest=rank + processes_per_layer_global, tag=0, comm=comm)

    return neuron_states, iteration - 1, relu_sum


@partial(jax.jit, static_argnames=['params', 'grad'])
def predict_chunk(params, key, rnn_weights, empty_neuron_states,
                  batch_chunk_data, initial_h_states, grad=False):
    """
    Process one chunk of events for all batch samples.
    Uses jax.lax.scan over the batch, calling top-level JIT functions.

    Args:
        batch_chunk_data: (batch_size, chunk_len, 2) for input layer, ignored for hidden
        initial_h_states: (batch_size, H) — hidden states from previous chunk
        rnn_weights: dict with MinimalRNN weights (hidden layers only)
    Returns:
        all_h_finals: (batch_size, H)
        all_iterations: (batch_size,)
        all_neuron_states: batched NeuronStates (stacked along axis 0)
    """
    def loop_over_batches(_, inputs):
        x_chunk, h_init = inputs
        neuron_states = empty_neuron_states.replace(h_state=h_init)
        if layer_idx == 0:
            new_ns, iters = input_layer_one_sample(neuron_states, x_chunk, key)
            relu_sum = jnp.zeros_like(h_init)
        else:
            new_ns, iters, relu_sum = hidden_layer_one_sample(params, key, rnn_weights, neuron_states, grad)
        return None, (new_ns.h_state, iters, new_ns, relu_sum)

    _, (all_h_finals, all_iterations, all_neuron_states, all_relu_sums) = jax.lax.scan(
        loop_over_batches, None, (batch_chunk_data, initial_h_states))

    return all_h_finals, all_iterations, all_neuron_states, all_relu_sums


# ---------------------------------------------------------------------------
# MinimalRNN backward pass in JAX
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=['params'])
def minimalrnn_back_prop(params, rnn_weights, all_neuron_states, dL_dh_batch, dL_dh_output=None):
    """
    BPTT within one chunk for MinimalRNN hidden layer using stored traces.

    Args:
        rnn_weights: dict with W_phi, b_phi, W_gate, b_gate
        all_neuron_states: batched NeuronStates with trace buffers
        dL_dh_batch: (B, H) gradient of loss w.r.t. final hidden state (from next layer tag=2)
        dL_dh_output: (B, H) or None — when fptt_relu_output=True, the per-step gradient
                      d_logits @ W_out.T that should be injected at every timestep a neuron fired.
                      None means inject only at the final timestep (non-relu path).

    Returns:
        grads: dict mapping weight names to gradient arrays
        dL_dinput: (B, input_dim) gradient w.r.t. inputs (for sending to previous hidden layer)
    """
    W_phi = rnn_weights['W_phi']
    W_gate = rnn_weights['W_gate']
    H = int(W_phi.shape[1])
    _input_dim = int(W_phi.shape[0])  # input_dim of this layer (= H of previous hidden layer)

    def single_sample_backward(ns, dL_dh, dL_dh_out):
        """
        Backward for one sample — SEED MinimalRNN.

        Forward recap:
            phi_pre = val * W_phi[nidx] + b_phi
            z       = tanh(relu(phi_pre))             [stored as all_z]
            s_prev  = H(h_prev - th)
            y_prev  = h_prev * s_prev                 [stored as all_h_prev]
            u       = sigmoid([y_prev, z] @ W_gate + b_gate)  [stored as all_u]
            h_new   = u * h_prev + (1-u) * z - s_prev * th    [stored as all_h_new]
            s_t     = H(h_new - th)
            y_t     = h_new * s_t

        STE: dH/dh ≈ 1  →  d(y_prev)/d(h_prev) ≈ s_prev
        dL_dh_out: (H,) per-step output gradient injected at each firing timestep.
        """
        trace_len = ns.trace_index
        th         = ns.threshold    # (H,) per-neuron learned threshold
        all_z      = ns.all_z        # (L, H) — z_t = tanh(relu(phi_pre))
        all_u      = ns.all_u        # (L, H) — u_t
        all_h_prev = ns.all_h_prev   # (L, H) — y_prev = h_prev * s_prev
        all_h_new  = ns.all_h_new    # (L, H) — h_new after state update
        all_nidx   = ns.all_neuron_idx
        all_val    = ns.all_value
        # s_prev = H(h_prev - th): reconstructed as (y_prev > 0) only when th=0.
        # With learned th, use all_h_new to infer s_t, and y_prev > 0 still gives s_prev
        # since y_prev = h_prev * s_prev so y_prev > 0 iff s_prev = 1.

        grad_W_phi  = jnp.zeros_like(W_phi)
        grad_b_phi  = jnp.zeros(H)
        grad_W_gate = jnp.zeros_like(W_gate)
        grad_b_gate = jnp.zeros(H)
        grad_input  = jnp.zeros(_input_dim)
        grad_th     = jnp.zeros(H)

        max_len = all_z.shape[0]

        def bptt_step(t_rev, carry):
            dL_dh, g_W_phi, g_b_phi, g_W_gate, g_b_gate, g_input, g_th = carry
            t = trace_len - 1 - t_rev

            z      = all_z[t]
            u      = all_u[t]
            y_prev = all_h_prev[t]   # y_prev = relu(h_prev) at th=0
            h_new  = all_h_new[t]    # h_new (after state update)
            nidx   = all_nidx[t]
            val    = all_val[t]

            # Recompute STE masks with learned threshold
            # s_prev: y_prev = h_prev * s_prev, so y_prev > 0 iff s_prev = 1 (th-independent)
            s_prev = (y_prev > 0).astype(jnp.float32)
            # s_t: h_new stores the actual post-update value; compare against current th
            s_t    = (h_new > th).astype(jnp.float32)

            valid = (t_rev < trace_len).astype(jnp.float32)

            # Inject per-step output gradient at every firing timestep (s_t = H(h_new - th))
            dL_dh = dL_dh + dL_dh_out * s_t * valid

            # h_t = u * h_prev + (1-u) * z - s_prev * th
            # dL/du: approximate h_prev ≈ y_prev (STE)
            dL_du      = dL_dh * (y_prev - z) * valid
            dL_dz      = dL_dh * (1.0 - u) * valid
            # dL/d(h_prev): direct path through u * h_prev
            dL_dh_prev = dL_dh * u * valid
            # dL/d(th): soft-reset term (-s_prev*th) and output gate (STE: -s_t)
            g_th = g_th + dL_dh * (-s_prev) * valid + dL_dh_out * (-s_t) * valid

            # Through sigmoid gate: u = sigmoid([y_prev, z] @ W_gate + b_gate)
            sig_deriv   = u * (1.0 - u)
            dL_dgate_pre = dL_du * sig_deriv
            dL_dcat      = dL_dgate_pre @ W_gate.T   # (2*H,)
            # dL/d(y_prev) from gate; chain through STE: dL/d(h_prev) += dL/d(y_prev) * s_prev
            dL_dh_prev   = dL_dh_prev + dL_dcat[:H] * s_prev
            dL_dz_gate   = dL_dcat[H:]
            dL_dz_total  = dL_dz + dL_dz_gate

            # Through tanh(relu(phi_pre)):
            relu_mask   = (z > 0).astype(jnp.float32)
            tanh_deriv  = 1.0 - z ** 2
            dL_dphi_pre = dL_dz_total * tanh_deriv * relu_mask  # (H,)

            # Gate gradients: outer([dL_dgate_pre], [y_prev, z])
            cat_yz  = jnp.concatenate([y_prev, z])
            g_W_gate = g_W_gate + jnp.outer(dL_dgate_pre, cat_yz).T * valid  # (2*H, H)
            g_b_gate = g_b_gate + dL_dgate_pre * valid

            # Phi gradients
            per_row  = val * dL_dphi_pre * valid
            safe_idx = jnp.maximum(nidx, 0)
            g_W_phi  = g_W_phi.at[safe_idx].add(per_row)
            g_b_phi  = g_b_phi + dL_dphi_pre * valid

            # Input gradient
            dinput_val = jnp.dot(dL_dphi_pre, W_phi[safe_idx]) * valid
            g_input    = g_input.at[safe_idx].add(dinput_val)

            return dL_dh_prev, g_W_phi, g_b_phi, g_W_gate, g_b_gate, g_input, g_th

        init_carry = (dL_dh, grad_W_phi, grad_b_phi, grad_W_gate, grad_b_gate, grad_input, grad_th)
        final_carry = jax.lax.fori_loop(0, max_len, bptt_step, init_carry)
        _, grad_W_phi, grad_b_phi, grad_W_gate, grad_b_gate, grad_input, grad_th = final_carry

        return grad_W_phi, grad_b_phi, grad_W_gate, grad_b_gate, grad_input, grad_th

    # Build per-step output gradient batch: zeros if not relu path
    if dL_dh_output is None:
        dL_dh_output_batch = jnp.zeros_like(dL_dh_batch)  # (B, H) — no per-step injection
    else:
        dL_dh_output_batch = dL_dh_output  # (B, H) — d_logits @ W_out.T

    # vmap over batch
    g_W_phi, g_b_phi, g_W_gate, g_b_gate, g_input, g_th = jax.vmap(
        single_sample_backward)(all_neuron_states, dL_dh_batch, dL_dh_output_batch)

    # Sum over batch (loss_grad already carries 1/B normalization)
    grads = {
        'W_phi': jnp.sum(g_W_phi, axis=0),
        'b_phi': jnp.sum(g_b_phi, axis=0),
        'W_gate': jnp.sum(g_W_gate, axis=0),
        'b_gate': jnp.sum(g_b_gate, axis=0),
    }
    # g_th: (B, H) — sum over batch to get threshold gradient
    grad_th = jnp.sum(g_th, axis=0)  # (H,)
    # g_input: (B, input_dim) — return per-sample gradient for sending backward via MPI
    return grads, g_input, grad_th


# ---------------------------------------------------------------------------
# FPTT loss computation
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=['n_classes'])
def fptt_loss_and_grad(logits, y, oracle_prob, beta_p, n_classes):
    """
    Compute FPTT loss and gradient w.r.t. logits.

    Returns:
        loss: scalar
        d_logits: (B, C) gradient of loss w.r.t. logits
    """
    B = logits.shape[0]
    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    one_hot_y = jax.nn.one_hot(y, n_classes)
    ce_loss = -jnp.mean(jnp.sum(one_hot_y * log_probs, axis=-1))
    oracle_loss = -jnp.mean(jnp.sum(oracle_prob * log_probs, axis=-1))
    loss = beta_p * ce_loss + (1.0 - beta_p) * oracle_loss

    d_ce = (probs - one_hot_y) / B
    d_oracle = (probs - oracle_prob) / B
    d_logits = beta_p * d_ce + (1.0 - beta_p) * d_oracle

    return loss, d_logits


# ---------------------------------------------------------------------------
# Oracle update (vectorized, JIT-compatible)
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=['n_classes'])
def vectorized_oracle_update(oracle, probs, y, chunk_idx, n_classes):
    """
    Update oracle estimates for misclassified samples.
    n_classes is static so the Python loop unrolls at trace time.
    """
    preds = jnp.argmax(probs, axis=1)
    misclassified = (preds != y)

    def update_one_class(oracle, c):
        class_mask = (y == c) & misclassified
        has_any = jnp.any(class_mask)
        first_idx = jnp.argmax(class_mask)
        new_probs = probs[first_idx]
        oracle = jax.lax.cond(
            has_any,
            lambda o: o.at[c, chunk_idx].set(new_probs),
            lambda o: o,
            oracle)
        return oracle

    for c in range(n_classes):
        oracle = update_one_class(oracle, c)

    return oracle


# ---------------------------------------------------------------------------
# Output layer: receive relu events from last hidden layer and accumulate logits
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=['n_classes', 'H_hidden'])
def output_layer_recv_one_sample(weights, n_classes, H_hidden):
    """
    Receive sparse (neuron_idx, value) events from the last hidden layer (tag=0)
    until END_SIGNAL, accumulate logits and store inputs for weight gradient.

    Used when fptt_relu_output=True — mirrors async_MLP output layer behaviour.

    Returns:
        logits: (n_classes,) accumulated class scores for one sample
        sparse_input: (H_hidden,) reconstructed dense input (scatter-add of events)
        n_events: number of events received
    """
    def cond(state):
        _, _, neuron_idx, _ = state
        return neuron_idx >= 0.0  # float comparison: END_SIGNAL has neuron_idx = -1.0

    def body(state):
        logits, sparse_input, _neuron_idx, n_events = state
        event = recv(jnp.zeros((2,)), source=rank - processes_per_layer_global, tag=0, comm=comm)
        neuron_idx_f, value = event[0], event[1]
        valid = (neuron_idx_f >= 0.0).astype(jnp.float32)
        safe_idx = jnp.maximum(neuron_idx_f.astype(jnp.int32), 0)
        logits = logits + valid * value * weights[safe_idx]          # (n_classes,)
        sparse_input = sparse_input.at[safe_idx].add(valid * value)  # accumulate input values
        return logits, sparse_input, neuron_idx_f, n_events + valid.astype(jnp.int32)

    init = (jnp.zeros(n_classes), jnp.zeros(H_hidden),
            jnp.array(0.0, dtype=jnp.float32), jnp.array(0, dtype=jnp.int32))
    logits, sparse_input, _, n_events = jax.lax.while_loop(cond, body, init)
    return logits, sparse_input, n_events


def output_layer_recv_events_batch(weights, batch_part, n_classes, H_hidden):
    """
    Receive events for a full batch (batch_part samples) sequentially.
    Python loop over JIT-compiled per-sample receive to avoid expensive scan+while_loop trace.
    Returns:
        all_logits: (batch_part, n_classes)
        all_sparse_inputs: (batch_part, H_hidden) — for weight gradient computation
        total_events: scalar
    """
    all_logits = []
    all_sparse_inputs = []
    total_events = 0
    for _ in range(batch_part):
        logits, sparse_input, n_events = output_layer_recv_one_sample(weights, n_classes, H_hidden)
        all_logits.append(logits)
        all_sparse_inputs.append(sparse_input)
        total_events = total_events + n_events
    return jnp.stack(all_logits), jnp.stack(all_sparse_inputs), total_events


# ---------------------------------------------------------------------------
# Consensus regularizer helpers
# ---------------------------------------------------------------------------
def init_consensus_state(rnn_weights):
    sm = jax.tree.map(lambda p: p.copy(), rnn_weights)
    lm = jax.tree.map(lambda p: jnp.zeros_like(p), rnn_weights)
    return sm, lm


def consensus_reg_grad(rnn_weights, sm, lm, alpha, lmbda, rho):
    return jax.tree.map(
        lambda p, s, l: (rho - 1.0) * l + lmbda * alpha * (p - s),
        rnn_weights, sm, lm)


def post_optimizer_update(rnn_weights, sm, lm, alpha, beta):
    new_lm = jax.tree.map(
        lambda l, p, s: l + (-alpha * (p - s)),
        lm, rnn_weights, sm)
    new_sm = jax.tree.map(
        lambda s, p, l: (1.0 - beta) * s + beta * p - (beta / alpha) * l,
        sm, rnn_weights, new_lm)
    return new_sm, new_lm


# ---------------------------------------------------------------------------
# MinimalRNN weight initialization
# ---------------------------------------------------------------------------
def init_minimalrnn_weights(key, input_dim, hidden_dim):
    k1, k2 = jax.random.split(key, 2)
    std_phi = jnp.sqrt(2.0 / input_dim)
    W_phi = jax.random.normal(k1, (input_dim, hidden_dim)) * std_phi
    b_phi = jnp.zeros(hidden_dim)

    gate_input_dim = 2 * hidden_dim
    std_gate = jnp.sqrt(2.0 / (gate_input_dim + hidden_dim))
    W_gate = jax.random.normal(k2, (gate_input_dim, hidden_dim)) * std_gate
    b_gate = jnp.full(hidden_dim, 3.0)  # HiPPO-motivated: u = sigmoid(3) ≈ 0.95, memory-preserving init

    return {
        'W_phi': W_phi, 'b_phi': b_phi,
        'W_gate': W_gate, 'b_gate': b_gate,
    }


def _validate_checkpoint_shape(sd, layer_sizes, pt_path):
    """Raise ValueError if the checkpoint architecture doesn't match layer_sizes."""
    n_hidden_ckpt = 1 + len([k for k in sd if k.startswith('W_phi_layers') and k.endswith('.weight')])
    n_hidden_cfg  = len(layer_sizes) - 2  # exclude input and output layers
    if n_hidden_ckpt != n_hidden_cfg:
        raise ValueError(
            f"Checkpoint '{pt_path}' has {n_hidden_ckpt} hidden layer(s) "
            f"but config layer_sizes {list(layer_sizes)} requires {n_hidden_cfg}. "
            f"Use mpirun -n {n_hidden_ckpt + 2} and matching layer_sizes."
        )
    H_ckpt = sd['W_phi'].shape[1]
    for li in range(1, len(layer_sizes) - 1):
        H_cfg = layer_sizes[li]
        if H_cfg != H_ckpt:
            raise ValueError(
                f"Checkpoint '{pt_path}' has hidden size H={H_ckpt} "
                f"but layer_sizes[{li}]={H_cfg}."
            )
    n_classes_ckpt = sd['W_out'].shape[1]
    n_classes_cfg  = layer_sizes[-1]
    if n_classes_ckpt != n_classes_cfg:
        raise ValueError(
            f"Checkpoint '{pt_path}' has {n_classes_ckpt} output classes "
            f"but layer_sizes[-1]={n_classes_cfg}."
        )


def load_minimalrnn_aed_weights(pt_path, layer_idx, layer_sizes):
    """
    Load FPTTMinimalRNNAED weights from a .pt checkpoint into the JAX rnn_weights dict.

    Layer 0 (first hidden, layer_idx=1):
        W_phi:  state_dict['W_phi']        (n_input, H)
        b_phi:  state_dict['b_phi']        (H,)
        W_gate: state_dict['W_gate']       (2H, H)
        b_gate: state_dict['b_gate']       (H,)

    Layer l (deeper hidden, layer_idx >= 2):
        W_phi:  state_dict['W_phi_layers.{l-1}.weight'].T   (H_in, H_out) after transpose
        b_phi:  state_dict['W_phi_layers.{l-1}.bias']       (H,)
        W_gate: state_dict['W_gate_layers.{l-1}.weight'].T  (2H, H) after transpose
        b_gate: state_dict['W_gate_layers.{l-1}.bias']      (H,)

    Returns rnn_weights dict, or {} for input/output layers.
    """
    last = len(layer_sizes) - 1
    if layer_idx == 0 or layer_idx == last:
        return {}

    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    _validate_checkpoint_shape(sd, layer_sizes, pt_path)

    hidden_layer_idx = layer_idx - 1  # 0-based index into hidden layers

    if hidden_layer_idx == 0:
        W_phi  = jnp.array(sd['W_phi'].numpy())           # (n_input, H)
        b_phi  = jnp.array(sd['b_phi'].numpy())           # (H,)
        W_gate = jnp.array(sd['W_gate'].numpy())          # (2H, H)
        b_gate = jnp.array(sd['b_gate'].numpy())          # (H,)
    else:
        l = hidden_layer_idx - 1  # index into W_phi_layers / W_gate_layers
        # nn.Linear stores weight as (out, in); we need (in, out) for dot(input, W)
        W_phi  = jnp.array(sd[f'W_phi_layers.{l}.weight'].numpy().T)   # (H_in, H_out)
        b_phi  = jnp.array(sd[f'W_phi_layers.{l}.bias'].numpy())       # (H,)
        W_gate = jnp.array(sd[f'W_gate_layers.{l}.weight'].numpy().T)  # (2H, H)
        b_gate = jnp.array(sd[f'W_gate_layers.{l}.bias'].numpy())      # (H,)

    return {'W_phi': W_phi, 'b_phi': b_phi, 'W_gate': W_gate, 'b_gate': b_gate}


def load_minimalrnn_aed_output_weights(pt_path, layer_sizes):
    """Load W_out from a .pt checkpoint as a JAX array (H, n_classes)."""
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    _validate_checkpoint_shape(sd, layer_sizes, pt_path)
    return jnp.array(sd['W_out'].numpy())  # (H, n_classes)


# ---------------------------------------------------------------------------
# MinimalRNN inference (no gradients, no weight updates)
# ---------------------------------------------------------------------------
def batch_predict_minimalrnn(params: Params, key, total_batches, rnn_weights, weights, empty_neuron_states, dataset="test", save=True, debug=True):
    """
    Inference-only forward pass for MinimalRNN (cell_type='minimalrnn').
    Mirrors train_fptt's forward pass without gradients or weight updates.
    Output layer sends a zero gradient backward so hidden layers stay in sync.
    """
    global test_generator, validation_generator, training_generator

    mpi4jax.barrier(comm=comm)
    start_time = time.time()

    n_classes = params.layer_sizes[-1]
    H_hidden = params.layer_sizes[last_layer - 1]
    P = params.fptt_parts
    accumulate_logits = params.fptt_accumulate_logits
    avg_logits = params.fptt_avg_logits

    H = params.layer_sizes[layer_idx] if (layer_idx > 0 and layer_idx != last_layer) else 0

    if dataset == "test":
        total_batches_n = total_batches[2]
        if layer_idx == 0:
            batch_iterator = iter(test_generator) if rank == 0 else None
    elif dataset == "val":
        total_batches_n = total_batches[1]
        if layer_idx == 0:
            batch_iterator = iter(validation_generator) if rank == 0 else None
    else:
        total_batches_n = total_batches[0]
        if layer_idx == 0:
            batch_iterator = iter(training_generator) if rank == 0 else None

    epoch_correct = 0
    epoch_total = 0
    all_iteration_mean = []

    for i in tqdm(range(total_batches_n), disable=TQDM_DISABLE):
        if layer_idx == 0:
            batch_x, batch_y = mpi_config.split_batch(params, batch_iterator, 2)
            mpi_config.send_labels(batch_y, mpi_config.batch_first_and_last_rank[0])
        if layer_idx == last_layer:
            y = mpi_config.recv_labels()

        if layer_idx > 0 and layer_idx != last_layer:
            h_states = jnp.zeros((batch_part_size, H))
        else:
            h_states = jnp.zeros((batch_part_size, 1))

        T = params.max_nonzero
        step = T // P
        _P = P if P * step >= T else P + 1

        if layer_idx == last_layer and accumulate_logits:
            accumulated_logits = jnp.zeros((batch_part_size, n_classes))

        key, subkey = jax.random.split(key)

        for p in range(_P):
            start_t = p * step
            end_t = min(start_t + step, T)
            if start_t >= T:
                break
            chunk_len = end_t - start_t

            if layer_idx == last_layer:
                if params.fptt_relu_output:
                    chunk_logits, _, _ = output_layer_recv_events_batch(
                        weights, batch_part_size, n_classes, H_hidden)
                else:
                    relu_sum = recv(jnp.zeros((batch_part_size, H_hidden)),
                                    source=rank - processes_per_layer_global, tag=3, comm=comm)
                    chunk_logits = jnp.dot(relu_sum, weights)

                if accumulate_logits:
                    accumulated_logits = accumulated_logits + chunk_logits
                    logits = accumulated_logits / (p + 1) if avg_logits else accumulated_logits
                else:
                    logits = chunk_logits

                # Send zero gradient backward so hidden layers stay in sync
                mpi_config.backward_send(jnp.zeros((batch_part_size, H_hidden)))

            elif layer_idx > 0:
                chunk_data = jnp.zeros((batch_part_size, chunk_len, 2))
                all_h_finals, _, _, all_relu_sums = predict_chunk(
                    params, subkey, rnn_weights, empty_neuron_states,
                    chunk_data, h_states, grad=False)

                if layer_idx == last_layer - 1 and not params.fptt_relu_output:
                    send(all_relu_sums, dest=rank + processes_per_layer_global, tag=3, comm=comm)

                # Receive (and discard) the zero gradient from next layer
                mpi_config.backward_recv()
                if layer_idx > 1:
                    mpi_config.backward_send(jnp.zeros((batch_part_size, params.layer_sizes[layer_idx - 1])))

                h_states = all_h_finals

            else:
                chunk_data = jnp.array(batch_x[:, start_t:end_t, :])
                predict_chunk(params, subkey, rnn_weights, empty_neuron_states,
                              chunk_data, h_states, grad=False)

            mpi4jax.barrier(comm=comm)

        if layer_idx == last_layer:
            # Normalise by n_real_events (= max_nonzero for dense sequences like smnist)
            # matching FPTTMinimalRNNAED.forward_chunk which divides logits by n_real_events
            norm = float(params.max_nonzero)
            preds = jnp.argmax(logits / norm, axis=-1)
            epoch_correct += int(jnp.sum(preds == y.astype(int)))
            epoch_total += batch_part_size

    end_time = time.time()
    epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0

    if layer_idx == last_layer and rank == mpi_config.get_last_layer_batch_leader:
        print(f"[MinimalRNN inference] {dataset} accuracy: {epoch_accuracy:.4f}")
        if debug:
            print(f"Execution Time: {end_time - start_time:.2f}s")

    if save and layer_idx == last_layer:
        weights_dict = {"W_out": np.array(weights).tolist()}
        store_training_data(
            params.layer_sizes, params, "inference",
            [-1], [-1], epoch_accuracy,
            end_time - start_time, [0], weights_dict, [], {}, "",
            "RNN",
            extra_fields={
                "cell_type": params.cell_type,
                "fptt_parts": params.fptt_parts,
                "fptt_relu_output": params.fptt_relu_output,
            })

    return epoch_accuracy


# ---------------------------------------------------------------------------
# FPTT training loop for MinimalRNN
# ---------------------------------------------------------------------------
def train_fptt(params: Params, key, total_batches, rnn_weights, weights, empty_neuron_states, opti, trial=None):
    """
    FPTT training loop with MinimalRNN cell on hidden layers.

    Architecture:
    - Input layer (rank 0): sends events to hidden layer
    - Hidden layers (rank 1..last-1): MinimalRNN cell processes events
    - Output layer (rank last): receives h_final from last hidden layer,
      computes logits = h_final @ weights, FPTT loss, sends dL/dh backward

    The output layer does NOT participate in predict_chunk. Instead,
    h_final is sent via MPI tag=3 after each chunk.
    """
    global training_generator
    global validation_generator
    global test_generator

    P = params.fptt_parts
    n_classes = params.layer_sizes[-1]
    H_hidden = params.layer_sizes[last_layer - 1]  # hidden dim of last hidden layer
    alpha = params.fptt_alpha
    beta_cons = params.fptt_beta
    lmbda = params.fptt_lambda
    rho = params.fptt_rho
    clip = params.fptt_clip
    warm_epochs = params.fptt_warm_epochs
    accumulate_logits = params.fptt_accumulate_logits
    avg_logits = params.fptt_avg_logits

    H = params.layer_sizes[layer_idx] if (layer_idx > 0 and layer_idx != last_layer) else 0

    # Initialize optimizer
    if rank == 0:
        print(f"FPTT training with MinimalRNN, P={P}, {opti} optimizer")
    if opti == "adam":
        solver = optax.adam(learning_rate=params.learning_rate)
    elif opti == "sgd":
        solver = optax.sgd(learning_rate=params.learning_rate, momentum=0.9)
    else:
        solver = optax.adam(learning_rate=params.learning_rate)

    # Initialize optimizer states per layer type
    if layer_idx > 0 and layer_idx != last_layer:
        opt_state = solver.init(rnn_weights)
        # Threshold optimizer (same lr as weights, per SEED paper)
        if params.threshold_lr != 0:
            th_solver = optax.adam(learning_rate=params.threshold_lr)
            th_opt_state = th_solver.init(empty_neuron_states.threshold)
        else:
            th_solver = None
            th_opt_state = None
    elif layer_idx == last_layer:
        opt_state = solver.init(weights)
        th_solver = None
        th_opt_state = None
    else:
        opt_state = None
        th_solver = None
        th_opt_state = None

    # Consensus state (hidden layers only)
    if layer_idx > 0 and layer_idx != last_layer:
        sm, lm = init_consensus_state(rnn_weights)
    else:
        sm, lm = None, None

    # Oracle distribution (on output layer only)
    if layer_idx == last_layer:
        oracle = jnp.full((n_classes, P, n_classes), 1.0 / n_classes)
        all_epoch_accuracies = []
        all_validation_accuracies = []
        all_loss = []
    else:
        oracle = None

    all_mean_iterations = []

    mpi4jax.barrier(comm=comm)
    start_time = time.time()

    for epoch in tqdm(range(params.num_epochs), disable=TQDM_DISABLE):
        key, subkey = jax.random.split(key)

        # Reset consensus state at epoch start
        if layer_idx > 0 and layer_idx != last_layer:
            sm, lm = init_consensus_state(rnn_weights)

        if layer_idx == last_layer:
            epoch_correct = 0
            epoch_total = 0
            epoch_loss = []

        epoch_iterations = []
        if layer_idx == 0:
            batch_iterator = None
            if rank == 0:
                batch_iterator = iter(training_generator)
        else:
            batch_iterator = None

        for i in tqdm(range(total_batches[0]), disable=TQDM_DISABLE):
            # Get batch data
            if layer_idx == 0:
                batch_x, batch_y = mpi_config.split_batch(params, batch_iterator, 2)
                mpi_config.send_labels(batch_y, mpi_config.batch_first_and_last_rank[0])

            if layer_idx == last_layer:
                y = mpi_config.recv_labels()

            # Initialize h_states for this batch (hidden layers only)
            if layer_idx > 0 and layer_idx != last_layer:
                h_states = jnp.zeros((batch_part_size, H))
            else:
                h_states = jnp.zeros((batch_part_size, 1))  # dummy for input/output layers

            # Compute chunk boundaries
            T = params.max_nonzero
            step = T // P
            _P = P if P * step >= T else P + 1

            batch_iters = jnp.zeros(batch_part_size)  # accumulated across chunks (input/hidden layers)
            output_iters = 0  # scalar counter for output layer events received

            # Accumulated logits across chunks (output layer only, if enabled)
            if layer_idx == last_layer and accumulate_logits:
                accumulated_logits = jnp.zeros((batch_part_size, n_classes))

            for p in range(_P):
                start_t = p * step
                end_t = min(start_t + step, T)
                if start_t >= T:
                    break
                chunk_len = end_t - start_t

                if layer_idx == last_layer:
                    # --- Output layer: receive activations from last hidden layer ---
                    if params.fptt_relu_output:
                        # Receive sparse relu events via while_loop (one per sample)
                        chunk_logits, sparse_inputs, n_events = output_layer_recv_events_batch(
                            weights, batch_part_size, n_classes, H_hidden)
                        output_iters += int(n_events)
                    else:
                        # Receive relu_sum = Σ_t relu(h_t) from last hidden layer (tag=3)
                        relu_sum = recv(jnp.zeros((batch_part_size, H_hidden)),
                                        source=rank - processes_per_layer_global, tag=3, comm=comm)
                        output_iters += H_hidden
                        chunk_logits = jnp.dot(relu_sum, weights)  # (B, C)

                    if accumulate_logits:
                        accumulated_logits = accumulated_logits + chunk_logits
                        logits = accumulated_logits / (p + 1) if avg_logits else accumulated_logits
                    else:
                        logits = chunk_logits

                    # Compute FPTT loss
                    beta_p = (p + 1) / _P
                    if p < _P - 1:
                        if epoch < warm_epochs:
                            oracle_prob = jnp.full((batch_part_size, n_classes), 1.0 / n_classes)
                        else:
                            oracle_prob = oracle[y.astype(int), p]
                    else:
                        oracle_prob = jax.nn.one_hot(y, n_classes)

                    loss, d_logits = fptt_loss_and_grad(
                        logits, y.astype(int), oracle_prob, beta_p, n_classes)

                    # Chain rule through logit accumulation
                    if accumulate_logits and avg_logits:
                        d_logits = d_logits / (p + 1)

                    if params.fptt_relu_output:
                        # dL/dW[i] = sum_b d_logits[b] * sparse_inputs[b, i]
                        # sparse_inputs: (B, H_hidden), d_logits: (B, C)
                        weight_grad = jnp.dot(sparse_inputs.T, d_logits)  # (H_hidden, C)
                    else:
                        # dL/dW = relu_sum.T @ d_logits
                        weight_grad = jnp.dot(relu_sum.T, d_logits)  # (H_hidden, C)

                    weight_grad = mpi_config.combine_batch_avg(jnp.expand_dims(weight_grad, axis=0))

                    # Compute gradient to send backward: dL/dh = d_logits @ W.T  (B, H_hidden)
                    dL_dh = jnp.dot(d_logits, weights.T)
                    mpi_config.backward_send(dL_dh)

                    # Update output weights
                    updates, opt_state = solver.update(weight_grad, opt_state, weights)
                    weights = optax.apply_updates(weights, updates)

                    # Update oracle (use accumulated logits for probability estimate)
                    if p < _P - 1:
                        probs = jax.nn.softmax(logits, axis=-1)
                        oracle = vectorized_oracle_update(
                            oracle, probs, y.astype(int), p, n_classes)

                    epoch_loss.append(float(loss))

                    # Track accuracy on last chunk of each batch
                    if p == _P - 1:
                        preds = jnp.argmax(logits, axis=-1)
                        batch_correct = jnp.sum(preds == y.astype(int))
                        epoch_correct += batch_correct
                        epoch_total += batch_part_size

                elif layer_idx > 0:
                    # --- Hidden layer: run MinimalRNN forward, send h_final, receive gradient ---
                    # Prepare chunk data
                    chunk_data = jnp.zeros((batch_part_size, chunk_len, 2))  # dummy

                    # Forward pass
                    all_h_finals, all_iters, all_ns, all_relu_sums = predict_chunk(
                        params, subkey, rnn_weights, empty_neuron_states,
                        chunk_data, h_states, grad=True)

                    batch_iters = batch_iters + all_iters  # accumulate across chunks

                    # Only the last hidden layer sends relu_sum to the output layer (tag=3).
                    # relu_sum = Σ_t relu(h_t) over the chunk; output computes relu_sum @ W_out.
                    # When fptt_relu_output=True, events were already sent during forward pass.
                    if layer_idx == last_layer - 1 and not params.fptt_relu_output:
                        send(all_relu_sums, dest=rank + processes_per_layer_global, tag=3, comm=comm)

                    # Receive gradient from next layer (output layer or next hidden layer).
                    # The next layer sends dL/d(this layer's h_state) with shape (B, H_this).
                    next_grad = mpi_config.backward_recv()  # Shape: (B, layer_size)

                    # MinimalRNN backward.
                    # For relu output: next_grad = d_logits @ W_out.T is injected at every
                    # timestep where a neuron fired (per-step gradient), not just at h_final.
                    # The final-state dL_dh starts at zero — all gradient flows through the
                    # per-step injection in bptt_step.
                    if params.fptt_relu_output and layer_idx == last_layer - 1:
                        dL_dh_final = jnp.zeros_like(next_grad)
                        grads, dL_dinput, grad_th = minimalrnn_back_prop(
                            params, rnn_weights, all_ns, dL_dh_final, dL_dh_output=next_grad)
                    else:
                        grads, dL_dinput, grad_th = minimalrnn_back_prop(
                            params, rnn_weights, all_ns, next_grad)

                    # If there is a previous hidden layer, send gradient backward to it (tag=2)
                    if layer_idx > 1:
                        mpi_config.backward_send(dL_dinput)

                    # Average across processes in same layer
                    # combine_batch_avg expects a leading "process" dimension, so wrap each grad
                    grads = {k: mpi_config.combine_batch_avg(jnp.expand_dims(v, axis=0))
                             for k, v in grads.items()}

                    # Consensus regularizer
                    reg = consensus_reg_grad(rnn_weights, sm, lm, alpha, lmbda, rho)
                    total_grads = jax.tree.map(lambda g, r: g + r, grads, reg)

                    # Clip gradients
                    if clip > 0:
                        total_grad_flat, tree_def = jax.tree.flatten(total_grads)
                        global_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in total_grad_flat))
                        scale = jnp.minimum(1.0, clip / (global_norm + 1e-6))
                        total_grad_flat = [g * scale for g in total_grad_flat]
                        total_grads = jax.tree.unflatten(tree_def, total_grad_flat)

                    # Optimizer step (weights)
                    updates, opt_state = solver.update(total_grads, opt_state, rnn_weights)
                    rnn_weights = optax.apply_updates(rnn_weights, updates)

                    # Threshold update (per-neuron Vth, SEED paper)
                    if th_solver is not None:
                        grad_th_avg = mpi_config.combine_batch_avg(jnp.expand_dims(grad_th, axis=0))
                        th_updates, th_opt_state = th_solver.update(
                            grad_th_avg, th_opt_state, empty_neuron_states.threshold)
                        new_th = optax.apply_updates(empty_neuron_states.threshold, th_updates)
                        empty_neuron_states = empty_neuron_states.replace(threshold=new_th)

                    # Consensus post-update
                    sm, lm = post_optimizer_update(rnn_weights, sm, lm, alpha, beta_cons)

                    # Carry h_states to next chunk
                    h_states = all_h_finals  # (B, H)

                else:
                    # --- Input layer: run forward pass (sends events to hidden) ---
                    chunk_data = jnp.array(batch_x[:, start_t:end_t, :])
                    _, all_iters, _, _ = predict_chunk(
                        params, subkey, rnn_weights, empty_neuron_states,
                        chunk_data, h_states, grad=False)
                    batch_iters = batch_iters + all_iters  # accumulate across chunks

                mpi4jax.barrier(comm=comm)

            if layer_idx == last_layer:
                # output_iters = H_hidden * number_of_chunks events received this batch
                if output_iters > 0:
                    epoch_iterations.append(jnp.array([float(output_iters) / batch_part_size]))
            else:
                valid_iters = batch_iters[batch_iters > 0]
                if valid_iters.size > 0:
                    epoch_iterations.append(valid_iters)

        # End of epoch: compute metrics
        if epoch_iterations:
            epoch_iterations_arr = jnp.concatenate(epoch_iterations)
            mean_iter = jnp.mean(epoch_iterations_arr) if epoch_iterations_arr.size > 0 else 0.0
        else:
            mean_iter = 0.0
        all_mean_iterations.append(mean_iter)

        if layer_idx == last_layer:
            mean_loss = jnp.mean(jnp.array(epoch_loss))
            all_loss.append(mean_loss)
            epoch_accuracy = epoch_correct / jnp.maximum(epoch_total, 1)
            all_epoch_accuracies.append(float(epoch_accuracy))

        # --- Validation loop ---
        mpi4jax.barrier(comm=comm)
        val_correct = 0
        val_total = 0
        if layer_idx == 0:
            val_batch_iter = None
            if rank == 0:
                val_batch_iter = iter(validation_generator)

        for vi in range(total_batches[1]):
            if layer_idx == 0:
                batch_x, batch_y = mpi_config.split_batch(params, val_batch_iter, 2)
                mpi_config.send_labels(batch_y, mpi_config.batch_first_and_last_rank[0])
            if layer_idx == last_layer:
                y = mpi_config.recv_labels()

            # Initialize hidden state
            if layer_idx > 0 and layer_idx != last_layer:
                val_h = jnp.zeros((batch_part_size, H))
            else:
                val_h = jnp.zeros((batch_part_size, 1))

            T = params.max_nonzero
            step = T // P
            _P = P if P * step >= T else P + 1

            if layer_idx == last_layer and accumulate_logits:
                val_accum_logits = jnp.zeros((batch_part_size, n_classes))

            for vp in range(_P):
                start_t = vp * step
                end_t = min(start_t + step, T)
                if start_t >= T:
                    break
                chunk_len = end_t - start_t

                if layer_idx == last_layer:
                    if params.fptt_relu_output:
                        chunk_logits, _, _ = output_layer_recv_events_batch(
                            weights, batch_part_size, n_classes, H_hidden)
                    else:
                        relu_sum_val = recv(jnp.zeros((batch_part_size, H_hidden)),
                                            source=rank - processes_per_layer_global, tag=3, comm=comm)
                        chunk_logits = jnp.dot(relu_sum_val, weights)
                    if accumulate_logits:
                        val_accum_logits = val_accum_logits + chunk_logits
                        val_logits = val_accum_logits / (vp + 1) if avg_logits else val_accum_logits
                    else:
                        val_logits = chunk_logits

                elif layer_idx > 0:
                    # Hidden layer forward only (no grad)
                    val_chunk = jnp.zeros((batch_part_size, chunk_len, 2))
                    all_h_final, all_iters, _, val_relu_sums = predict_chunk(
                        params, key, rnn_weights, empty_neuron_states,
                        val_chunk, val_h, grad=False)
                    val_h = all_h_final
                    if layer_idx == last_layer - 1 and not params.fptt_relu_output:
                        send(val_relu_sums, dest=rank + processes_per_layer_global, tag=3, comm=comm)

                else:
                    # Input layer: send chunk events
                    val_chunk = batch_x[:, start_t:end_t, :]
                    predict_chunk(params, key, rnn_weights, empty_neuron_states,
                                  val_chunk, val_h, grad=False)

            if layer_idx == last_layer:
                preds = jnp.argmax(val_logits, axis=-1)
                val_correct += jnp.sum(preds == y.astype(int))
                val_total += batch_part_size

        if layer_idx == last_layer:
            val_accuracy = val_correct / jnp.maximum(val_total, 1)
            all_validation_accuracies.append(float(val_accuracy))

            if mpi_config.get_last_layer_batch_leader:
                jax.debug.print(
                    "Epoch {} , Training Accuracy: {:.2f}%, Validation Accuracy: {:.2f}%, mean loss: {}",
                    epoch, epoch_accuracy * 100, val_accuracy * 100, mean_loss)

    elapsed = time.time() - start_time
    if rank == 0:
        print(f"FPTT training completed in {elapsed:.1f}s")

    # --- Save results (mirroring async_RNN.py pattern) ---
    mpi4jax.barrier(comm=comm)

    # Gather MinimalRNN weights from hidden layer(s) to output layer
    weights_dict = {}

    if layer_idx > 0 and layer_idx != last_layer:
        # Hidden layer: send rnn_weights to output layer
        for wname in ['W_phi', 'b_phi', 'W_gate', 'b_gate']:
            send(rnn_weights[wname], dest=last_layer * processes_per_layer_global, tag=5, comm=comm)
    elif layer_idx == last_layer:
        # Output layer: receive rnn_weights from hidden layer(s)
        for hidden_rank_layer in range(1, last_layer):
            hidden_rank = hidden_rank_layer * processes_per_layer_global
            input_dim = params.layer_sizes[hidden_rank_layer - 1]
            H = params.layer_sizes[hidden_rank_layer]
            recv_W_phi = recv(jnp.zeros((input_dim, H)), source=hidden_rank, tag=5, comm=comm)
            recv_b_phi = recv(jnp.zeros(H), source=hidden_rank, tag=5, comm=comm)
            recv_W_gate = recv(jnp.zeros((2 * H, H)), source=hidden_rank, tag=5, comm=comm)
            recv_b_gate = recv(jnp.zeros(H), source=hidden_rank, tag=5, comm=comm)
            weights_dict[f"layer_{hidden_rank_layer}_W_phi"] = recv_W_phi.tolist()
            weights_dict[f"layer_{hidden_rank_layer}_b_phi"] = recv_b_phi.tolist()
            weights_dict[f"layer_{hidden_rank_layer}_W_gate"] = recv_W_gate.tolist()
            weights_dict[f"layer_{hidden_rank_layer}_b_gate"] = recv_b_gate.tolist()
        # Add output layer weights
        weights_dict[f"layer_{last_layer}"] = weights.tolist()

    # Gather per-epoch mean iterations from each layer to the output layer
    # all_mean_iterations is a list of per-epoch scalars on each rank
    n_epochs = params.num_epochs
    all_iteration_mean = []
    if layer_idx != last_layer:
        # Pad to n_epochs in case fewer epochs ran
        iter_arr = jnp.array(all_mean_iterations + [0.0] * (n_epochs - len(all_mean_iterations)))
        send(iter_arr[:n_epochs], dest=last_layer * processes_per_layer_global, tag=6, comm=comm)
    elif layer_idx == last_layer:
        for i in range(last_layer):
            it_arr = recv(jnp.zeros(n_epochs), source=i * processes_per_layer_global, tag=6, comm=comm)
            all_iteration_mean.append(it_arr.tolist())
        # Append output layer's own iterations (now tracked)
        iter_arr = jnp.array(all_mean_iterations + [0.0] * (n_epochs - len(all_mean_iterations)))
        all_iteration_mean.append(iter_arr[:n_epochs].tolist())

    # Run test evaluation
    test_accuracy = -1.0
    mpi4jax.barrier(comm=comm)
    test_iter = iter(test_generator) if layer_idx == 0 and rank == 0 else None
    if layer_idx == 0:
        for i in range(total_batches[2]):
            batch_x, batch_y = mpi_config.split_batch(params, test_iter, 2)
            mpi_config.send_labels(batch_y, mpi_config.batch_first_and_last_rank[0])
            T = params.max_nonzero
            step = T // P
            _P_test = P if P * step >= T else P + 1
            for p in range(_P_test):
                start_t = p * step
                end_t = min(start_t + step, T)
                if start_t >= T:
                    break
                chunk_data = jnp.array(batch_x[:, start_t:end_t, :])
                predict_chunk(params, key, rnn_weights, empty_neuron_states,
                              chunk_data, jnp.zeros((batch_part_size, 1)), grad=False)
            mpi4jax.barrier(comm=comm)

    elif layer_idx > 0 and layer_idx != last_layer:
        h_states_test = jnp.zeros((batch_part_size, H))
        for i in range(total_batches[2]):
            T = params.max_nonzero
            step = T // P
            _P_test = P if P * step >= T else P + 1
            for p in range(_P_test):
                start_t = p * step
                end_t = min(start_t + step, T)
                if start_t >= T:
                    break
                chunk_data = jnp.zeros((batch_part_size, end_t - start_t, 2))
                all_h, _, _, test_relu_sums = predict_chunk(params, key, rnn_weights, empty_neuron_states,
                                                            chunk_data, h_states_test, grad=False)
                h_states_test = all_h
                if layer_idx == last_layer - 1 and not params.fptt_relu_output:
                    send(test_relu_sums, dest=rank + processes_per_layer_global, tag=3, comm=comm)
            h_states_test = jnp.zeros((batch_part_size, H))
            mpi4jax.barrier(comm=comm)

    elif layer_idx == last_layer:
        test_correct = 0
        test_total = 0
        for i in range(total_batches[2]):
            y = mpi_config.recv_labels()
            T = params.max_nonzero
            step = T // P
            _P_test = P if P * step >= T else P + 1
            test_accum_logits = jnp.zeros((batch_part_size, n_classes)) if accumulate_logits else None
            for p in range(_P_test):
                start_t = p * step
                end_t = min(start_t + step, T)
                if start_t >= T:
                    break
                if params.fptt_relu_output:
                    chunk_logits, _, _ = output_layer_recv_events_batch(
                        weights, batch_part_size, n_classes, H_hidden)
                else:
                    relu_sum_test = recv(jnp.zeros((batch_part_size, H_hidden)),
                                        source=rank - processes_per_layer_global, tag=3, comm=comm)
                    chunk_logits = jnp.dot(relu_sum_test, weights)
                if accumulate_logits:
                    test_accum_logits = test_accum_logits + chunk_logits
                    test_logits = test_accum_logits / (p + 1) if avg_logits else test_accum_logits
                else:
                    test_logits = chunk_logits
            preds = jnp.argmax(test_logits, axis=-1)
            test_correct += jnp.sum(preds == y.astype(int))
            test_total += batch_part_size
            mpi4jax.barrier(comm=comm)
        test_accuracy = float(test_correct / jnp.maximum(test_total, 1))

    # Store results to JSON (output layer only)
    mpi4jax.barrier(comm=comm)
    if layer_idx == last_layer:
        print(f"\n{'='*60}")
        print(f"Training complete — {params.num_epochs} epochs in {elapsed:.1f}s")
        print(f"  Final Train Accuracy:  {all_epoch_accuracies[-1]*100:.2f}%")
        print(f"  Final Val Accuracy:    {all_validation_accuracies[-1]*100:.2f}%")
        print(f"  Test Accuracy:         {test_accuracy*100:.2f}%")
        print(f"  Best Val Accuracy:     {max(all_validation_accuracies)*100:.2f}% (epoch {all_validation_accuracies.index(max(all_validation_accuracies))})")
        print(f"{'='*60}\n")
        result_path_str = store_training_data(
            size,
            params,
            "train",
            all_epoch_accuracies,
            all_validation_accuracies,
            test_accuracy,
            elapsed,
            all_iteration_mean,
            weights_dict,
            [float(l) for l in all_loss],
            {},  # thresholds_dict (not used in MinimalRNN)
            opti,
            "RNN",
            None,  # all_history
            total_batches[0],
            extra_fields={
                "cell_type": params.cell_type,
                "use_tanh": params.use_tanh,
                "fptt_parts": params.fptt_parts,
                "fptt_alpha": params.fptt_alpha,
                "fptt_beta": params.fptt_beta,
                "fptt_lambda": params.fptt_lambda,
                "fptt_rho": params.fptt_rho,
                "fptt_clip": params.fptt_clip,
                "fptt_warm_epochs": params.fptt_warm_epochs,
                "fptt_accumulate_logits": params.fptt_accumulate_logits,
                "fptt_avg_logits": params.fptt_avg_logits,
                "fptt_relu_output": params.fptt_relu_output,
            })
        print(f"Results saved to {result_path_str}")

    mpi4jax.barrier(comm=comm)
    return rnn_weights, weights


#endregion


def _pad_to_layers(seq, layer_sizes, default):
    """Extend or truncate seq to len(layer_sizes), filling with default."""
    n = len(layer_sizes)
    lst = list(seq) if seq is not None else []
    if len(lst) < n:
        lst += [default] * (n - len(lst))
    return lst[:n]


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

    print(f"Rank {rank} loaded config: {config["recurrence"]}")

    # Extract configuration parameters
    dataset = config['dataset']
    layer_sizes = tuple(config['layer_sizes'])
    batch_size = config['batch_size']

    restrict = config['restrict']    
    init_thresholds = config['init_thresholds']
    load_file = config['load_file']
    best = config['best']
    rerun = config['rerun']

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

    if batch_size % processes_per_layer_global != 0:
        print(f"Error: one batch ({batch_size}) must be divisible by the number of processes per layer ({processes_per_layer_global})")
        sys.exit(1)
    
    # for f_nb in [2, 4, 8, 16, 32]: # Loop for multiple experiments
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
                    sequential = False
                    permuted = False
                    if layer_sizes[0] == 14*14:
                        downsample = True
                    if dataset == "smnist" or dataset == "psmnist":
                        sequential = True
                        if dataset == "psmnist":
                            permuted = True

                    loader = partial(mnist_loader_manual, 
                                        sequential=sequential, 
                                        permuted=permuted)
                case "shd":
                    loader = torch_SHD_loader
                case "neural_decoding" | "primate_reaching":
                    from dataset_helpers.primate_reaching_helper import torch_primate_reaching_loader
                    loader = partial(
                        torch_primate_reaching_loader,
                        filename=config.get('filename', 'indy_20160622_01.mat'),
                        window=config.get('window', 50),
                        collapse_units=config.get('collapse_units', True),
                        preserve_exact_times=config.get('preserve_exact_times', False),
                        truncate=True,
                    )
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
            restrict=restrict,
            # firing_nb=f_nb,
            firing_nb=config['firing_nb'],
            sync_rate=config['sync_rate'],
            max_nonzero=max_nonzero,
            shuffle_input=config['shuffle_input'],
            threshold_lr=config['threshold_lr'],
            sparsity_impact=tuple(_pad_to_layers(config['sparsity_impact'], layer_sizes, 0.0)),
            w_reg=config['w_reg'],
            rerun="",
            top_weights=config['top_weights'],
            history_size=config['history_size'],
            recurrence=tuple(_pad_to_layers(config['recurrence'], layer_sizes, None)),
            use_bias=config['use_bias'],
            use_tanh=config['use_tanh'],
            exact_rtrl=config.get('exact_rtrl', False),
            cell_type=config.get('cell_type', 'aed'),
            fptt_parts=config.get('fptt_parts', 1),
            fptt_alpha=config.get('fptt_alpha', 0.1),
            fptt_beta=config.get('fptt_beta', 0.5),
            fptt_lambda=config.get('fptt_lambda', 2.0),
            fptt_rho=config.get('fptt_rho', 0.0),
            fptt_clip=config.get('fptt_clip', 1.0),
            fptt_warm_epochs=config.get('fptt_warm_epochs', 1),
            fptt_accumulate_logits=config.get('fptt_accumulate_logits', True),
            fptt_avg_logits=config.get('fptt_avg_logits', False),
            fptt_relu_output=config.get('fptt_relu_output', False),
        )
        if trial is not None:
            params = dataclasses.replace(trial_params, max_nonzero=max_nonzero)

        print("recurrence in params:", params.recurrence)
        if rerun is not None:                  
            override_list = config.get('override_params', None)
            params, weights, thresholds = rerun_init(
                rerun, 
                mpi_config, 
                params, 
                override_params=override_list
            )
        print("rank recurrence in params:", rank, params.recurrence)

        if rank == 0:
            print(f"Number of training batches: {total_train_batches}, validation batches: {total_val_batches}, test batches: {total_test_batches}")
            print(params)
        
        # Instantiate the neuron states with the correct shapes and initial values
        layer_key = jax.random.fold_in(key, layer_idx)
        # sync_rate_vector = jax.random.randint(layer_key, shape=(layer_sizes[layer_idx],), minval=1, maxval=params.sync_rate)
        sync_rate_vector = jnp.full(shape=(layer_sizes[layer_idx],), fill_value=params.sync_rate)

        key, subkey = jax.random.split(key) 
        # Never allocate exact RTRL traces for input/output layers (even if recurrence is enabled)
        use_exact_rtrl = params.exact_rtrl and (layer_idx != 0) and (layer_idx != last_layer)
        empty_neuron_states = NeuronStates( values=jnp.zeros(layer_sizes[layer_idx]),
                                            bias=jnp.zeros(layer_sizes[layer_idx]),
                                            thresholds=thresholds,
                                            input_residuals=np.zeros((layer_sizes[layer_idx-1],)),
                                            output_residuals=np.zeros((layer_sizes[layer_idx],)),
                                            input_order=jnp.full((layer_sizes[layer_idx-1],), -1, dtype=int), 
                                            input_activity=jnp.full((layer_sizes[layer_idx-1],), 0, dtype=int),
                                            layer_activity=jnp.zeros((layer_sizes[layer_idx],), dtype=int),
                                            output_activity=jnp.zeros((layer_sizes[layer_idx-1], layer_sizes[layer_idx])),
                                            last_sent_iteration=-1,
                                            input_vector=jnp.zeros((layer_sizes[layer_idx-1]), dtype=int),
                                            output_vector=jnp.zeros((layer_sizes[layer_idx]), dtype=int),
                                            sync_rate_vector=sync_rate_vector,
                                            recurrent_weight=init_recurrent(subkey, layer_sizes[layer_idx], gain=0.5),
                                            values_history=jnp.zeros((params.history_size, layer_sizes[layer_idx])),
                                            history_index=jnp.array(0, dtype=jnp.int32),
                                            rnn_running_sum=jnp.zeros((layer_sizes[layer_idx-1], layer_sizes[layer_idx])),
                                            rnn_total_sum=jnp.zeros((layer_sizes[layer_idx-1], layer_sizes[layer_idx])),
                                            rnn_running_product=jnp.zeros((layer_sizes[layer_idx], layer_sizes[layer_idx])),
                                            rnn_total_product_sum=jnp.zeros((layer_sizes[layer_idx], layer_sizes[layer_idx])),
                                            bias_running_sum=jnp.zeros(layer_sizes[layer_idx]),
                                            bias_total_sum=jnp.zeros(layer_sizes[layer_idx]),
                                            prev_active=jnp.zeros((layer_sizes[layer_idx],), dtype=int),
                                            prev_activated_output=jnp.zeros((layer_sizes[layer_idx],)),
                                            prev_tanh_deriv=jnp.ones((layer_sizes[layer_idx],)),
                                            **(dict(
                                                exact_hh_running=jnp.zeros((layer_sizes[layer_idx], layer_sizes[layer_idx], layer_sizes[layer_idx])),
                                                exact_hh_total=jnp.zeros((layer_sizes[layer_idx], layer_sizes[layer_idx], layer_sizes[layer_idx])),
                                                exact_bias_running=jnp.zeros((layer_sizes[layer_idx], layer_sizes[layer_idx])),
                                                exact_bias_total=jnp.zeros((layer_sizes[layer_idx], layer_sizes[layer_idx])),
                                            ) if use_exact_rtrl else {}),
                                            )
        # print(f"rank {rank} sync rates: {sync_rate_vector}")
        total_batches = (total_train_batches, total_val_batches, total_test_batches)
        print(empty_neuron_states.recurrent_weight.shape)
        mode = config['mode']

        if params.cell_type == 'minimalrnn':
            # --- MinimalRNN + FPTT path ---
            # Hidden layers use MinimalRNN cell, output layer uses standard weights
            n_classes = layer_sizes[-1]
            max_chunk_len = (params.max_nonzero // max(params.fptt_parts, 1)) + 1

            pt_weights_path = config.get('pt_weights_path', None)

            # Initialize or load MinimalRNN weights (hidden layers only, not output/input)
            key, wkey = jax.random.split(key)
            if layer_idx > 0 and layer_idx != last_layer:
                H = layer_sizes[layer_idx]
                input_dim = layer_sizes[layer_idx - 1]
                if pt_weights_path:
                    rnn_weights = load_minimalrnn_aed_weights(pt_weights_path, layer_idx, layer_sizes)
                    print(f"Rank {rank}: loaded MinimalRNN weights from {pt_weights_path} for layer {layer_idx}")
                else:
                    rnn_weights = init_minimalrnn_weights(wkey, input_dim, H)
            elif layer_idx == last_layer and pt_weights_path:
                rnn_weights = {}
                H = layer_sizes[layer_idx]
                weights = load_minimalrnn_aed_output_weights(pt_weights_path, layer_sizes)
                print(f"Rank {rank}: loaded W_out from {pt_weights_path}")
            else:
                rnn_weights = {}
                H = layer_sizes[layer_idx]

            if layer_idx > 0 and layer_idx != last_layer:
                # Hidden layer: MinimalRNN NeuronStates with trace buffers
                sync_rate_vector_mr = jnp.full(shape=(H,), fill_value=params.sync_rate)
                # Thresholds at zero (frozen) — isolating b_gate effect
                key, th_key = jax.random.split(key)
                init_th = jnp.zeros(H)
                minimalrnn_ns_fields = dict(
                    values=jnp.zeros(H),
                    bias=jnp.zeros(H),
                    thresholds=init_th,
                    h_state=jnp.zeros(H),
                    input_residuals=jnp.zeros(layer_sizes[layer_idx - 1]),
                    output_residuals=jnp.zeros(H),
                    input_order=jnp.full((layer_sizes[layer_idx - 1],), -1, dtype=int),
                    input_activity=jnp.zeros(layer_sizes[layer_idx - 1], dtype=int),
                    layer_activity=jnp.zeros(H, dtype=int),
                    output_activity=jnp.zeros((layer_sizes[layer_idx - 1], H)),
                    last_sent_iteration=-1,
                    input_vector=jnp.zeros(layer_sizes[layer_idx - 1], dtype=int),
                    output_vector=jnp.zeros(H, dtype=int),
                    sync_rate_vector=sync_rate_vector_mr,
                    prev_activated_output=jnp.zeros(H),
                    recurrent_weight=jnp.zeros((1, 1)),  # unused for MinimalRNN, kept for pytree compatibility
                    values_history=jnp.zeros((max(params.history_size, 1), H)),
                    history_index=jnp.array(0, dtype=jnp.int32),
                    # Per-neuron threshold (SEED MinimalRNN), same as thresholds above
                    threshold=init_th,
                    # Trace buffers for backward pass
                    all_z=jnp.zeros((max_chunk_len, H)),
                    all_u=jnp.zeros((max_chunk_len, H)),
                    all_h_prev=jnp.zeros((max_chunk_len, H)),   # stores y_prev = relu(h_prev) at th=0
                    all_h_new=jnp.zeros((max_chunk_len, H)),    # h_new after state update (for s_t recomputation)
                    all_neuron_idx=jnp.zeros(max_chunk_len, dtype=int),
                    all_value=jnp.zeros(max_chunk_len),
                    trace_index=jnp.array(0, dtype=jnp.int32),
                )
                empty_neuron_states_mr = NeuronStates(**minimalrnn_ns_fields)
            else:
                # Input/output layer: use the standard empty_neuron_states already built
                empty_neuron_states_mr = empty_neuron_states

            n_params = sum(w.size for w in jax.tree.leaves(rnn_weights)) if rnn_weights else 0
            if layer_idx > 0 and layer_idx != last_layer:
                print(f"Rank {rank}: MinimalRNN cell, H={H}, input_dim={layer_sizes[layer_idx-1]}, rnn_params={n_params}")
            if rank == 0:
                print(f"MinimalRNN cell_type, H={layer_sizes[1]}, FPTT parts={params.fptt_parts}")

            if mode == 'training':
                train_fptt(params, key, total_batches, rnn_weights, weights,
                           empty_neuron_states_mr, config['optimizer'], trial)
            elif mode == 'inference':
                batch_predict_minimalrnn(params, key, total_batches, rnn_weights, weights,
                                         empty_neuron_states_mr, dataset="test", save=True, debug=True)
            else:
                print(f"Unknown mode for MinimalRNN, choose 'training' or 'inference', got {mode}")
        else:
            # --- Standard AED path ---
            if mode == 'inference':
                batch_predict(params, key, total_batches, weights, empty_neuron_states, "test", save=True, debug=True)
            elif mode == 'training':
                val_acc, result_path = train(params, key, total_batches, weights, empty_neuron_states, config['optimizer'], trial)
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


    # config = load_config_with_defaults(args.config)
    # config = parse_unknown_args_and_overrides_config(unknown, config)
    # print(config['recurrence'])
    main(random_seed, key, rank, size, comm, config_path=args.config, data_dir=args.data_dir)
# JAX_PLATFORMS=cpu mpirun -n 4 python async_MLP.py --config "MLP_config.yaml"
