import os
os.environ["JAX_PLATFORMS"] = "cpu"

from mpi4py import MPI
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
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna

import mpi4jax
from mpi4jax import send, recv, bcast

from dataset_helpers.mnist_helper import mnist_loader_manual
from dataset_helpers.shd_helper import torch_SHD_loader
from dataset_helpers.nmnist_helper import torch_nmnist_loader
from dataset_helpers.dvs_helper import torch_DVSGesture_loader
from dataset_helpers.iris_species_helper import torch_iris_loader
from dataset_helpers.network_helper import one_hot_encode

from other_helpers.helpers import Params, NeuronStates
from other_helpers.helpers import accuracy, store_training_data, rerun_init, store_data_to_json
from other_helpers.helpers import activation_func, keep_top_k, output_vector_to_event
from other_helpers.helpers import update_history, process_history, load_config_with_defaults
from other_helpers.backpropagation import back_prop
from other_helpers.loss_functions import loss_bpp, mean_loss
from other_helpers.MPI_helpers import (MPIConfig, combine_batch_avg, gather_batch,
                                        split_batch, l2_weight_regularization)

jax.config.update("jax_debug_nans", True)

TQDM_DISABLE        = False
STORE_EACH_EPOCH    = True
BUFFER_SIZE         = 0

LIF_TAU_M         = 100.0   # overwritten from YAML key `tau_m` in main()
SCHEDULING_POLICY = "ms"   # overwritten from YAML key `scheduling_policy` in main()
MS_NOISE_SCALE    = 1e-6   # overwritten from YAML key `momentum_noise_scale` in main()

comm               = None
rank               = None
size               = None
layer_idx          = None
process_per_layer  = None
last_layer         = None
batch_part         = None
mpi_config         = None

training_generator   = None
validation_generator = None
test_generator       = None


def lif_decay(values, last_update_iteration, iteration):
    current_time  = jnp.asarray(iteration, dtype=jnp.float32)
    last_time     = last_update_iteration.astype(jnp.float32)
    dt            = jnp.maximum(current_time - last_time, 0.0)
    decay_factor  = jnp.exp(-dt / LIF_TAU_M)
    return values * decay_factor


def lif_input_current(params, neuron_idx, layer_input, weights):
    valid_event     = neuron_idx >= 0
    safe_neuron_idx = jnp.where(valid_event, neuron_idx, 0)
    filtered_weights = keep_top_k(weights[safe_neuron_idx],
                                   params.top_weights, apply_abs=True)
    input_current = jax.lax.cond(
        valid_event,
        lambda _: layer_input * filtered_weights,
        lambda _: jnp.zeros_like(filtered_weights),
        None)
    return input_current, valid_event, safe_neuron_idx


def lif_select_spikes(params, spikes):
    return keep_top_k(spikes, params.firing_nb)


def lif_neuron_forward(params, old_values, u_decayed, input_current,
                        thresholds, valid_event, grad=False):
    u_pre = u_decayed + input_current
    hard_spikes      = (u_pre > thresholds).astype(jnp.float32)
    surrogate_spikes = activation_func(thresholds, u_pre)
    if grad:
        raw_spikes = surrogate_spikes + jax.lax.stop_gradient(
            hard_spikes - surrogate_spikes)
    else:
        raw_spikes = hard_spikes

    raw_spikes = jax.lax.cond(
        valid_event,
        lambda _: raw_spikes,
        lambda _: jnp.zeros_like(raw_spikes),
        None)

    selected_spikes      = lif_select_spikes(params, raw_spikes)
    selected_spikes      = (selected_spikes > 0).astype(jnp.float32)
    selected_hard_spikes = lif_select_spikes(params, hard_spikes)
    u_new = jnp.where(selected_hard_spikes > 0, 0.0, u_pre)
    u_new = jax.lax.cond(
        valid_event,
        lambda _: u_new,
        lambda _: old_values,
        None)
    return selected_spikes, u_new, u_pre


def lif_network_layers_forward(params, key, selected_spikes):
    valid_elements   = jnp.count_nonzero(selected_spikes)
    processed_output = output_vector_to_event(
        key, selected_spikes, params, params.layer_sizes[layer_idx])
    return valid_elements, processed_output


def apply_random_scheduling(key, activated_output, valid_elements):
    new_key, subkey = jax.random.split(key)
    n_rows  = activated_output.shape[0]
    perm    = jax.random.permutation(subkey, n_rows)
    return activated_output[perm], new_key


def apply_momentum_scheduling(key, processed_output, u_pre):
    """
    Momentum Scheduling (MS) — TMLR paper §3.1.2.
    Reorder outgoing spike events so neurons with highest pre-spike membrane
    potential (u_pre) are sent first. Ties broken by λ_MS Gaussian noise.
    processed_output: (firing_nb, 2) rows of [neuron_idx, spike_val]
    u_pre:            (n_neurons,)  pre-spike membrane potentials
    """
    _, subkey = jax.random.split(key)
    noise  = jax.random.normal(subkey, u_pre.shape) * MS_NOISE_SCALE
    scores = u_pre + noise                              # (n_neurons,)

    # Map each output row to its neuron's score; padding rows (idx<0) get -inf
    neuron_idx_col = processed_output[:, 0]
    safe_idx       = jnp.where(neuron_idx_col >= 0,
                                neuron_idx_col.astype(jnp.int32), 0)
    row_scores     = scores[safe_idx]
    row_scores     = jnp.where(neuron_idx_col >= 0, row_scores, scores.min() - 1.0)

    sorted_rows = jnp.argsort(row_scores, descending=True)
    return processed_output[sorted_rows]


@partial(jax.jit, static_argnames=['params', 'grad'])
def layer_computation(params, key, neuron_idx, layer_input, weights,
                       neuron_states, iteration=0, grad=False):
    safe_idx_readout   = jnp.where(neuron_idx >= 0, neuron_idx, 0)
    filtered_w_readout = keep_top_k(weights[safe_idx_readout],
                                     params.top_weights, apply_abs=True)
    input_current_readout = jax.lax.cond(
        neuron_idx < 0,
        lambda _: jnp.zeros_like(neuron_states.values),
        lambda _: layer_input * filtered_w_readout,
        None)
    activations = neuron_states.values + input_current_readout

    if grad:
        new_input_residuals = jax.lax.cond(
            neuron_idx < 0,
            lambda _: neuron_states.input_residuals,
            lambda _: neuron_states.input_residuals.at[neuron_idx].add(layer_input),
            None)
        new_input_activity = jax.lax.cond(
            neuron_idx < 0,
            lambda _: neuron_states.input_activity,
            lambda _: neuron_states.input_activity.at[neuron_idx].add(1),
            None)
    else:
        new_input_residuals = neuron_states.input_residuals
        new_input_activity  = neuron_states.input_activity

    @jit
    def last_layer_case(_):
        new_vh, new_hi = neuron_states.values_history, neuron_states.history_index
        if params.history_size > 0:
            new_vh, new_hi = update_history(new_vh, new_hi, activations)
        dummy = jnp.zeros((activations.shape[0], 2))
        return jnp.array(0), dummy, NeuronStates(
            values=activations,
            thresholds=neuron_states.thresholds,
            input_residuals=new_input_residuals,
            input_order=neuron_states.input_order,
            input_activity=new_input_activity,
            layer_activity=neuron_states.layer_activity,
            output_activity=neuron_states.output_activity,
            last_sent_iteration=neuron_states.last_sent_iteration,
            input_vector=neuron_states.input_vector,
            output_vector=neuron_states.output_vector,
            values_history=new_vh,
            history_index=new_hi)

    @jit
    def hidden_layer_case(_):
        input_current, valid_event, safe_neuron_idx = lif_input_current(
            params=params, neuron_idx=neuron_idx,
            layer_input=layer_input, weights=weights)

        u_decayed = lif_decay(
            values=neuron_states.values,
            last_update_iteration=neuron_states.last_sent_iteration,
            iteration=iteration)

        layer_sync_rate = params.sync_rate[layer_idx]
        should_fire     = (iteration >= layer_sync_rate - 1)
        should_fire     = jnp.logical_or(should_fire, ~valid_event)

        activated_output, new_values, u_pre = lif_neuron_forward(
            params=params,
            old_values=neuron_states.values,
            u_decayed=u_decayed,
            input_current=input_current,
            thresholds=neuron_states.thresholds,
            valid_event=valid_event,
            grad=grad)

        activated_output = jax.lax.cond(
            should_fire,
            lambda _: activated_output,
            lambda _: jnp.zeros_like(activated_output),
            None)

        if grad:
            active_indexes     = jnp.where(activated_output > 0, 1, 0)
            new_layer_activity = neuron_states.layer_activity + active_indexes
            new_input_order    = jax.lax.cond(
                valid_event,
                lambda _: neuron_states.input_order.at[safe_neuron_idx].set(iteration),
                lambda _: neuron_states.input_order, None)
            new_output_activity = jax.lax.cond(
                valid_event,
                lambda _: neuron_states.output_activity.at[safe_neuron_idx].add(active_indexes),
                lambda _: neuron_states.output_activity, None)
            new_input_vector = jax.lax.cond(
                valid_event,
                lambda _: neuron_states.input_vector.at[safe_neuron_idx].set(iteration + 1),
                lambda _: neuron_states.input_vector, None)
            new_output_vector = jnp.where(
                activated_output > 0, iteration + 1, neuron_states.output_vector)
        else:
            new_layer_activity  = neuron_states.layer_activity
            new_input_order     = neuron_states.input_order
            new_output_activity = neuron_states.output_activity
            new_input_vector    = neuron_states.input_vector
            new_output_vector   = neuron_states.output_vector

        new_last_sent_iteration = jax.lax.cond(
            valid_event,
            lambda _: jnp.full_like(neuron_states.last_sent_iteration, iteration),
            lambda _: neuron_states.last_sent_iteration, None)

        new_neuron_states = NeuronStates(
            values=new_values,
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

        valid_elements, processed_output = lif_network_layers_forward(
            params=params, key=key, selected_spikes=activated_output)

        if SCHEDULING_POLICY == "rs":
            processed_output, _ = apply_random_scheduling(
                key, processed_output, valid_elements)
        elif SCHEDULING_POLICY == "ms":
            processed_output = apply_momentum_scheduling(
                key, processed_output, u_pre)

        return valid_elements, processed_output, new_neuron_states

    cond = layer_idx == last_layer
    return jax.lax.cond(cond, last_layer_case, hidden_layer_case, None)


@partial(jax.jit, static_argnames=['params', 'grad'])
def predict(params, key, weights, empty_neuron_states,
            batch_data: jnp.ndarray, grad=False):

    @jit
    def input_layer(args):
        neuron_states, x = args
        x_p = jnp.array(x)
        SPIKE_THRESHOLD = 0.1

        if x_p.ndim == 2 and x_p.shape[-1] == 2:
            events = x_p.astype(jnp.float32)
            # Pixel values are already in [0, 1] from the CSV loader.
            # No normalization needed.
            if params.shuffle_input:
                perm   = jax.random.permutation(key, events.shape[0])
                events = events[perm]

            def send_input(i, _):
                send(events[i], dest=rank + process_per_layer, tag=0, comm=comm)
                return None
            jax.lax.fori_loop(0, events.shape[0], send_input, None)
            iteration = events.shape[0]

        else:
            x_p = x_p.reshape(-1)
            # x_p = (x_p > SPIKE_THRESHOLD).astype(jnp.float32)
            x_p = x_p.astype(jnp.float32)   # keep continuous values [0, 1]

            perm = (jax.random.permutation(key, x_p.shape[0])
                    if params.shuffle_input else jnp.arange(x_p.shape[0]))

            def send_input(i, _):
                nidx  = perm[i]
                val = x_p[nidx]
                combined = jnp.array(
                    [nidx.astype(jnp.float32),
                     val], dtype=jnp.float32)
                send(combined, dest=rank + process_per_layer, tag=0, comm=comm)
                return None
            jax.lax.fori_loop(0, x_p.shape[0], send_input, None)
            iteration = x_p.shape[0]

        send(jnp.array([-1.0, -1.0], dtype=jnp.float32),
             dest=rank + process_per_layer, tag=0, comm=comm)
        return (jnp.zeros(()), neuron_states, iteration,
                jnp.zeros((BUFFER_SIZE, 2), dtype=jnp.float32))

    @jit
    def other_layers(args):
        neuron_states, _ = args

        def cond(state):
            # FIX: cast to int32 so -1.0 (float) matches -1 (int)
            _, _, neuron_idx, _, _ = state
            return neuron_idx.astype(jnp.int32) != -1

        @jit
        def forward_pass(state):
            layer_input, neuron_states, neuron_idx, iteration, buffer = state

            @jit
            def send_to_next_layer(args):
                """
                FIX: Send exactly `firing_nb` messages always (static loop bound).
                Padding spikes (index < 0) are ignored by the receiver via
                lif_input_current's valid_event guard.
                Returns jnp.array(0) so JAX cond branches have matching pytrees.
                """
                activated_output = args  # shape (firing_nb, 2)
                firing_nb = params.firing_nb

                def send_one(i, acc):
                    send(activated_output[i],
                         dest=rank + process_per_layer, tag=0, comm=comm)
                    return acc
                jax.lax.fori_loop(0, firing_nb, send_one, jnp.array(0))
                return jnp.array(0)

            received    = recv(jnp.zeros((2,)),
                               source=rank - process_per_layer, tag=0, comm=comm)
            neuron_idx  = received[0]
            layer_input = received[1]

            loop_iterations, activated_output, new_neuron_states = layer_computation(
                params, key, neuron_idx.astype(int), layer_input,
                weights, neuron_states, iteration, grad)

            neuron_states = new_neuron_states

            # Forward to next layer only for hidden layers (not output)
            # Both branches must return same pytree (jnp.array(0))
            jax.lax.cond(
                layer_idx == last_layer,
                lambda _: jnp.array(0),
                send_to_next_layer,
                activated_output)

            return layer_input, neuron_states, neuron_idx, iteration + 1, buffer

        neuron_idx    = 0
        layer_input   = jnp.zeros(())
        initial_state = (layer_input, neuron_states, neuron_idx, 0,
                         jnp.zeros((BUFFER_SIZE, 2)))

        layer_input, neuron_states, neuron_idx, iteration, buffer = (
            jax.lax.while_loop(cond, forward_pass, initial_state))

        # Forward end signal to next layer
        # Both branches must return the same pytree — use a dummy jnp.array(0)
        # so JAX is satisfied. mpi4jax.send returns [] which is not None,
        # hence we wrap and discard the return value explicitly.
        def _send_end(_):
            send(jnp.array([-1.0, -1.0]),
                 dest=rank + process_per_layer, tag=0, comm=comm)
            return jnp.array(0)

        def _no_send(_):
            return jnp.array(0)

        jax.lax.cond(layer_idx != last_layer, _send_end, _no_send, operand=None)

        return layer_input, neuron_states, iteration - 1, buffer

    @jit
    def loop_over_batches(_, x):
        neuron_states = empty_neuron_states
        layer_input, new_neuron_states, iterations, buffer = jax.lax.cond(
            layer_idx == 0, input_layer, other_layers, (neuron_states, x))
        return None, (new_neuron_states.values, iterations,
                      new_neuron_states, buffer)

    _, (all_outputs, all_iterations, all_neuron_states, buffer) = (
        jax.lax.scan(loop_over_batches, None, batch_data))

    mpi4jax.barrier(comm=comm)
    return all_outputs, all_iterations, all_neuron_states, buffer


@partial(jax.jit, static_argnames=['params'])
def predict_bwd(params, key, weights, empty_neuron_states, batch_data):
    all_outputs, iterations, all_neuron_states, buffer = predict(
        params, key, weights, empty_neuron_states, batch_data, grad=True)
    l2_weight_regularization(mpi_config, weights)

    next_grad = recv(jnp.zeros((batch_part, params.layer_sizes[layer_idx])),
                     source=rank + process_per_layer, tag=2, comm=comm)
    weight_grad, th_grad, weight_res = back_prop(
        params, all_neuron_states, next_grad, layer_idx)
    weight_grad += 2 * params.w_reg * weights

    if layer_idx > 1:
        send_grad  = jnp.dot(next_grad, weights.T)
        send_grad *= (~jnp.all(weight_res == 0, axis=2))
        send(send_grad, dest=rank - process_per_layer, tag=2, comm=comm)

    all_activations, all_iterations, sparsity_L = sparsity_loss(
        params, all_neuron_states, iterations)

    scaling = jax.lax.cond(
        params.sparsity_impact[layer_idx] > 0,
        lambda _: params.sparsity_impact[layer_idx] / (
            all_iterations * batch_part * process_per_layer),
        lambda _: 0.0, None)

    input_activity = jnp.sum(all_neuron_states.input_activity, axis=0)
    layer_activity = jnp.sum(all_neuron_states.layer_activity, axis=0)
    layer_activity = gather_batch(layer_activity, mpi_config, average=False)
    input_activity = gather_batch(input_activity, mpi_config, average=False)

    sparsity_residuals   = scaling * layer_activity
    th_sparsity_grad     = -sparsity_residuals
    weight_sparsity_grad = jnp.outer(input_activity, sparsity_residuals)

    return all_outputs, iterations, all_neuron_states, (
        weight_grad, th_grad, weight_sparsity_grad, th_sparsity_grad)


@partial(jax.jit, static_argnames=['params'])
def loss_fn(params, key, weights, empty_neuron_states, target, batch_data):
    all_outputs, iterations, all_neuron_states, buffer = predict(
        params, key, weights, empty_neuron_states, batch_data, grad=True)
    l2_weight_regularization(mpi_config, weights)

    loss, loss_grad = jax.value_and_grad(mean_loss)(all_outputs, target)
    loss_grad /= process_per_layer
    loss      += params.w_reg * jnp.sum(weights ** 2)

    out_grad, weight_grad = jax.vmap(loss_bpp, in_axes=(None, 0, 0))(
        weights, all_neuron_states, loss_grad)
    mean_weight_grad  = jnp.mean(weight_grad, axis=0)
    mean_weight_grad += 2 * params.w_reg * weights
    mean_weight_grad  = jnp.expand_dims(mean_weight_grad, axis=0)

    send(out_grad, dest=rank - process_per_layer, tag=2, comm=comm)
    all_activations, all_iterations, sparsity_L = sparsity_loss(
        params, all_neuron_states, iterations)
    total_loss = loss + sparsity_L

    acc_history, avg_rank = None, None
    if params.history_size > 0:
        target_labels = jnp.argmax(target, axis=-1)
        acc_history, avg_rank = process_history(
            all_neuron_states.values_history,
            all_neuron_states.history_index, target_labels)

    return (loss, all_outputs, iterations, total_loss,
            (acc_history, avg_rank)), (mean_weight_grad, loss_grad)


def sparsity_loss(params, all_neuron_states, iterations):
    if all(x <= 0.0 for x in params.sparsity_impact):
        return 0, 1, 0

    leader_rank = layer_idx * process_per_layer
    activations = gather_batch(all_neuron_states.input_residuals,
                                mpi_config, average=False)
    iterations  = gather_batch(iterations, mpi_config, average=True)

    all_iterations  = 0.0
    all_activations = 0.0
    sparsity_L      = 0.0

    if layer_idx != last_layer and rank == leader_rank:
        send(jnp.sum(activations),
             dest=last_layer * process_per_layer, tag=6, comm=comm)
        if rank == 0:
            send(jnp.mean(iterations),
                 dest=last_layer * process_per_layer, tag=6, comm=comm)
    elif layer_idx == last_layer and rank == leader_rank:
        for i in range(last_layer):
            act_sum = recv(jnp.zeros(1), source=i * process_per_layer,
                           tag=6, comm=comm)
            all_activations += params.sparsity_impact[i] * act_sum[0]
            if i == 0:
                it_mean        = recv(jnp.zeros(1),
                                      source=i * process_per_layer,
                                      tag=6, comm=comm)
                all_iterations = it_mean[0]
        all_activations += params.sparsity_impact[layer_idx] * jnp.sum(activations)
        sparsity_L = all_activations / (
            all_iterations * batch_part * process_per_layer)

    all_iterations = bcast(all_iterations,
                            root=last_layer * process_per_layer, comm=comm)
    return all_activations, all_iterations, sparsity_L


def train(params: Params, key, total_batches, weights,
          empty_neuron_states, opti, trial=None, readInputJson=False):
    global training_generator, validation_generator, test_generator

    if layer_idx == last_layer:
        all_epoch_accuracies      = []
        all_validation_accuracies = []
        all_loss                  = []
        all_history               = []
    all_mean_iterations = []

    if rank == 0:
        print(f"{opti} optimizer selected")
    if opti == "adam":
        solver = optax.adam(learning_rate=params.learning_rate)
    elif opti == "adamw":
        solver = optax.adam(learning_rate=params.learning_rate)
    elif opti == "sgd":
        solver = optax.sgd(learning_rate=params.learning_rate, momentum=0.9)
    elif opti == "rmsprop":
        solver = optax.rmsprop(learning_rate=params.learning_rate,
                                decay=0.9, eps=1e-8)
    elif opti == "lion":
        solver = optax.lion(learning_rate=params.learning_rate)
    else:
        solver = None

    if solver is not None:
        opt_state = solver.init(weights)

    th_solver    = optax.adam(learning_rate=params.threshold_lr)
    th_opt_state = th_solver.init(
        jax.scipy.special.logit(empty_neuron_states.thresholds))

    mpi4jax.barrier(comm=comm)
    start_time = time.time()

    for epoch in tqdm(range(params.num_epochs), disable=TQDM_DISABLE):
        key, subkey = jax.random.split(key)

        if layer_idx == last_layer:
            epoch_correct = 0
            epoch_total   = 0
            epoch_loss    = []

        epoch_iterations = []

        if layer_idx == 0:
            batch_iterator = None
            if rank == 0:
                batch_iterator = iter(training_generator)

        for i in tqdm(range(total_batches[0]), disable=TQDM_DISABLE):
            neuron_states = empty_neuron_states

            if layer_idx == 0:
                batch_x, batch_y = split_batch(
                    params, batch_iterator, mpi_config, 2)
                send(batch_y,
                     dest=last_layer * process_per_layer + rank,
                     tag=10, comm=comm)
                outputs, iterations, all_neuron_states, buffer = predict(
                    params, subkey, weights, neuron_states,
                    batch_data=jnp.array(batch_x))
                all_activations, all_iterations, sparsity_L = sparsity_loss(
                    params, all_neuron_states, iterations)
            else:
                if layer_idx == last_layer:
                    y = recv(jnp.zeros((batch_part,)),
                             source=rank - (last_layer * process_per_layer),
                             tag=10, comm=comm)
                    y_encoded = jnp.array(
                        one_hot_encode(y, num_classes=params.layer_sizes[-1]))
                    (loss, outputs, iterations, total_loss, history), gradients = (
                        loss_fn)(params, subkey, weights, neuron_states,
                                 y_encoded,
                                 jnp.zeros((batch_part, params.layer_sizes[0])))
                    weight_grad = combine_batch_avg(gradients[0], mpi_config)
                    valid_y, batch_correct = accuracy(
                        i, outputs, y, iterations, False)
                    epoch_correct += batch_correct
                    epoch_total   += valid_y.shape[0]
                    epoch_loss.append(loss)
                    if params.history_size > 0:
                        all_history.append(history)
                else:
                    outputs, iterations, all_neuron_states, grads = predict_bwd(
                        params, subkey, weights, neuron_states,
                        jnp.zeros((batch_part, params.layer_sizes[0])))
                    weight_grad, threshold_grad, wsg, tsg = grads
                    threshold_grad = gather_batch(
                        threshold_grad, mpi_config, average=True)
                    weight_grad    = combine_batch_avg(weight_grad, mpi_config)

                    if jnp.any(jnp.array(params.sparsity_impact) > 0):
                        weight_grad    = weight_grad + wsg
                        threshold_grad = threshold_grad + tsg

                    if params.threshold_lr != 0:
                        th_updates, th_opt_state = solver.update(
                            threshold_grad, th_opt_state,
                            empty_neuron_states.thresholds)
                        new_thresholds = jax.nn.sigmoid(
                            optax.apply_updates(
                                jax.scipy.special.logit(
                                    empty_neuron_states.thresholds),
                                th_updates))
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

                if solver is not None:
                    updates, opt_state = solver.update(
                        weight_grad, opt_state, weights)
                    weights = optax.apply_updates(weights, updates)
                else:
                    weights -= params.learning_rate * weight_grad

            epoch_iterations.append(iterations[iterations > 1])

        epoch_iterations    = jnp.concatenate(epoch_iterations)
        mean                = jnp.mean(epoch_iterations)
        all_mean_iterations.append(mean)
        all_mean_iterations = gather_batch(all_mean_iterations, mpi_config)
        all_mean_iterations = all_mean_iterations.tolist()

        if layer_idx != 0 and trial is None:
            jax.debug.print(
                "Rank {} epoch avg_iter={} n_points={} mean_thr={}",
                rank, mean, epoch_iterations.shape[0],
                jnp.mean(empty_neuron_states.thresholds))

        val_accuracy, val_mean, _ = batch_predict(
            params, key, total_batches, weights,
            empty_neuron_states, dataset="val", save=False, debug=False)

        epoch_accuracy = 0.0
        if layer_idx == last_layer:
            mean_loss_val = jnp.mean(jnp.array(epoch_loss))
            all_loss.append(mean_loss_val)
            mean_loss_val = gather_batch(mean_loss_val, mpi_config)
            epoch_accuracy = epoch_correct / epoch_total
            all_epoch_accuracies.append(epoch_accuracy)
            all_validation_accuracies.append(val_accuracy)
            all_epoch_accuracies      = gather_batch(
                all_epoch_accuracies, mpi_config)
            all_validation_accuracies = gather_batch(
                all_validation_accuracies, mpi_config)
            all_epoch_accuracies      = all_epoch_accuracies.tolist()
            all_validation_accuracies = all_validation_accuracies.tolist()
            if rank == size - 1:
                jax.debug.print(
                    "Epoch {} Train={:.2f}% Val={:.2f}% loss={} val_iter={}",
                    epoch, all_epoch_accuracies[-1] * 100,
                    val_accuracy * 100, mean_loss_val, val_mean)

        epoch_accuracy = bcast(epoch_accuracy, root=size - 1, comm=comm)
        if epoch_accuracy >= 0.9999:
            break

        if STORE_EACH_EPOCH:
            weights_dict, all_iteration_mean, thresholds_dict = gather_w_it_th(
                params, weights, jnp.array(all_mean_iterations),
                empty_neuron_states.thresholds)
            if rank == last_layer * process_per_layer:
                store_training_data(
                    size, params, "train",
                    all_epoch_accuracies, all_validation_accuracies,
                    -1.0, time.time() - start_time,
                    all_iteration_mean, weights_dict, all_loss,
                    thresholds_dict, opti, "MLP_temp", all_history,
                    total_batches[0])

    test_accuracy, test_mean, _ = batch_predict(
        params, key, total_batches, weights, empty_neuron_states,
        dataset="test", save=False, debug=True)

    weights_dict, all_iteration_mean, thresholds_dict = gather_w_it_th(
        params, weights, jnp.array(all_mean_iterations),
        empty_neuron_states.thresholds)

    mpi4jax.barrier(comm=comm)
    end_time = time.time()

    MAX_LEN     = 256
    result_path = jnp.zeros(MAX_LEN, dtype=jnp.uint8)
    if rank == last_layer * process_per_layer:
        execution_time  = end_time - start_time
        print(f"Execution Time: {execution_time:.6f} seconds")
        result_path_str = store_training_data(
            size, params, "train",
            all_epoch_accuracies, all_validation_accuracies,
            test_accuracy, execution_time,
            all_iteration_mean, weights_dict, all_loss,
            thresholds_dict, opti, "MLP", all_history, total_batches[0])
        encoded     = np.frombuffer(result_path_str.encode("utf-8"), dtype=np.uint8)
        if encoded.size > MAX_LEN:
            raise ValueError("result_path too long")
        padded      = np.pad(encoded, (0, MAX_LEN - encoded.size), constant_values=0)
        result_path = jnp.array(padded)

    result_path = bcast(result_path, root=last_layer * process_per_layer, comm=comm)
    result_path = bytes(result_path).decode("utf-8").rstrip("\x00")
    mpi4jax.barrier(comm=comm)
    return val_accuracy, result_path


def batch_predict(params: Params, key, total_batches, weights,
                   empty_neuron_states: NeuronStates,
                   dataset: str = "train", save=True, debug=True,
                   readInputJson=False):
    global training_generator, validation_generator, test_generator

    mpi4jax.barrier(comm=comm)
    start_time = time.time()

    if dataset == "train":
        total_batches = total_batches[0]
        if layer_idx == 0 and rank == 0:
            print("Inference on the training set...")
            batch_iterator = iter(training_generator)
        elif layer_idx == 0:
            batch_iterator = None
    elif dataset == "val":
        total_batches = total_batches[1]
        if layer_idx == 0 and rank == 0:
            print("Inference on the validation set...")
            batch_iterator = iter(validation_generator)
        elif layer_idx == 0:
            batch_iterator = None
    elif dataset == "test":
        total_batches = total_batches[2]
        if layer_idx == 0 and rank == 0:
            print("Inference on the test set...")
            batch_iterator = iter(test_generator)
        elif layer_idx == 0:
            batch_iterator = None
    else:
        print("INVALID DATASET")
        return

    if total_batches == 0:
        return -0.01, -1.0, -1.0

    if layer_idx == last_layer:
        epoch_correct = 0
        epoch_total   = 0
        all_history   = []

    epoch_iterations = []

    for i in tqdm(range(total_batches), disable=TQDM_DISABLE):
        neuron_states = empty_neuron_states

        if layer_idx == 0:
            batch_x, batch_y = split_batch(
                params, batch_iterator, mpi_config, 2)
            outputs, iterations, all_neuron_states, buffer = predict(
                params, key, weights, neuron_states, jnp.array(batch_x))
            send(batch_y,
                 dest=last_layer * process_per_layer + rank, tag=10, comm=comm)
        else:
            outputs, iterations, all_neuron_states, buffer = predict(
                params, key, weights, neuron_states,
                jnp.zeros((batch_part, params.layer_sizes[0])))

            if layer_idx == last_layer:
                y = recv(jnp.zeros((batch_part,)),
                         source=rank - (last_layer * process_per_layer),
                         tag=10, comm=comm)
                valid_y, batch_correct = accuracy(
                    i, outputs, y, iterations, print=False)
                epoch_correct += batch_correct
                epoch_total   += valid_y.shape[0]

                if params.history_size > 0:
                    history = process_history(
                        all_neuron_states.values_history,
                        all_neuron_states.history_index, y)
                    all_history.append(history)

        epoch_iterations.append(iterations[iterations > 1])

    epoch_iterations = jnp.concatenate(epoch_iterations)
    mean             = jnp.mean(epoch_iterations)
    mean             = gather_batch(mean, mpi_config)

    if rank != 0 and debug:
        jax.debug.print("Rank {} avg_iter={} n_points={}",
                         rank, mean,
                         epoch_iterations.shape[0] * process_per_layer)

    epoch_accuracy = -1.0
    if layer_idx == last_layer:
        print(f"epoch correct {epoch_correct}, epoch total: {epoch_total}")
        epoch_accuracy = epoch_correct / epoch_total
        epoch_accuracy = gather_batch(epoch_accuracy, mpi_config)
        if debug:
            jax.debug.print("Accuracy: {:.10f}%", epoch_accuracy * 100)

    weights_dict, all_iteration_mean, thresholds_dict = gather_w_it_th(
        params, weights, mean, empty_neuron_states.thresholds)

    mpi4jax.barrier(comm=comm)
    end_time = time.time()

    if rank == last_layer * process_per_layer:
        execution_time = end_time - start_time
        if debug:
            print(f"Execution Time: {execution_time:.6f} seconds")
        if save:
            accuracies = {"train": [-1], "val": [-1], "test": [-1]}
            if dataset in accuracies:
                accuracies[dataset] = [epoch_accuracy]
            store_training_data(
                size, params, "inference",
                accuracies["train"], accuracies["val"], accuracies["test"][0],
                execution_time, all_iteration_mean, weights_dict,
                [], thresholds_dict, "", "MLP", all_history, total_batches)

    return epoch_accuracy, mean, end_time - start_time


def random_layer_params(m, n, key, scale=1e-2):
    w_key, _ = jax.random.split(key)
    return scale * jax.random.normal(w_key, (n, m))


def load_pretrained_weights_from_json(filepath, layer_idx, layer_sizes):
    with open(filepath, 'r') as f:
        data = json.load(f)
    if 'weights' in data:
        weight_data = data['weights']
    else:
        weight_data = data
    json_key = f"layer_{layer_idx}"
    if json_key not in weight_data:
        print(f"  Warning: '{json_key}' not in weights. "
              f"Available keys: {list(weight_data.keys())}")
        return None
    w            = jnp.array(weight_data[json_key])
    expected_in  = layer_sizes[layer_idx - 1]
    expected_out = layer_sizes[layer_idx]
    if w.shape == (expected_out, expected_in):
        w = w.T
        print(f"  Transposed layer_{layer_idx}: {(expected_out, expected_in)} → {w.shape}")
    elif w.shape == (expected_in, expected_out):
        print(f"  Loaded layer_{layer_idx}: shape {w.shape} (correct)")
    else:
        print(f"  Warning: unexpected shape {w.shape}")
    return w


def init_params(key, batch_size, layer_sizes, load_file=False,
                best=False, pretrained_path=None):
    keys = jax.random.split(key, len(layer_sizes))
    if layer_idx != 0:
        if pretrained_path and pretrained_path not in ("", "null"):
            print(f"  Loading pre-trained weights for layer {layer_idx} "
                  f"from: {pretrained_path}")
            w = load_pretrained_weights_from_json(
                pretrained_path, layer_idx, layer_sizes)
            if w is not None:
                return w
            print(f"  Fallback: random init for layer {layer_idx}")
        if load_file:
            filename = (f"tensor_data_{'_'.join(map(str, layer_sizes))}"
                        f"_batch{batch_size}.npz")
            if best:
                filename = "best_" + filename
            filepath = os.path.join("weight/", filename)
            print(f"  Loading weights from {filepath}...")
            w_data = np.load(filepath)
            for i, k in enumerate(w_data.files):
                if i == layer_idx - 1:
                    return jnp.array(w_data[k])
        return random_layer_params(
            layer_sizes[layer_idx], layer_sizes[layer_idx - 1],
            keys[layer_idx])
    else:
        return jnp.zeros((layer_sizes[-1], layer_sizes[0]))


def gather_w_it_th(params, weights, mean_iterations, thresholds):
    leader_rank = layer_idx * process_per_layer
    weights_dict, all_iteration_mean, thresholds_dict = {}, [], {}

    if layer_idx != last_layer and rank == leader_rank:
        send(mean_iterations, dest=last_layer * process_per_layer,
             tag=5, comm=comm)
        if layer_idx != 0:
            send(weights, dest=last_layer * process_per_layer,
                 tag=5, comm=comm)
            send(thresholds, dest=last_layer * process_per_layer,
                 tag=5, comm=comm)
    elif layer_idx == last_layer and rank == leader_rank:
        for i in range(last_layer):
            it_mean = recv(mean_iterations,
                           source=i * process_per_layer, tag=5, comm=comm)
            all_iteration_mean.append(it_mean)
            if i == 0:
                continue
            w = recv(jnp.zeros((params.layer_sizes[i - 1],
                                 params.layer_sizes[i])),
                     source=i * process_per_layer, tag=5, comm=comm)
            weights_dict[f"layer_{i}"] = w.tolist()
            thr = recv(jnp.zeros(params.layer_sizes[i]),
                       source=i * process_per_layer, tag=5, comm=comm)
            thresholds_dict[f"thresholds_{i}"] = thr.tolist()
        all_iteration_mean.append(mean_iterations)
        weights_dict[f"layer_{last_layer}"] = weights.tolist()
        print("all_iteration_mean rank", rank, all_iteration_mean)

    return weights_dict, all_iteration_mean, thresholds_dict


def get_layer_idx(batch_size, layer_sizes, trial=None):
    global layer_idx, process_per_layer, last_layer, batch_part, mpi_config
    last_layer        = len(layer_sizes) - 1
    process_per_layer = size // (last_layer + 1)
    layer_idx         = rank // process_per_layer
    batch_part        = batch_size // process_per_layer
    mpi_config = MPIConfig(
        rank=rank, layer_idx=layer_idx, last_layer=last_layer,
        process_per_layer=process_per_layer, batch_part=batch_part, comm=comm)
    if trial is None:
        print(f"Rank {rank}: layer={layer_idx}  batch_part={batch_part}  "
              f"proc/layer={process_per_layer}  last_layer={last_layer}")


def main(random_seed, key, rank_, size_, comm_,
         trial=None, trial_params=None, config_path=None, data_dir=""):

    global training_generator, validation_generator, test_generator
    global rank, size, comm, TQDM_DISABLE
    global LIF_TAU_M, SCHEDULING_POLICY, MS_NOISE_SCALE

    rank, size, comm = rank_, size_, comm_
    if rank != 0:
        TQDM_DISABLE = True

    config = load_config_with_defaults(config_path)

    LIF_TAU_M            = float(config.get('tau_m', 100.0))
    SCHEDULING_POLICY    = config.get('scheduling_policy', 'ms')
    MS_NOISE_SCALE       = float(config.get('momentum_noise_scale', 1e-6))
    forward_group_size   = int(config.get('forward_group_size', 4))
    stop_condition       = config.get('stop_condition', 'on_output')
    momentum_noise_scale = float(config.get('momentum_noise_scale', 0.1))

    if rank == 0:
        print(f"TMLR config: scheduling={SCHEDULING_POLICY}  "
              f"tau_m={LIF_TAU_M}  F={forward_group_size}  "
              f"stop={stop_condition}  momentum_noise={momentum_noise_scale}")

    dataset         = config['dataset']
    layer_sizes     = tuple(config['layer_sizes'])
    batch_size      = config['batch_size']
    restrict        = config['restrict']
    init_thresholds = config['init_thresholds']
    load_file       = config['load_file']
    best            = config['best']
    rerun           = config['rerun']

    if size % len(layer_sizes) != 0:
        print(f"Error: number of MPI ranks ({size}) must be divisible by "
              f"number of layers ({len(layer_sizes)})")
        sys.exit(1)

    get_layer_idx(batch_size, layer_sizes, trial)

    if batch_size % process_per_layer != 0:
        print(f"Error: batch_size ({batch_size}) must be divisible by "
              f"process_per_layer ({process_per_layer})")
        sys.exit(1)

    for _ in [128]:
        key, subkey = jax.random.split(key)
        total_train_batches = total_val_batches = total_test_batches = 0
        max_nonzero = 0

        pretrained_path = config.get("pretrained_path", None)
        if pretrained_path in ("", "null", "Null", None):
            pretrained_path = None
        weights = init_params(subkey, batch_size, layer_sizes,
                               load_file=load_file, best=best,
                               pretrained_path=pretrained_path)

        if rank == 0:
            downsample = False
            match dataset:
                case "mnist" | "smnist" | "psmnist":
                    sequential = (dataset in ("smnist", "psmnist"))
                    permuted   = (dataset == "psmnist")
                    if layer_sizes[0] == 14 * 14:
                        downsample = True
                    loader = partial(mnist_loader_manual,
                                     sequential=sequential, permuted=permuted)
                case "shd":
                    loader = torch_SHD_loader
                case "nmnist":
                    loader = torch_nmnist_loader
                case "dvs":
                    if layer_sizes[0] == 64 * 64 * 2:
                        downsample = True
                    loader = torch_DVSGesture_loader
                case "iris":
                    loader = torch_iris_loader
                case _:
                    raise ValueError(f"Unknown dataset: {dataset}")

            train_data, val_data, test_data, max_nonzero = loader(
                batch_size=batch_size, shuffle=False,
                CNN_preprocess=False, downsample=downsample,
                data_dir=data_dir)
            training_generator,   total_train_batches = train_data
            validation_generator, total_val_batches   = val_data
            test_generator,       total_test_batches  = test_data

        total_train_batches, total_val_batches, total_test_batches = bcast(
            jnp.array([total_train_batches, total_val_batches,
                       total_test_batches]), root=0, comm=comm)
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
            firing_nb=config['firing_nb'],
            sync_rate=config['sync_rate'],
            max_nonzero=max_nonzero,
            shuffle_input=config['shuffle_input'],
            threshold_lr=config['threshold_lr'],
            sparsity_impact=tuple(config['sparsity_impact']),
            w_reg=config['w_reg'],
            rerun="",
            top_weights=config['top_weights'],
            history_size=config['history_size'])

        if rerun is not None and rerun not in ("", "null", "Null"):
            override_list = config.get('override_params', None)
            params, weights, thresholds = rerun_init(
                rerun, mpi_config, params, override_params=override_list)

        if rank == 0:
            print(f"Batches: train={total_train_batches}  "
                  f"val={total_val_batches}  test={total_test_batches}")
            print(params)

        empty_neuron_states = NeuronStates(
            values=jnp.zeros(layer_sizes[layer_idx]),
            thresholds=thresholds,
            input_residuals=np.zeros((layer_sizes[layer_idx - 1],)),
            input_order=jnp.full(
                (layer_sizes[layer_idx - 1],), -1, dtype=int),
            input_activity=jnp.full(
                (layer_sizes[layer_idx - 1],), 0, dtype=int),
            layer_activity=jnp.zeros(
                (layer_sizes[layer_idx],), dtype=int),
            output_activity=jnp.zeros(
                (layer_sizes[layer_idx - 1], layer_sizes[layer_idx])),
            last_sent_iteration=jnp.zeros(
                (layer_sizes[layer_idx],), dtype=jnp.int32),
            input_vector=jnp.zeros(
                (layer_sizes[layer_idx - 1],), dtype=int),
            output_vector=jnp.zeros(
                (layer_sizes[layer_idx],), dtype=int),
            values_history=jnp.zeros(
                (params.history_size, layer_sizes[layer_idx])),
            history_index=jnp.array(0, dtype=jnp.int32))

        total_batches = (total_train_batches, total_val_batches,
                         total_test_batches)

        mode = config['mode']
        if mode == 'inference':
            batch_predict(params, key, total_batches, weights,
                          empty_neuron_states, "test", save=True, debug=True)
        elif mode == 'training':
            train(params, key, total_batches, weights,
                  empty_neuron_states, config['optimizer'], trial)
        else:
            print(f"Unknown mode '{mode}'. Use 'training' or 'inference'.")
            sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Async SNN — JAX + MPI")
    parser.add_argument('--config',   type=str, default=None)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--data_dir', type=str, default="")
    args = parser.parse_args()

    random_seed = args.seed
    key         = jax.random.key(random_seed)
    comm        = MPI.COMM_WORLD
    rank        = comm.Get_rank()
    size        = comm.Get_size()

    main(random_seed, key, rank, size, comm,
         config_path=args.config, data_dir=args.data_dir)