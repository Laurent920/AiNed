import os

# os.environ["JAX_PLATFORMS"] = "gpu"
# os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=18'
import jax
print("Running on ", jax.default_backend())

import jax.numpy as jnp
from jax import jit
from functools import partial
import optax
from flax.struct import dataclass
import dataclasses
from typing import List, Tuple
import time
import numpy as np
from tqdm import tqdm

from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils

# Configure JAX
jax.config.update("jax_debug_nans", True)

# ============================================================================
# region DATA STRUCTURES
# ============================================================================

@dataclass
class NeuronStates:
    values: jnp.ndarray
    thresholds: jnp.ndarray
    input_residuals: jnp.ndarray
    input_order: jnp.ndarray
    input_activity: jnp.ndarray
    layer_activity: jnp.ndarray
    output_activity: jnp.ndarray
    last_sent_iteration: int
    input_vector: jnp.ndarray
    output_vector: jnp.ndarray
    values_history: jnp.ndarray
    history_index: jnp.ndarray

@dataclasses.dataclass(frozen=True)
class Params:
    dataset: str
    random_seed: int
    layer_sizes: Tuple[int, ...]
    init_thresholds: float
    num_epochs: int
    learning_rate: float
    batch_size: int
    load_file: bool
    shuffle_activations: bool
    restrict: Tuple[float, ...]
    firing_nb: int
    sync_rate: int
    max_nonzero: int
    shuffle_input: bool
    threshold_lr: float
    sparsity_impact: Tuple[float, ...]
    rerun: str
    async_layer: int
    history_size: int

# ============================================================================
# region ACTIVATIONS
# ============================================================================

@jit
def activation_func(neuron_states, activations):
    """Apply threshold-based activation function"""
    return jnp.where(activations > neuron_states.thresholds, activations, 0.0)

@partial(jax.jit, static_argnames=['k'])
def keep_top_k(x, k):
    """Keep only top-k activations, zero out the rest"""
    k_safe = min(k, x.shape[0])
    _, top_indices = jax.lax.top_k(x, k_safe)
    
    mask = jnp.zeros(x.shape)
    mask = mask.at[top_indices].set(1)
    
    return x * mask

@partial(jax.jit, static_argnames=['params', 'layer_idx'])
def process_activated_output(key, arr: jnp.ndarray, params, layer_idx: int):
    """
    Process the output of a layer from (1d array) to (2d array) with [(neuron idx, value)]
    Value == 0 are filled with index==-2
    """
    max_len = params.layer_sizes[layer_idx]
    
    # Indices of nonzero values (padded with -2)
    idx = jnp.nonzero(arr, size=max_len, fill_value=-2)[0]
    vals = jnp.where(idx != -2, arr[idx], -2)
    
    # Stack before shuffle
    pairs = jnp.stack([idx, vals], axis=1)
    
    def do_shuffle(pairs):
        mask = (idx != -2).astype(jnp.int32)
        rand_keys = jax.random.uniform(key, (max_len,))
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

# ============================================================================
# region LAYER COMPUTATION
# ============================================================================

@partial(jax.jit, static_argnames=['params', 'grad', 'layer_idx', 'is_last_layer'])
def layer_computation(params, key, neuron_idx, layer_input, weights, neuron_states, 
                     iteration=0, grad=False, layer_idx=0, is_last_layer=False):
    """
    Compute layer activations for a single input event.
    
    Args:
        params: Network parameters
        key: Random key for shuffling
        neuron_idx: Index of input neuron (-2 for padding, -1 for end signal)
        layer_input: Input value(s)
        weights: Weight matrix for this layer
        neuron_states: Current neuron states
        iteration: Current iteration number
        grad: Whether to track gradients
        layer_idx: Current layer index
        is_last_layer: Whether this is the output layer
    
    Returns:
        (valid_elements, processed_output, new_neuron_states)
    """
    
    # Compute activations
    activations = jax.lax.cond(
        neuron_idx < 0,
        lambda _: neuron_states.values,
        lambda _: jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values,
        None
    )
    
    # Update residuals and activity if tracking gradients
    if grad:
        new_input_residuals = jax.lax.cond(
            neuron_idx < 0,
            lambda _: neuron_states.input_residuals,
            lambda _: neuron_states.input_residuals.at[neuron_idx].add(layer_input),
            None
        )
        new_input_activity = jax.lax.cond(
            neuron_idx < 0,
            lambda _: neuron_states.input_activity,
            lambda _: neuron_states.input_activity.at[neuron_idx].add(1),
            None
        )
    else:
        new_input_residuals = neuron_states.input_residuals
        new_input_activity = neuron_states.input_activity
    
    @jit
    def last_layer_case(_):
        """Handle output layer - just accumulate activations"""
        new_values_history = neuron_states.values_history
        new_history_index = neuron_states.history_index
        
        if params.history_size > 0:
            new_values_history = new_values_history.at[new_history_index].set(activations)
            new_history_index = (new_history_index + 1) % params.history_size
        
        dummy_activations = jnp.zeros((activations.shape[0], 2))
        dummy_activations = jnp.zeros((activations.shape[0]))
        
        return  jnp.array(0), dummy_activations, NeuronStates(
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
            values_history=new_values_history,
            history_index=new_history_index
        )
    
    @jit
    def hidden_layer_case(_):
        """Handle hidden layers - apply thresholds and firing rules"""
        # Determine if we should fire
        fire = (iteration - neuron_states.last_sent_iteration) >= params.sync_rate
        async_fire = jnp.logical_or(params.async_layer < 0, layer_idx <= params.async_layer)
        fire = jnp.logical_and(fire, async_fire)
        fire = jnp.logical_or(fire, neuron_idx < 0)  # Fire if last input received
        
        # Apply activation function
        activated_output = jax.lax.cond(
            fire,
            lambda args: activation_func(args[0], args[1]),
            lambda _: jnp.zeros(activations.shape),
            (neuron_states, activations)
        )
        
        # Keep top-k activations
        activated_output = keep_top_k(activated_output, params.firing_nb)
        
        # Apply restriction penalty
        penalty = jax.lax.cond(
            params.restrict[layer_idx] <= 0,
            lambda _: activated_output,
            lambda _: activated_output * params.restrict[layer_idx],
            None
        )
        
        if grad:
            # Track which neurons activated
            active_indexes = jnp.where(activated_output > 0, 1, 0)
            new_layer_activity = neuron_states.layer_activity + active_indexes
            
            last_neuron_idx = jnp.argmax(neuron_states.input_order)
            new_neuron_idx = jax.lax.cond(
                neuron_idx < 0,
                lambda _: last_neuron_idx,
                lambda _: neuron_idx,
                None
            )
            
            new_input_order = neuron_states.input_order.at[new_neuron_idx].set(iteration)
            new_output_activity = neuron_states.output_activity.at[new_neuron_idx].add(active_indexes)
            
            new_input_vector = neuron_states.input_vector.at[neuron_idx].set(iteration + 1)
            new_output_vector = jnp.where(
                activated_output > 0,
                iteration + 1,
                neuron_states.output_vector
            )
        else:
            new_layer_activity = neuron_states.layer_activity
            new_input_order = neuron_states.input_order
            new_output_activity = neuron_states.output_activity
            new_input_vector = neuron_states.input_vector
            new_output_vector = neuron_states.output_vector
        
        new_last_sent_iteration = jax.lax.cond(
            fire,
            lambda _: iteration,
            lambda _: neuron_states.last_sent_iteration,
            None
        )
        
        new_neuron_states = NeuronStates(
            values=activations - penalty,
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
            history_index=neuron_states.history_index
        )
        
        valid_elements = jnp.count_nonzero(activated_output)
        processed_output = process_activated_output(key, activated_output, params, layer_idx)
        return valid_elements, activated_output, new_neuron_states
        return valid_elements, processed_output, new_neuron_states
    
    return jax.lax.cond(is_last_layer, last_layer_case, hidden_layer_case, None)

def organize(all_outputs, layer_size, max_events):
    flat_output = all_outputs.flatten()
    # allocate
    total_size = max_events * layer_size
    
    # Indices of nonzero values (padded with -2)
    idx = jnp.nonzero(flat_output, size=total_size, fill_value=-2)[0]
    vals = jnp.where(idx != -2, flat_output[idx], -2)

    # Stack before shuffle
    pairs = jnp.stack([idx%layer_size, vals], axis=1)
    return pairs

# ============================================================================
# region NN CLASS
# ============================================================================

class EventBasedNN:
    """
    Event-based neural network that processes sparse neuron activations.
    Each event is represented as (neuron_index, value).
    """
    
    def __init__(self, params: Params, weights: List[jnp.ndarray]):
        """
        Initialize the network.
        
        Args:
            params: Network parameters
            weights: List of weight matrices for each layer
        """
        self.params = params
        self.layer_sizes = params.layer_sizes
        self.num_layers = len(self.layer_sizes)
        self.weights = weights
        
        # Initialize neuron states for each layer
        self.empty_neuron_states = []
        for i in range(self.num_layers):
            init_thresholds = params.init_thresholds if params.init_thresholds is not None else 0.0
            thresholds = jnp.full(self.layer_sizes[i], init_thresholds)
            
            prev_layer_size = self.layer_sizes[i-1] if i > 0 else self.layer_sizes[i]
            
            state = NeuronStates(
                values=jnp.zeros(self.layer_sizes[i]),
                thresholds=thresholds,
                input_residuals=jnp.zeros((prev_layer_size,)),
                input_order=jnp.full((prev_layer_size,), -1, dtype=int),
                input_activity=jnp.full((prev_layer_size,), 0, dtype=int),
                layer_activity=jnp.zeros((self.layer_sizes[i],), dtype=int),
                output_activity=jnp.zeros((prev_layer_size, self.layer_sizes[i])),
                last_sent_iteration=0,
                input_vector=jnp.zeros((prev_layer_size), dtype=int),
                output_vector=jnp.zeros((self.layer_sizes[i]), dtype=int),
                # Only the output layer records history; other layers keep an empty buffer.
                values_history=jnp.zeros((params.history_size if i == self.num_layers - 1 else 0, self.layer_sizes[i])),
                history_index=jnp.array(0, dtype=jnp.int32)
            )
            self.empty_neuron_states.append(state)

    def process_layer(self, key, layer_idx: int, input_events: jnp.ndarray, nb_valid: int,
                     neuron_states: NeuronStates, grad: bool = False) -> Tuple:
        """
        Process all events through a single layer.
        
        Args:
            key: Random key
            layer_idx: Current layer index (0 = input, num_layers-1 = output)
            input_events: Input events array (max_events, 2) with [(neuron_idx, value)]
            nb_valid: Number of valid events to process (dynamic)
            neuron_states: Current neuron states
            grad: Whether to track gradients
        
        Returns:
            (output_events, final_neuron_states, iteration_count)
        """
        is_last_layer = (layer_idx == self.num_layers - 1)
        weights = self.weights[layer_idx] if layer_idx > 0 else None
        max_events = input_events.shape[0]
        
        # Initialize output arrays
        all_outputs = jnp.full((max_events, self.layer_sizes[layer_idx]), 0.0)
        all_valid = jnp.zeros(max_events, dtype=jnp.int32)
        
        def process_single_event(i, carry):
            neuron_states, iteration, outputs, valid_arr = carry
            
            event_data = input_events[i]
            neuron_idx, value = event_data
            neuron_idx = neuron_idx.astype(int)
            
            # Process the event
            def valid_event(_):
                valid_elements, processed_output, new_neuron_states = layer_computation(
                    self.params, key, neuron_idx, value, weights, neuron_states,
                    iteration, grad, layer_idx, is_last_layer
                )
                
                # valid_elements, processed_output, new_neuron_states = layer_computation(
                #     self.params, key, neuron_idx, value, weights, neuron_states,
                #     iteration, grad, layer_idx, is_last_layer
                # )
                return new_neuron_states, processed_output, valid_elements, iteration+1
            
            def skip_event(_):
                # return neuron_states, jnp.full((self.layer_sizes[layer_idx], 2), -2.0), jnp.array(0), iteration
                return neuron_states, jnp.full((self.layer_sizes[layer_idx],), 0.0), jnp.array(0), iteration
            
            new_neuron_states, output, valid, new_iteration = jax.lax.cond(
                neuron_idx >= -1,
                valid_event,
                skip_event,
                None
            )
            # if layer_idx == 2:
            #     jax.debug.print("Adding at index {}: {}, event data : {}", i, valid, event_data)
            # Update arrays
            outputs = outputs.at[i].set(output)
            valid_arr = valid_arr.at[i].set(valid)
            
            return (new_neuron_states, new_iteration, outputs, valid_arr)
        
        # Use fori_loop to iterate only over nb_valid elements
        initial_carry = (neuron_states, 0, all_outputs, all_valid)
        final_states, final_iteration, all_outputs, all_valid = jax.lax.fori_loop(
            0, nb_valid, process_single_event, initial_carry
        )
        
        valid, organized_all_outputs = 0, jnp.zeros(1)
        if not is_last_layer:
            organized_all_outputs = organize(all_outputs, self.layer_sizes[layer_idx], max_events)
            valid = jnp.count_nonzero(all_outputs)
        return valid, organized_all_outputs, final_states, final_iteration

    # def process_layer(self, key, layer_idx: int, input_events: jnp.ndarray, valid_nb: int,
    #                  neuron_states: NeuronStates, grad: bool = False) -> Tuple:
    #     """
    #     Process all events through a single layer.
        
    #     Args:
    #         key: Random key
    #         layer_idx: Current layer index (0 = input, num_layers-1 = output)
    #         input_events: Input events array (max_events, 2) with [(neuron_idx, value)]
    #         neuron_states: Current neuron states
    #         grad: Whether to track gradients
        
    #     Returns:
    #         (output_events, final_neuron_states, iteration_count)
    #     """
    #     is_last_layer = (layer_idx == self.num_layers - 1)
    #     weights = self.weights[layer_idx] if layer_idx > 0 else None
        
    #     # Process events sequentially
    #     def process_event_step(carry, event_data):
    #         neuron_states, iteration = carry
    #         neuron_idx, value = event_data
    #         neuron_idx = neuron_idx.astype(int)
            
    #         # Skip padding (-2) but process end signal (-1)
    #         def valid_event(_):
    #             valid_elements, processed_output, new_neuron_states = layer_computation(
    #                 self.params, key, neuron_idx, value, weights, neuron_states,
    #                 iteration, grad, layer_idx, is_last_layer
    #             )
    #             return new_neuron_states, processed_output, valid_elements, iteration+1
            
    #         def skip_event(_):
    #             return neuron_states, jnp.full((self.layer_sizes[layer_idx], 2), -2.0), jnp.array(0), iteration
            
    #         new_neuron_states, output, valid, iteration = jax.lax.cond(
    #             neuron_idx >= -1,
    #             valid_event,
    #             skip_event,
    #             None
    #         )
            
    #         return (new_neuron_states, iteration), (output, valid, neuron_idx)
        
    #     # Scan through all events
    #     (final_states, final_iteration), (all_outputs, all_valid, all_indices) = jax.lax.scan(
    #         process_event_step,
    #         (neuron_states, 0),
    #         input_events
    #     )
    #     # Flatten outputs and remove padding
    #     shapes = all_outputs.shape
    #     all_outputs = all_outputs.reshape((shapes[0]*shapes[1], *shapes[2:]))
    #     valid, organized_all_outputs = 0, all_outputs
    #     return valid, organized_all_outputs, final_states, final_iteration
    
    def forward(self, key, batch_events: jnp.ndarray, grad: bool = False):
        """
        Process a batch of event sequences through the entire network.
        
        Args:
            key: Random key
            batch_events: Batch of event sequences, shape (batch_size, max_events, 2)
            grad: Whether to track gradients
        
        Returns:
            (batch_outputs, all_neuron_states, iterations)
        """
        
        def process_sample(sample_key, sample_events):
            """Process a single sample through all layers"""
            current_events = sample_events
            valid_nb = jnp.count_nonzero(sample_events != -2)//2
            all_layer_states = []
            
            for layer_idx in range(1, self.num_layers):
                layer_key = jax.random.fold_in(sample_key, layer_idx)
                
                # jax.debug.print("events: {}", current_events.shape)
                # Get initial neuron states for this layer
                neuron_states = self.empty_neuron_states[layer_idx]
                
                # Process through layer
                valid_nb, layer_outputs, final_states, iterations = self.process_layer(
                    layer_key, layer_idx, current_events, valid_nb, neuron_states, grad
                )
                
                all_layer_states.append(final_states)
                
                # Output of this layer becomes input to next layer
                current_events = layer_outputs  # Placeholder
            
            return final_states.values, iterations, all_layer_states
        
        # Process each sample in batch
        batch_size = batch_events.shape[0]
        keys = jax.random.split(key, batch_size)
        
        # TODO: Decide whether to use vmap or scan for batch processing
        # Option 1: vmap (parallel) - uses more memory but faster
        # all_outputs, all_iterations, all_states = jax.vmap(
        #     process_sample, in_axes=(0, 0)
        # )(keys, batch_events)
        
        # Option 2: scan (sequential) - uses less memory
        def scan_batch(_, data):
            key, events = data
            return None, process_sample(key, events)
        
        _, (all_outputs, all_iterations, all_states) = jax.lax.scan(
            scan_batch, None, (keys, batch_events)
        )
        
        return all_outputs, all_iterations, all_states
    
    def forward_with_loss(self, key, batch_events: jnp.ndarray, batch_labels: jnp.ndarray):
        """
        Forward pass with loss computation and gradient tracking.
        
        Args:
            key: Random key
            batch_events: Batch of event sequences, shape (batch_size, max_events, 2)
            batch_labels: One-hot encoded labels, shape (batch_size, num_classes)
        
        Returns:
            (loss, outputs, iterations, all_layer_states, gradients)
        """
        # Forward pass with gradient tracking
        outputs, iterations, all_layer_states = self.forward(key, batch_events, grad=True)
        # print(outputs.shape, iterations.shape, len(all_layer_states))
        # (loss, total_loss, outputs, iterations), (all_weight_grads, all_th_grads, sparsity_grads)
        # Compute loss and its gradient
        loss, loss_grad = jax.value_and_grad(mean_loss)(outputs, batch_labels)
        
        # Compute gradients for output layer
        output_layer_idx = self.num_layers - 1
        out_grad, weight_grad = jax.vmap(loss_bpp, in_axes=(None, 0, 0))(
            self.weights[-1], all_layer_states[-1], loss_grad
        )  # Shapes: (batch, prev_layer), (batch, prev_layer, output_layer)
        
        mean_weight_grad = jnp.mean(weight_grad, axis=0)
        mean_weight_grad = jnp.expand_dims(mean_weight_grad, axis=0)
        # print(out_grad.shape, weight_grad.shape, mean_weight_grad.shape)
        # Store gradients for all layers
        all_weight_grads = [mean_weight_grad]
        all_th_grads = []
        
        # Backpropagate through hidden layers
        current_grad = out_grad
        for layer_idx in range(self.num_layers - 2, 0, -1):
            # Compute gradients for this layer
            weight_grad, th_grad, weight_res = back_prop(
                self.params, all_layer_states[layer_idx-1], current_grad, layer_idx
            )
            
            all_weight_grads.insert(0, weight_grad)
            all_th_grads.insert(0, th_grad)
            
            # Compute gradient for previous layer
            current_grad = jnp.dot(current_grad, self.weights[layer_idx].T)
                    
        # Compute sparsity loss and gradients if enabled
        sparsity_grads = None
        total_loss = loss
        if any(x > 0 for x in self.params.sparsity_impact):
            all_activations, mean_iterations, sparsity_L = sparsity_loss(
                self.params, all_layer_states, iterations
            )
            total_loss += sparsity_L
            
            # Compute sparsity gradients for each layer
            sparsity_grads = []
            for layer_idx in range(1, self.num_layers):
                scaling = self.params.sparsity_impact[layer_idx]
                if scaling > 0:
                    scaling /= (mean_iterations * self.params.batch_size)
                    
                    layer_activity = jnp.sum(
                        all_layer_states[layer_idx].layer_activity, axis=0
                    )
                    input_activity = jnp.sum(
                        all_layer_states[layer_idx].input_activity, axis=0
                    )
                    
                    sparsity_residuals = scaling * layer_activity
                    th_sparsity_grad = -sparsity_residuals
                    weight_sparsity_grad = jnp.outer(input_activity, sparsity_residuals)
                    
                    sparsity_grads.append((weight_sparsity_grad, th_sparsity_grad))
                else:
                    sparsity_grads.append((None, None))
        
        return (loss, total_loss, outputs, iterations), (all_weight_grads, all_th_grads, sparsity_grads)
    
    def train_step(self, key, batch_events: jnp.ndarray, batch_labels: jnp.ndarray,
                   opt_states, th_opt_states, solvers):
        """
        Single training step.
        
        Args:
            key: Random key
            batch_events: Batch of event sequences
            batch_labels: One-hot encoded labels
            opt_states: Optimizer states for weights
            th_opt_states: Optimizer states for thresholds
            solvers: Tuple of (weight_solver, threshold_solver)
        
        Returns:
            Updated weights, thresholds, optimizer states, and metrics
        """
        weight_solver, th_solver = solvers
        
        # Forward pass with loss
        (loss, total_loss, outputs, iterations), (weight_grads, th_grads, sparsity_grads) = \
            self.forward_with_loss(key, batch_events, batch_labels)
        
        # print((weight_grads[1].shape), len(th_grads), sparsity_grads)
        # return 0,0, jnp.zeros((batch_events.shape[0], 10)), jnp.zeros(batch_events.shape[0]), opt_states, th_opt_states
        # Update weights and thresholds for each layer
        new_weights = [jnp.zeros(1)]
        new_opt_states = [opt_states]
        new_th_opt_states = []
        new_neuron_states = []
        
        for layer_idx in range(1, self.num_layers):
            # Weight updates
            w_grad = weight_grads[layer_idx - 1][0]  # Remove batch dimension
            
            # Add sparsity gradients if applicable
            if sparsity_grads is not None and sparsity_grads[layer_idx - 1][0] is not None:
                w_grad = w_grad + sparsity_grads[layer_idx - 1][0]
            
            # Apply optimizer
            if weight_solver is not None:
                # print(f"w grad shape {w_grad.shape}, w {self.weights[layer_idx].shape}")
                updates, new_opt_state = weight_solver.update(
                    w_grad, opt_states[layer_idx], self.weights[layer_idx]
                )
                new_weight = optax.apply_updates(self.weights[layer_idx], updates)
            else:
                new_weight = self.weights[layer_idx]
                new_opt_state = opt_states[layer_idx]
            
            new_weights.append(new_weight)
            new_opt_states.append(new_opt_state)
            
            # Threshold updates (only for hidden layers)
            if layer_idx-1 < self.num_layers -2:
                if layer_idx-1 < self.num_layers - 1 and self.params.threshold_lr > 0:
                    th_grad = th_grads[layer_idx - 1]
                    
                    # Add sparsity gradients if applicable
                    if sparsity_grads is not None and sparsity_grads[layer_idx - 1][1] is not None:
                        th_grad = th_grad + sparsity_grads[layer_idx - 1][1]
                    
                    # Update in logit space for stability
                    current_thresholds = self.empty_neuron_states[layer_idx].thresholds
                    th_updates, new_th_opt_state = th_solver.update(
                        th_grad, th_opt_states[layer_idx-1],
                        jax.scipy.special.logit(current_thresholds)
                    )
                    new_thresholds = jax.nn.sigmoid(
                        optax.apply_updates(jax.scipy.special.logit(current_thresholds), th_updates)
                    )
                    
                    # Update neuron states with new thresholds
                    updated_state = dataclasses.replace(
                        self.empty_neuron_states[layer_idx],
                        thresholds=new_thresholds
                    )
                    new_neuron_states.append(updated_state)
                else:
                    new_th_opt_state = th_opt_states[layer_idx-1] if layer_idx-1 < len(th_opt_states) else None
                    new_neuron_states.append(self.empty_neuron_states[layer_idx])
                    
                if new_th_opt_state is not None:
                    new_th_opt_states.append(new_th_opt_state)

        # Update network weights and states
        self.weights = new_weights
        # for i, state in enumerate(new_neuron_states):
        #     if i + 1 < len(self.empty_neuron_states):
        #         self.empty_neuron_states[i + 1] = state
        
        return loss, total_loss, outputs, iterations, new_opt_states, new_th_opt_states

# ============================================================================
# region LOSS / TRAINING 
# ============================================================================

@jax.jit
def softmax_cross_entropy_with_logits(logits, labels):
    """Compute softmax cross-entropy loss for a single sample"""
    # Compute the softmax in a numerically stable way
    logits_max = jnp.max(logits, axis=0, keepdims=True)
    exps = jnp.exp(logits - logits_max)
    softmax = exps / (jnp.sum(exps, axis=0, keepdims=True) + 1e-8)
    # Compute the cross-entropy loss
    cross_entropy = -jnp.sum(labels * jnp.log(softmax + 1e-8), axis=0)
    return cross_entropy

@jax.jit
def mean_loss(logits, labels):
    """Compute mean cross-entropy loss over batch"""
    batched_softmax_cross_entropy = jax.vmap(softmax_cross_entropy_with_logits, in_axes=(0, 0))
    losses = batched_softmax_cross_entropy(logits, labels)
    return jnp.mean(losses)

@jax.jit
def loss_bpp(weights, all_neuron_states, loss_grad):
    """
    Compute gradients for output layer.
    For each batch element:
    - Compute the gradient w.r.t the output of the layer
    - Compute the gradient w.r.t the weights of the layer
    
    Args:
        weights: Output layer weights, shape (prev_layer, output_layer)
        all_neuron_states: Neuron states for this sample
        loss_grad: Loss gradient, shape (output_layer,)
    
    Returns:
        out_grad: Gradient w.r.t layer input, shape (prev_layer,)
        weight_grad: Gradient w.r.t weights, shape (prev_layer, output_layer)
    """
    # (1) Gradient w.r.t the output of the previous layer
    out_grad = jnp.dot(weights, loss_grad)  # Shape: (prev_layer,)
    
    # (2) Gradient w.r.t the weights
    loss_grad_expanded = jnp.expand_dims(loss_grad, axis=1)  # Shape: (output_layer, 1)
    all_residuals = all_neuron_states.input_residuals  # Shape: (prev_layer,)
    
    weight_grad = loss_grad_expanded * all_residuals  # Shape: (output_layer, prev_layer)
    
    return out_grad, weight_grad.T

def sparsity_loss(params, all_neuron_states, iterations):
    """
    Compute sparsity loss based on layer activity.
    
    Args:
        params: Network parameters
        all_neuron_states: List of neuron states for each layer
        iterations: Number of iterations per sample
    
    Returns:
        (all_activations, all_iterations, sparsity_L)
    """
    if all(x <= 0.0 for x in params.sparsity_impact):
        return 0.0, iterations, 0.0
    
    all_activations = 0.0
    
    # Sum activations weighted by sparsity impact for each layer
    for layer_idx, neuron_states in enumerate(all_neuron_states):
        if params.sparsity_impact[layer_idx] > 0:
            # Sum of input residuals for this layer
            layer_activations = jnp.sum(neuron_states.input_residuals)
            all_activations += params.sparsity_impact[layer_idx] * layer_activations
    
    # Normalize by iterations and batch size
    mean_iterations = jnp.mean(iterations)
    sparsity_L = all_activations / (mean_iterations * params.batch_size)
    
    return all_activations, mean_iterations, sparsity_L

@partial(jax.jit, static_argnames=['params', 'layer_idx'])
def compute_full_bpp(params, all_neuron_states, next_grad, layer_idx):
    """
    Compute full backpropagation for a single element in the batch.
    
    This implements the biological plausibility constraint where gradients
    flow only through connections that were actually used (weight residuals).
    
    Args:
        params: Network parameters
        all_neuron_states: Neuron states for this sample
        next_grad: Gradient from next layer, shape (current_layer_size,)
        layer_idx: Current layer index
    
    Returns:
        weight_grad: Gradient w.r.t weights, shape (prev_layer, current_layer)
        weight_res: Weight residuals mask, shape (prev_layer, current_layer)
    """
    input_vector = all_neuron_states.input_vector
    output_vector = all_neuron_states.output_vector
    layer_activity = all_neuron_states.layer_activity
    
    # (1) Compute weight residuals: connections that were used
    # Shape: (prev_layer, current_layer)
    weight_res = (input_vector[:, None] <= output_vector[None, :])
    
    # (2) Apply restriction (geometric series for repeated activations)
    # Shape: (current_layer,)
    a = params.restrict[layer_idx]
    new_layer_activity = jnp.where(
        a > 0,
        (1 - jnp.power((1 - a), layer_activity + 1)) / a,
        1
    )
    
    # (3) Compute partial gradient w.r.t the weights
    # Shape: (prev_layer, current_layer)
    next_grad_expanded = jnp.expand_dims(next_grad, axis=0)  # Shape: (1, current_layer)
    z_grad = weight_res * next_grad_expanded
    
    # (4) Compute full gradient w.r.t the weights
    # Shape: (prev_layer, current_layer)
    x = all_neuron_states.input_residuals  # Shape (prev_layer,)
    x_reshaped = x[..., jnp.newaxis]  # Shape becomes (prev_layer, 1)
    weight_grad = x_reshaped * z_grad  # (prev_layer, current_layer)
    
    return weight_grad, weight_res

@partial(jax.jit, static_argnames=['params', 'layer_idx'])
def back_prop(params, all_neuron_states, next_grad, layer_idx):
    """
    Backpropagation for a batch of samples.
    
    Args:
        params: Network parameters
        all_neuron_states: Neuron states for batch, stacked
        next_grad: Gradient from next layer, shape (batch, current_layer)
        layer_idx: Current layer index
    
    Returns:
        mean_weight_grad: Average weight gradient, shape (1, prev_layer, current_layer)
        th_grad: Threshold gradient, shape (current_layer,)
        weight_res: Weight residuals for batch, shape (batch, prev_layer, current_layer)
    """
    # Compute gradients for each sample in batch
    weight_grad, weight_res = jax.vmap(
        compute_full_bpp, in_axes=(None, 0, 0, None)
    )(params, all_neuron_states, next_grad, layer_idx)
    # Shape: (batch, prev_layer, current_layer)
    
    # Average over batch
    mean_weight_grad = jnp.mean(weight_grad, axis=0)  # (prev_layer, current_layer)
    mean_weight_grad = jnp.expand_dims(mean_weight_grad, axis=0)  # (1, prev_layer, current_layer)
    
    # Compute threshold gradient
    layer_activity = jnp.where(all_neuron_states.layer_activity > 0, 1, 0)
    th_grad = -jnp.mean(next_grad * layer_activity, axis=0)  # Shape: (current_layer,)
    
    # Apply sigmoid derivative for threshold updates
    thresholds = all_neuron_states.thresholds[0]  # Batch shares same thresholds
    th_grad = th_grad * thresholds * (thresholds - 1)
    
    return mean_weight_grad, th_grad, weight_res

def accuracy(batch_number, outputs, y, iterations, print_results=False):
    """
    Compute accuracy for a batch.
    
    Args:
        batch_number: Current batch index
        outputs: Network outputs, shape (batch, num_classes)
        y: True labels, shape (batch,)
        iterations: Number of iterations per sample
        print_results: Whether to print debug info
    
    Returns:
        valid_y: Valid labels (excluding padding)
        batch_correct: Number of correct predictions
    """
    # Get predictions (indices of max values)
    predictions = jnp.argmax(outputs, axis=-1)
    
    # Calculate accuracy for this batch
    valid_mask = y != -1
    valid_y = y[valid_mask]
    valid_predictions = predictions[valid_mask]
    
    batch_correct = jnp.sum(valid_predictions == valid_y)
    
    if print_results:
        jax.debug.print(
            "Batch {}: Predictions: {}, True: {}, Iterations avg: {}, Correct: {}/{}, last network output: {}",
            batch_number, valid_predictions, valid_y, jnp.mean(iterations),
            batch_correct, valid_y.shape[0], outputs[-1]
        )
    
    return valid_y, batch_correct

# ============================================================================
# region DATA LOADING
# ============================================================================
def load_dataset(dataset: str, batch_size: int, shuffle=False):
    """
    Load your dataset.
    
    Args:
        filepath: Path to your dataset file (or dataset name like 'mnist')
        batch_size: Batch size
        
    Returns:
        (train_data, val_data, test_data, num_train_batches, num_val_batches, num_test_batches)
    """
    try:
        from dataset_helpers.mnist_helper import mnist_loader_manual
        from dataset_helpers.shd_helper import torch_SHD_loader
    except ImportError:
        print("Error: Could not import mnist_loader_manual")
        print("Make sure dataset_helpers/mnist_helper.py is in your path")
        return None, None, None, (0, 0, 0), 0
    
    dataset = dataset.lower() 
    match dataset:
        case "mnist":
            loader = mnist_loader_manual
        case "shd":
            loader = torch_SHD_loader
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")
        
    # Load the data
    (train_loader, total_train_batches), \
    (val_loader, total_val_batches), \
    (test_loader, total_test_batches), \
    max_nonzero = loader(batch_size, shuffle=shuffle)
    
    total_batches = (total_train_batches, total_val_batches, total_test_batches)
    
    return train_loader, val_loader, test_loader, total_batches, max_nonzero

# ============================================================================
# region MAIN EXECUTION
# ============================================================================

def train(network: EventBasedNN, params: Params, key, train_loader, val_loader,
          total_batches: Tuple[int, int, int], optimizer_name: str = "adam", mesh=None):
    """
    Train the network.
    
    Args:
        network: EventBasedNN instance
        params: Network parameters
        key: Random key
        train_loader: Training data loader (iterator)
        val_loader: Validation data loader (iterator)
        total_batches: Tuple of (train_batches, val_batches, test_batches)
        optimizer_name: Name of optimizer to use
    
    Returns:
        Training history
    """
    # Initialize optimizers
    if optimizer_name == "adam":
        weight_solver = optax.adam(learning_rate=params.learning_rate)
    elif optimizer_name == "sgd":
        weight_solver = optax.sgd(learning_rate=params.learning_rate)
    elif optimizer_name == "rmsprop":
        weight_solver = optax.rmsprop(learning_rate=params.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    th_solver = optax.adam(learning_rate=params.threshold_lr)
    
    # Initialize optimizer states
    opt_states = [weight_solver.init(w) for w in network.weights]
    th_opt_states = []
    for i in range(1, network.num_layers - 1):  # Only hidden layers need threshold updates
        th_opt_states.append(
            th_solver.init(jax.scipy.special.logit(network.empty_neuron_states[i].thresholds))
        )
    print(f"initial th_opt_state length: {len(th_opt_states)}")

    # Setup sharding if mesh provided
    if mesh is not None:
        _, batch_sharding, label_sharding_1d, label_sharding_2d = setup_mesh_and_sharding()
        
        # JIT compile train_step with sharding annotations
        @jax.jit
        def sharded_train_step(key, batch_x, batch_y_onehot, opt_states, th_opt_states):
            return network.train_step(
                key, batch_x, batch_y_onehot, opt_states, th_opt_states,
                (weight_solver, th_solver)
            )
    else:
        batch_sharding = None
        label_sharding_1d = None
        label_sharding_2d = None
        sharded_train_step = network.train_step

    # Training history
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    all_iterations = []
    
    print(f"Starting training with {optimizer_name} optimizer...")
    
    for epoch in tqdm(range(params.num_epochs), desc="Epochs"):
        key, epoch_key = jax.random.split(key)
        
        epoch_loss = []
        epoch_correct = 0
        epoch_total = 0
        epoch_iters = []
        
        # Create batch iterator for this epoch
        batch_iterator = iter(train_loader)
        
        # Training loop
        for batch_idx in tqdm(range(total_batches[0]), desc=f"Epoch {epoch+1}", leave=False):
            key, batch_key = jax.random.split(key)
            
            # Get batch from your dataloader
            batch_x, batch_y = next(batch_iterator)
            batch_x = jnp.array(batch_x)
            batch_y = jnp.array(batch_y)
            
            # Pad incomplete batches to be divisible by num_devices
            if mesh is not None:
                num_devices = len(mesh.devices)
                current_batch_size = batch_x.shape[0]
                
                if current_batch_size % num_devices != 0:
                    pad_size = num_devices - (current_batch_size % num_devices)
                    # Pad batch_x - replicate last sample
                    pad_x = jnp.repeat(batch_x[-1:], pad_size, axis=0)
                    batch_x = jnp.concatenate([batch_x, pad_x], axis=0)
                    
                    # Pad batch_y with -1 (invalid label marker)
                    if len(batch_y.shape) == 1:
                        pad_y = jnp.full((pad_size,), -1, dtype=batch_y.dtype)
                    else:
                        pad_y = jnp.zeros((pad_size, batch_y.shape[1]), dtype=batch_y.dtype)
                    batch_y = jnp.concatenate([batch_y, pad_y], axis=0)

            # Shard data across devices if using parallelization
            if mesh is not None:
                with mesh:
                    batch_x = jax.device_put(batch_x, batch_sharding)
                    if len(batch_y.shape) == 1:
                        batch_y = jax.device_put(batch_y, label_sharding_1d)
                    else:
                        batch_y = jax.device_put(batch_y, label_sharding_2d)
            
            # Convert labels to one-hot if needed
            if len(batch_y.shape) == 1:
                batch_y_onehot = jax.nn.one_hot(batch_y, params.layer_sizes[-1])
            else:
                batch_y_onehot = batch_y
            
            # Training step
            loss, total_loss, outputs, iterations, opt_states, th_opt_states = network.train_step(
                batch_key, batch_x, batch_y_onehot, opt_states, th_opt_states,
                (weight_solver, th_solver)
            )
            
            # Track metrics
            epoch_loss.append(loss)

            # Gather results from all devices for accuracy computation
            if mesh is not None:
                outputs = jnp.array(outputs)  # Gather from devices
                batch_y = jnp.array(batch_y)
                iterations = jnp.array(iterations)

            valid_y, batch_correct = accuracy(batch_idx, outputs, batch_y, iterations, False)
            epoch_correct += batch_correct
            epoch_total += len(valid_y)
            epoch_iters.append(jnp.mean(iterations))
        
        # Epoch metrics
        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        mean_loss = jnp.mean(jnp.array(epoch_loss))
        mean_iter = jnp.mean(jnp.array(epoch_iters))
        
        train_accuracies.append(train_acc)
        train_losses.append(float(mean_loss))
        all_iterations.append(float(mean_iter))
        
        # Validation
        val_acc, val_iter = evaluate(network, key, val_loader, total_batches[1], params, desc="Validation", mesh=mesh)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch + 1}/{params.num_epochs} - "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
              f"Loss: {mean_loss:.4f}, Avg Iter: {mean_iter:.2f}")
        
        # Early stopping if perfect accuracy
        if train_acc >= 0.9999:
            print("Reached perfect accuracy, stopping training.")
            break
    
    return {
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_losses': train_losses,
        'iterations': all_iterations
    }

def evaluate(network: EventBasedNN, key, data_loader, num_batches: int, 
             params: Params, desc: str = "Evaluation", mesh=None):
    """
    Evaluate the network on a dataset.
    
    Args:
        network: EventBasedNN instance
        key: Random key
        data_loader: Data loader (iterator)
        num_batches: Number of batches to evaluate
        params: Network parameters
        desc: Description for progress bar
    
    Returns:
        (accuracy, mean_iterations)
    """
    total_correct = 0
    total_samples = 0
    all_iterations = []
    
    # Setup sharding if mesh provided
    if mesh is not None:
        _, batch_sharding, label_sharding_1d, _ = setup_mesh_and_sharding()
    else:
        batch_sharding = None
        label_sharding_1d = None

    # Create batch iterator
    batch_iterator = iter(data_loader)
    
    for batch_idx in tqdm(range(num_batches), desc=desc, leave=False):
        key, batch_key = jax.random.split(key)
        
        # Get batch
        batch_x, batch_y = next(batch_iterator)
        batch_x = jnp.array(batch_x)
        batch_y = jnp.array(batch_y)
        
        # Pad incomplete batches to be divisible by num_devices
        if mesh is not None:
            num_devices = len(mesh.devices)
            current_batch_size = batch_x.shape[0]
            
            if current_batch_size % num_devices != 0:
                pad_size = num_devices - (current_batch_size % num_devices)
                # Pad batch_x - replicate last sample
                pad_x = jnp.repeat(batch_x[-1:], pad_size, axis=0)
                batch_x = jnp.concatenate([batch_x, pad_x], axis=0)
                
                # Pad batch_y with -1 (invalid label marker)
                pad_y = jnp.full((pad_size,), -1, dtype=batch_y.dtype)
                batch_y = jnp.concatenate([batch_y, pad_y], axis=0)
                
        # Shard data across devices if using parallelization
        if mesh is not None:
            with mesh:
                batch_x = jax.device_put(batch_x, batch_sharding)
                batch_y = jax.device_put(batch_y, label_sharding_1d)

        # Forward pass
        outputs, iterations, _ = network.forward(batch_key, batch_x, grad=False)
        
        # Gather results from all devices
        if mesh is not None:
            outputs = jnp.array(outputs)
            batch_y = jnp.array(batch_y)
            iterations = jnp.array(iterations)

        # Compute accuracy
        valid_y, batch_correct = accuracy(batch_idx, outputs, batch_y, iterations, False)
        total_correct += batch_correct
        total_samples += len(valid_y)
        all_iterations.append(jnp.mean(iterations))
    
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    mean_iter = jnp.mean(jnp.array(all_iterations)) if all_iterations else 0.0
    
    return float(acc), float(mean_iter)

# ============================================================================
# region SHARDING
# ============================================================================

# Set up device mesh for parallelization
def setup_mesh_and_sharding(num_devices=None):
    """
    Setup mesh and sharding for parallelization across CPU cores.
    
    Args:
        num_devices: Number of devices to use. If None, uses all available.
    
    Returns:
        mesh: Device mesh
        batch_sharding: Sharding for batch dimension
    """
    devices = jax.local_devices()
    if num_devices is not None:
        devices = devices[:num_devices]
    
    num_devices = len(devices)
    print(f"Using {num_devices} devices for parallelization")
    
    # Create a 1D mesh along the batch dimension
    mesh = Mesh(devices, axis_names=('batch',))
    
    # Create sharding that splits batch dimension across devices
    # For inputs: (batch, seq_len, features) -> shard batch dimension
    batch_sharding = NamedSharding(mesh, P('batch', None, None))
    
    # For labels: (batch,) or (batch, num_classes)
    label_sharding_1d = NamedSharding(mesh, P('batch',))
    label_sharding_2d = NamedSharding(mesh, P('batch', None))
    
    return mesh, batch_sharding, label_sharding_1d, label_sharding_2d

# ============================================================================
# region Main
# ============================================================================
def main():
    # Set random seed
    random_seed = 42
    key = jax.random.key(random_seed)
    
    mesh, batch_sharding, _, _ = setup_mesh_and_sharding()  # Uses all available CPU cores
    # batch_sharding = sharding.reshape(-1, 1, 1)  # Shard along batch dimension
    # print(batch_sharding)

    dataset = "mnist"
    # dataset = "shd"
    
    match dataset:
        case "mnist":
            layer_sizes = (784, 128, 128, 10)
        case "shd":
            layer_sizes = (700, 128, 20)

    batch_size = 126
    
    print(f"Network architecture: {layer_sizes}")
    print(f"Batch size: {batch_size}")
    
    # Load dataset first to get max_nonzero
    train_loader, val_loader, test_loader, total_batches, max_nonzero = load_dataset(
        dataset, batch_size
    )
    
    if train_loader is None:
        print("Failed to load dataset. Exiting.")
        return None
    
    print(f"Train batches: {total_batches[0]}, Val batches: {total_batches[1]}, Test batches: {total_batches[2]}")
    print(f"Max non-zero elements per sample: {max_nonzero}")
    
    # Create parameters
    params = Params(
        dataset=dataset,
        random_seed=random_seed,
        layer_sizes=layer_sizes,
        init_thresholds=0.0,
        num_epochs=1,
        learning_rate=0.0001,
        batch_size=batch_size,
        load_file=False,
        shuffle_activations=False,
        restrict=(0.0, 0.0, 0.0, 0.0),
        firing_nb=128,
        sync_rate=1,
        max_nonzero=max_nonzero,
        shuffle_input=False,
        threshold_lr=0.00,
        sparsity_impact=(0.0, 0.0, 0.0, 0.0),
        rerun="",
        async_layer=-1,
        history_size=0
    )
    
    print(f"\nNetwork Parameters:")
    print(f"  Learning rate: {params.learning_rate}")
    print(f"  Threshold LR: {params.threshold_lr}")
    print(f"  Firing number: {params.firing_nb}")
    print(f"  Sync rate: {params.sync_rate}")
    print(f"  Sparsity impact: {params.sparsity_impact}")
    
    # Initialize weights with proper shapes
    # Note: weights[i] has shape (layer_sizes[i+1], layer_sizes[i])
    print("\nInitializing network weights...")
    # key, *subkeys = jax.random.split(key, len(layer_sizes))
    weights = [jnp.zeros((layer_sizes[-1], layer_sizes[0]))]
    key, subkey = jax.random.split(key) 
    keys = jax.random.split(key, len(layer_sizes))

    for i in range(1, len(layer_sizes)):
        # # Xavier initialization
        # fan_in = layer_sizes[i-1]
        # bound = jnp.sqrt(2.0 / fan_in)
        # w = jax.random.uniform(
        #     subkeys[i-1], (layer_sizes[i-1], layer_sizes[i]),
        #     minval=-bound, maxval=bound
        # )
        # weights.append(w)
        # print(f"  Layer {i}: weights shape {w.shape}")
        w_key, b_key = jax.random.split(keys[i])
        weights.append(1e-2 * jax.random.normal(w_key, (layer_sizes[i-1], layer_sizes[i])))

    # Create network
    network = EventBasedNN(params, weights)
    print(f"\nNetwork created with {network.num_layers} layers")
    
    # Train the network
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    # history = train(network, params, key, train_loader, val_loader, 
    #                   total_batches, optimizer_name="adam", sharding=None)
    with mesh:
        history = train(network, params, key, train_loader, val_loader, 
                        total_batches, optimizer_name="adam", mesh=mesh)
    
    end_time = time.time()
    
    print(f"\n" + "="*70)
    print(f"Training Complete!")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Final train accuracy: {history['train_accuracies'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_accuracies'][-1]:.4f}")
    print("="*70 + "\n")
    
    # Test evaluation
    print("Evaluating on test set...")
    # test_acc, test_iter = evaluate(network, key, test_loader, total_batches[2], params, desc="Test")
    with mesh:
        test_acc, test_iter = evaluate(network, key, test_loader, total_batches[2], params, desc="Test", mesh=mesh)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Average iterations: {test_iter:.2f}")
    
    # Plot training history
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        epochs = list(range(1, len(history['train_accuracies']) + 1))
        
        # Accuracy plot
        axes[0].plot(epochs, history['train_accuracies'], 'o-', label='Train')
        axes[0].plot(epochs, history['val_accuracies'], 's-', label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy vs Epoch')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(epochs, history['train_losses'], 'o-')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss vs Epoch')
        axes[1].grid(True)
        
        # Iterations plot
        axes[2].plot(epochs, history['iterations'], 'o-')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Avg Iterations')
        axes[2].set_title('Average Iterations vs Epoch')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print(f"\nTraining history saved to training_history.png")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plot")
    
    return network, history


if __name__ == "__main__":
    outputs = main()