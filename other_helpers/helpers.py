import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import dataclasses
from jax import custom_jvp, jit
from functools import partial
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


#region ACCURACY
def accuracy(batch_number, outputs, y, iterations, rank=None, print=False):
    # Get predictions (indices of max values)
    predictions = jnp.argmax(outputs, axis=-1)
    
    # Calculate accuracy for this batch
    valid_mask = y != -1
    valid_y = y[valid_mask]
    valid_predictions = predictions[valid_mask]

    batch_correct = jnp.sum(valid_predictions == valid_y)
    if print:
        jax.debug.print("Batch {}, rank {}:  Predictions: {}, True: {}, Iterations avg: {}, Correct: {}/{}, last network output: {}",
                batch_number, rank, valid_predictions, valid_y, jnp.mean(iterations), batch_correct, valid_y.shape[0], outputs[-1])
    return valid_y, batch_correct

#region Activation_func
# @custom_jvp # If thresholds == 0 then this behaves as a ReLu activation function 
@jit
def activation_func(thresholds, activations):
    # return jax.nn.relu(activations)
    return jnp.where(activations > thresholds, activations, 0.0)

#region keep_top_k
@partial(jax.jit, static_argnames=['k', 'apply_abs', 'max_kernel',])
def keep_top_k(x, k, apply_abs=False, max_kernel=None):
    '''
    Return the top-k values in the input x in the same shape and masks all other entries to 0
    If multiple values are tied it returns the ones with the smallest indices

    :param x: Input array/matrix, either the neuron states or a row of weights 
    :param k: The exact number of highest values that you want to keep
    :param apply_abs: Get the top-k according to the absolute values of x and returns the values in their original sign
    :param max_kernel: Currently serves as a flag to determine whether the input x needs to be flattened
    '''
    if k < 0:
        return x

    x = jnp.atleast_1d(x)
    k = min(k, x.size)
    x_flat = x.flatten()

    if apply_abs:
        x_flat = jnp.abs(x_flat)

    _, top_indices = jax.lax.top_k(x_flat, k)

    # Create a mask with 1s at top-k indices, 0 elsewhere
    mask = jnp.zeros(x_flat.size)
    mask = mask.at[top_indices].set(1)

    out = x_flat * mask
    return out.reshape(x.shape)

#region output_vector_to_event 
@partial(jax.jit, static_argnames=['params', 'max_len'])
def output_vector_to_event(key, arr: jnp.ndarray, params, max_len):
    '''
    Processed the output of a layer from (1d array) to (2d array) with [(neuron idx, value)]
    value == 0 are filled with index==-2
    '''
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

@jax.tree_util.register_pytree_node_class
class NeuronStates:
    '''
    Shapes are referenced for a layer with weights of shape: (784, 128)

    values: jnp.ndarray             # Current state of the neurons in the layer, shape: (layer_sizes[rank],) __ (128,)
    thresholds: jnp.float32         # An array of thresholds, one per neuron, shape: (layer_sizes[rank],) __ (128,)
    input_residuals: jnp.ndarray    # Sum of all input neurons, shape: (layer_sizes[rank-1],) __ (784,)
    output_residuals: jnp.ndarray   # Sum of all output neurons, shape: (layer_sizes[rank],) __ (128,)
    input order                     # Set input neuron to the iteration at which the input is received to record the order of input received, shape: (layer_sizes[rank-1],) __ (784,)
    input activity                  # Count the number of times an input neuron fired, shape: (layer_sizes[rank-1],) __ (784,)
    layer activity                  # Count the number of times a neuron activated in this layer, only used for restrict parameter and threshold, shape: (layer_sizes[rank],) __ (128,)
    output activity                 # For each input neuron stores the hidden neurons that fire, shape: (layer_sizes[rank-1], layer_sizes[rank]) __ (784, 128)  
    
    ____ Convolution fields
    weights_shape                   # Shape of the weights
    is_conv                         # Is convolution layer               
    '''
    # Define which fields are always arrays (JAX leaves) vs static scalars
    _ARRAY_FIELDS = frozenset({
        "values", "bias", "thresholds", "input_residuals", "output_residuals",
        "input_order", "input_activity", "layer_activity", "output_activity",
        "last_sent_iteration", "input_vector", "output_vector",
    })
    _OPTIONAL_ARRAY_FIELDS = frozenset({
        "sync_rate_vector", "recurrent_weight", "values_history",
        "history_index", "weights_shape",
    })
    _STATIC_FIELDS = frozenset({"is_conv"})  # non-array, treated as aux

    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)
        self._fields = dict(fields)  # preserve insertion order

    def tree_flatten(self):
        # Split into leaves (arrays) and statics (aux_data)
        array_keys = []
        array_vals = []
        static_items = {}

        for k, v in self._fields.items():
            if k in self._STATIC_FIELDS:
                static_items[k] = v
            else:
                array_keys.append(k)
                array_vals.append(v)

        # aux_data is a hashable tuple of (ordered keys, static values)
        aux_data = (tuple(array_keys), tuple(sorted(static_items.items())))
        return array_vals, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        array_keys, static_items = aux_data
        fields = dict(zip(array_keys, children))
        fields.update(dict(static_items))
        return cls(**fields)

    def replace(self, **updates):
        new_fields = {**self._fields, **updates}
        return NeuronStates(**new_fields)

    def __getattr__(self, name):
        # Fallback for fields not set — return None for optional fields
        if name.startswith("_"):
            raise AttributeError(name)
        return None
    
#region Params
@dataclasses.dataclass(frozen=True)
class BaseParams:
    """
    Universal hyperparameters shared by every async network file.

    Each file defines its own frozen dataclass that inherits from this one and
    adds only the fields that file actually uses.  Example:

        @dataclasses.dataclass(frozen=True)
        class Params(BaseParams):
            fptt_parts: int = 1
            fptt_alpha: float = 0.1

    JAX JIT treats frozen dataclasses as static (hashable), so subclasses work
    identically to the original monolithic Params — no changes to JIT decorators
    or @partial calls are needed.

    rerun_init() works with any subclass: it reads the base fields from the JSON,
    then calls dataclasses.replace(new_params, **base_overrides) so that subclass
    fields are preserved automatically.
    """
    dataset: str
    random_seed: int
    layer_sizes: tuple[int, ...]
    init_thresholds: float       # Starting thresholds
    num_epochs: int
    learning_rate: float
    batch_size: int
    load_file: bool
    shuffle_activations: bool    # Shuffle the activations in the network
    restrict: float | tuple[float]   # Soft reset multiplier applied to fired neuron values
    firing_nb: int | tuple[int]      # Max neurons that can fire per event per layer
    sync_rate: int | tuple[int]      # Events to accumulate before a neuron can fire again
    max_nonzero: int
    shuffle_input: bool          # Shuffle the input data
    threshold_lr: float
    sparsity_impact: float | tuple[float]
    w_reg: float                 # L2 weight regularisation coefficient
    rerun: str
    top_weights: int             # Top-k weights by absolute value used per neuron (-1 = all)
    history_size: int = 0        # Output states to keep for history plots
    use_bias: bool = False
    dataset_file: str | None = None
    collapse_units: bool = True
    preserve_exact_times: bool = False

# Backwards-compatible alias: files that haven't switched yet can still import
# `Params` and get a class that already has all the old optional fields.
@dataclasses.dataclass(frozen=True)
class Params(BaseParams): #TODO Make params a file-local subclass of BaseParams in each file and remove the legacy Params class
    """
    Legacy all-in-one params class kept for backwards compatibility.
    Prefer defining a file-local subclass of BaseParams instead.
    """
    # Additional inference logic parameters
    exploration_rate: float = 0.0
    trace_event_timing: bool = False

    # CNN parameters
    max_kernel: int | None = None
    flat_layer_sizes: tuple[int, ...] | None = None

    # Recurrent parameters (for RTRL and FPTT)
    use_tanh: bool = False
    exact_rtrl: bool = False
    recurrence: tuple[int, ...] | None = None

    # FPTT (Forward Propagation Through Time) parameters
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

#region RERUN
def rerun_init(data_file_path, mpi_config, new_params, override_params=None):
    '''
    Rerun from an existing file by replacing specified fields with values from new_params.
    
    Args:
        data_file_path: Path to the JSON file containing stored parameters
        mpi_config: MPI configuration object
        new_params: Params object containing new parameter values
        override_params: List of parameter names to override from new_params.
                        If None or empty, all parameters are loaded from the file.
                        
    Available parameter names to override:
        - 'shuffle_activations': Whether to shuffle activations in hidden layers
        - 'shuffle_input': Whether to shuffle input values
        - 'firing_nb': Maximum number of neurons allowed to fire per layer
        - 'sync_rate': Number of input values needed before firing
        - 'batch_size': Training batch size
        - 'learning_rate': Learning rate for optimizer
        - 'init_thresholds': Initial threshold values for neurons
        - 'restrict': Reset rate for each layer
        - 'sparsity_impact': Sparsity loss impact per layer
        - 'w_reg': Weight regularization impact
        - 'threshold_lr': Threshold learning rate
        - 'top_weights': Top-k weights to use
        - 'history_size': Number of output states to keep for plotting
    
    Example:        
        # Override multiple training parameters
        params, weights, thresholds = rerun_init(
            "path/to/model.json",
            mpi_config,
            new_params,
            override_params=['learning_rate', 'batch_size', 'sparsity_impact']
        )
    '''
    layer_idx = mpi_config.layer_idx
    last_layer = mpi_config.last_layer_idx

    with open(data_file_path, "r") as f:
        stored_data = json.load(f)

    stored_dataset = stored_data.get("dataset", new_params.dataset)
    assert stored_dataset == new_params.dataset, f"Rerun can only be used on the same dataset, got {stored_dataset} and {new_params.dataset}"

    load_file = stored_data["loadfile"]
    layer_sizes = list_to_tuple_deep(stored_data["layer_sizes"])
    # assert layer_sizes == new_params.layer_sizes, f"Network structure must be the same to rerun, got {layer_sizes} and {new_params.layer_sizes}"
    # Replacement for the assert line:
    if not check_core_structure(layer_sizes, new_params.layer_sizes):
        raise AssertionError(
            f"Network structure mismatch! Core components must match. "
            f"Got: \n{new_params.layer_sizes} \nvs \n{layer_sizes}"
        )

    # Initialize override_params as empty set if None
    if override_params is None:
        override_params = set()
    else:
        override_params = set(override_params)  # Convert to set for faster lookup
    
    # Helper function to decide whether to use new_params or stored value
    def get_param(param_name, stored_key=None, default=None, transform=None):
        """
        Get parameter value from either new_params or stored_data.
        
        Args:
            param_name: Name of the parameter in new_params object
            stored_key: Key in stored_data dictionary
            default: Default value if key not found in stored_data
            transform: Optional function to transform stored value (e.g., tuple conversion)
        
        Returns:
            Parameter value from new_params if param_name in override_params, else from stored_data
        """
        if stored_key is None:
            stored_key = param_name
        
        if param_name in override_params:
            return getattr(new_params, param_name)
        else:
            value = stored_data.get(stored_key, default)
            if transform is not None and value is not None:
                value = transform(value)
            return value
    
    def tuple_if_sequence(value):
        """
        Keep scalar hyperparameters as scalars while normalizing saved lists to tuples.
        Older result files sometimes store values like `restrict: 1` instead of `(1, ...)`.
        """
        if isinstance(value, (list, tuple, np.ndarray)):
            return tuple(value)
        return value

    # Extract all parameters using the helper function
    shuffle_activations_val = get_param(
        'shuffle_activations',
        'shuffle activations',
        default=new_params.shuffle_activations,
    )
    shuffle_input_val = get_param(
        'shuffle_input',
        'shuffle input',
        default=new_params.shuffle_input,
    )

    firing_nb_val = get_param('firing_nb', 'firing number')    
    sync_rate_val = get_param('sync_rate', 'synchronization rate')
    batch_size_val = get_param('batch_size')
    learning_rate_val = get_param('learning_rate', 'learning rate')

    # Special handling for init_thresholds
    if 'init_thresholds' in override_params:
        init_thresholds_val = new_params.init_thresholds
    else:
        init_thresholds_val = extract_scalar(stored_data["thresholds"]["thresholds_1"])
    
    restrict_val = get_param('restrict', transform=tuple_if_sequence)
    sparsity_impact_val = get_param('sparsity_impact','sparsity impact', default=0, transform=tuple_if_sequence)
    w_reg_val = get_param('w_reg', 'weight regularization', default=0.0)
    threshold_lr_val = get_param('threshold_lr', 'threshold lr')
    top_weights_val = get_param('top_weights', 'top weights', default=-1)
    history_size_val = get_param('history_size', default=0)

    # Convert firing_nb to tuple if it's a list
    if isinstance(firing_nb_val, list):
        firing_nb_val = tuple(firing_nb_val)

    # Build the set of BaseParams fields to restore from the JSON, then call
    # dataclasses.replace(new_params, ...) so that any subclass-specific fields
    # (fptt_*, rhythm_*, etc.) are preserved from new_params unchanged.
    base_overrides = dict(
        dataset=new_params.dataset,
        dataset_file=stored_data.get("dataset_file", new_params.dataset_file),
        collapse_units=stored_data.get("collapse_units", new_params.collapse_units),
        preserve_exact_times=stored_data.get("preserve_exact_times", new_params.preserve_exact_times),
        random_seed=new_params.random_seed,
        layer_sizes=layer_sizes,
        init_thresholds=init_thresholds_val,
        num_epochs=new_params.num_epochs,
        learning_rate=learning_rate_val,
        batch_size=batch_size_val,
        load_file=load_file,
        shuffle_activations=shuffle_activations_val,
        restrict=restrict_val,
        firing_nb=firing_nb_val,
        sync_rate=sync_rate_val,
        max_nonzero=new_params.max_nonzero,
        shuffle_input=shuffle_input_val,
        threshold_lr=threshold_lr_val,
        sparsity_impact=sparsity_impact_val,
        w_reg=w_reg_val,
        rerun=data_file_path,
        top_weights=top_weights_val,
        history_size=history_size_val,
    )
    # Only pass fields that actually exist on this subclass (guards against
    # files that use BaseParams directly without the optional base fields)
    all_fields = {f.name for f in dataclasses.fields(new_params)}
    base_overrides = {k: v for k, v in base_overrides.items() if k in all_fields}
    params = dataclasses.replace(new_params, **base_overrides)
    
    # Load weights and thresholds
    threshold_dict = stored_data["thresholds"]
    weights_dict = stored_data["weights"]

    if layer_idx > 0:
        weights = jnp.array(weights_dict["layer_"+str(layer_idx)])
        if layer_idx < last_layer:
            thresholds = jnp.array(threshold_dict["thresholds_"+str(layer_idx)])
        else:
            thresholds = jnp.zeros(layer_sizes[layer_idx])
    else:
        l = layer_sizes[layer_idx]
        if isinstance(l, int) or len(l) == 1:
            weights = thresholds = jnp.zeros((layer_sizes[0]))
        else:
            weights = thresholds = jnp.zeros((1,1,1,1))
            
    return params, weights, thresholds

def check_core_structure(ls1, ls2):
    # Check whether two network structures match, for CNN we only check the channel dimension, the filter sizes and the pooling
    if len(ls1) != len(ls2):
        return False
    
    for e1, e2 in zip(ls1, ls2):
        # If the element has multiple parts (is a tuple/list)
        if hasattr(e1, '__len__') and len(e1) > 1:
            # Check first and second elements (e.g., in_channels and height)
            if e1[0] != e2[0] or e1[1] != e2[1] or (len(e1) >= 5 and len(e2) >= 5 and e1[4] != e2[4]):
                return False
        else:
            # If it's a single value (like [128]), just compare the whole thing
            if e1 != e2:
                return False
    return True

def list_to_tuple_deep(obj):
    if isinstance(obj, list):
        return tuple(list_to_tuple_deep(item) for item in obj)
    return obj

def extract_scalar(x):
    if np.isscalar(x):
        return x
    if isinstance(x, (list, tuple)):
        return extract_scalar(x[0])

# region SAVING DATA
def prepare_result_payload(size, network, mode, all_epoch_accuracies, all_validation_accuracies, test_accuracy, execution_time, all_iteration_mean, weights_dict, all_loss, thresholds_dict, optiname, network_type, extra_fields=None):
    filename_add_on = ""
    if optiname is not None:
        filename_add_on = f"_{optiname}_"
    
    if hasattr(network, "params"):
        params = network.params
        filename_nn = network.filename
    else:
        params = network
        filename_nn = f"_b{params.batch_size}_" + "_".join(map(str, params.layer_sizes)) 

    # Choose the saving folder
    if mode == "train":
        result_dir = os.path.join("network_results", params.dataset, "training", network_type)
        filename = f"{params.random_seed}" + f"_ep{params.num_epochs}" + filename_nn
        iteration_mean_payload = np.array(all_iteration_mean).tolist()
    elif mode == "inference":
        result_dir = os.path.join("network_results", params.dataset, "inference", network_type)
        filename = f"{params.random_seed}" + f"_load{params.load_file}" + filename_nn
        iteration_mean_payload = np.array(all_iteration_mean).flatten().tolist()
    else:
        print("Wrong mode for storing data choose 'train' or 'inference'. No data is saved")
        return None, None

    train_accuracy = float(all_epoch_accuracies[-1])
    val_accuracy = float(all_validation_accuracies[-1])   
    test_accuracy = float(test_accuracy)    

    jax.debug.print(
        "Final Training Accuracy: {train:.2f}%, Final Validation Accuracy: {val:.2f}%, Test Accuracy: {test:.2f}%",
        train=train_accuracy * 100 if train_accuracy != -1 else jnp.nan,
        val=val_accuracy * 100 if val_accuracy != -1 else jnp.nan,
        test=test_accuracy * 100 if test_accuracy != -1 else jnp.nan,
    )
    
    accuracy = -1.0
    for acc in [train_accuracy, val_accuracy, test_accuracy]:
        if acc >= 0:
            accuracy = acc

    # Set up file path and changing the name if same name exists already 
    # filename = filename_header + "_".join(map(str, params.layer_sizes)) 
    filename += f"_acc{accuracy:.4f}" 
    # if best:
    #     filename = "best_" + filename         

    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, filename) + filename_add_on
    
    if os.path.exists(result_path + ".json"):
        index = 1
        while True:
            new_result_path = result_path + f"({index})"
            if os.path.exists(new_result_path + ".json"):
                index+=1
            else:
                result_path = new_result_path
                break                
    
    result_data = {
        "dataset": params.dataset,
        "dataset_file": params.dataset_file,
        "collapse_units": params.collapse_units,
        "preserve_exact_times": params.preserve_exact_times,
        "time": float(execution_time),
        "loadfile": params.load_file,
        "shuffle activations": params.shuffle_activations,
        "shuffle input": params.shuffle_input,
        "rerun": params.rerun,
        "processes": size,
        "firing number": params.firing_nb,
        "synchronization rate": params.sync_rate,
        "top weights": params.top_weights,
        "restrict": params.restrict,
        "sparsity impact": params.sparsity_impact,
        "weight regularization": params.w_reg,
        "threshold init": params.init_thresholds,
        "threshold lr": params.threshold_lr,
        "test accuracy": test_accuracy,
        "batch_size": params.batch_size,
        "learning rate": params.learning_rate,
        "use_bias": params.use_bias,
        "layer_sizes": params.layer_sizes,
        "training accuracy": np.array(all_epoch_accuracies).tolist(),
        "validation accuracy": np.array(all_validation_accuracies).tolist(),
        "iterations mean": iteration_mean_payload,
        "loss": [float(loss) for loss in all_loss],
        "thresholds": thresholds_dict,
        "weights": weights_dict
    }
    if extra_fields:
        items = list(result_data.items())
        for i, (k, _) in enumerate(items):
            if k == "processes":
                items = items[:i+1] + list(extra_fields.items()) + items[i+1:]
                break
        else:
            items = items + list(extra_fields.items())
        result_data = dict(items)

    return result_path, result_data


def store_result_artifacts(result_path, mode, all_epoch_accuracies, all_validation_accuracies, test_accuracy, all_loss, all_iteration_mean, all_history=None, total_batches=None):
    if all_history is not None and len(all_history) > 0 and total_batches is not None:
        store_history(jnp.array(all_history), result_path, total_batches)

    if mode == "train":
        epochs = [i + 1 for i in range(len(all_epoch_accuracies))]        
        # Plot accuracies and loss values
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(epochs, all_epoch_accuracies, 'o-', label='Training Accuracy')
        ax1.plot(epochs, all_validation_accuracies, 's-', label='Validation Accuracy')
        ax1.axhline(test_accuracy, color='red', linestyle='--', label=f'Test Accuracy ({test_accuracy:.4f})')
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


def store_training_data(size, network, mode, all_epoch_accuracies, all_validation_accuracies, test_accuracy, execution_time, all_iteration_mean, weights_dict, all_loss, thresholds_dict, optiname, network_type, all_history=None, total_batches=None, extra_fields=None):
    result_path, result_data = prepare_result_payload(
        size,
        network,
        mode,
        all_epoch_accuracies,
        all_validation_accuracies,
        test_accuracy,
        execution_time,
        all_iteration_mean,
        weights_dict,
        all_loss,
        thresholds_dict,
        optiname,
        network_type,
        extra_fields=extra_fields,
    )
    if result_path is None:
        return

    with open(result_path + ".json", "w") as f:
        json.dump(result_data, f, indent=4)

    print(f"Results saved to {result_path}")
    store_result_artifacts(
        result_path,
        mode,
        all_epoch_accuracies,
        all_validation_accuracies,
        test_accuracy,
        all_loss,
        all_iteration_mean,
        all_history,
        total_batches,
    )
    return result_path + ".json"

#region HISTORY
def update_history(history, index, new_value):
    # Replace value at current index
    new_history = history.at[index].set(new_value)

    # Increment index and wrap around
    new_index = (index + 1) % history.shape[0]
    return new_history, new_index


def process_history(history, index, target_labels):
    '''
    Reorder the history to index 0
    history shape: (B, T, 10)
    index shape: (B,)

    Compute the accuracy history and the avg rank history
    values_history: (B, T, 10)
    target_labels: (B)
    
    return acc_history (T), avg_rank (T)
    '''
    def reorder_single(h, idx):
        return jnp.roll(h, shift=-idx, axis=0)
    # Vectorize across batch
    values_history = jax.vmap(reorder_single)(history, index)

    def preprocess_history(values_history, target_labels):
        def get_all_max(single_values_history, targets):
            return jax.vmap(lambda v, t: jnp.argmax(v) == t, in_axes=(0, None))(single_values_history, targets)
        
        def get_target_rank(single_values_history, targets):
            def single_history(history, single_target):
                return jnp.sum(history > history[(single_target.astype(int))]) + 1
            return jax.vmap(single_history, in_axes=(0, None))(single_values_history, targets)
        
        # Get the output prediction of all stored steps
        out_history = jax.vmap(get_all_max)(values_history, target_labels)
        
        # Get the rank of the position corresponding to the target value
        correct_target_ranks = jax.vmap(get_target_rank)(values_history, target_labels)
        return out_history, correct_target_ranks

    out_hist, correct_target = preprocess_history(values_history, target_labels)

    # Compute the average value over the batch 
    acc_history = jnp.sum(out_hist, axis=0) / out_hist.shape[0]
    avg_rank = jnp.sum(correct_target, axis=0) / correct_target.shape[0]

    return acc_history, avg_rank

def store_history(all_history, result_path, total_batches):
    all_history = all_history.transpose(1, 0, 2)
    # print("all history shape: ",all_history.shape)

    def flatten_history(history, batch_number):
        # shape (epoch * num_batches, 100) -> (epoch, num_batches, 100)
        T, H = history.shape # T: total iterations, H: history size
        
        assert T % batch_number == 0, f"T={T} must be divisible by batch_number={batch_number}"
        E = T // batch_number # E: epochs
        
        return history.reshape(E, batch_number, H) # (E, batch_number, H)
    
    flat_history = jnp.stack([
                    flatten_history(arr_slice, total_batches)
                    for arr_slice in all_history  # arr is shape (2, T, H)
                    ])
    # print(f"Flattened shape: {flat_history[0].shape}, {flat_history[1].shape}")
    
    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    H = flat_history.shape[-1]
    # assert H == params.history_size, f"History param must match the values got {params.history_size} and {H}"

    # PLot the average of history and rank 
    for epoch in range(flat_history.shape[1]):
        out_history_data = flat_history[0][epoch]         # shape: (data_points, H)
        correct_target = flat_history[1][epoch]         # shape: (data_points, H)
        
        acc = jnp.sum(out_history_data, axis=0) / out_history_data.shape[0]
        avg_rank = jnp.sum(correct_target, axis=0) / correct_target.shape[0]
        
        axes[0].plot(range(H), acc, label=f"Epoch {epoch} ({acc[-1]*100:.2f}%)")
        axes[1].plot(range(H), avg_rank, label=f"Epoch {epoch}")
        

    axes[0].set_xlabel(f"Iteration from {H} iterations before final output")
    axes[0].set_ylabel("Average prediction accuracy")
    axes[0].set_title(f"Correct predictions for last {H} values in the output layer")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel(f"Iteration from {H} iterations before final output")
    axes[1].set_ylabel("Average rank of target value")
    axes[1].set_title(f"Average rank of target for last {H} values in the output layer")
    axes[1].legend()
    axes[1].grid(True)

    print("history saved in ", result_path)
    # Save both plots into one image
    plt.tight_layout()
    plt.savefig(result_path + "_history.png")
    plt.close()
    
# region store to json
def store_data_to_json(filename, data_to_add, label=None):
    '''
    Storing input data, labels and network output for validating the implementation on the Spinnaker2
    '''
    # Step 1: Load existing file or start with an empty list
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        if label is None:   # list mode 
            data = []
        else:               # dict mode
            data = {               
                "outputs": [],
                "labels": []
            }

    # Step 2: Append new data depending on mode
    if label is None:
        # List mode → append the raw data
        data.append(data_to_add)
    else:
        # Dict mode → append to "outputs" and "labels"
        # Make sure keys exist (in case file existed but was missing them)
        if "outputs" not in data:
            data["outputs"] = []
        if "labels" not in data:
            data["labels"] = []
        
        data["outputs"].append(data_to_add)
        data["labels"].append(label)

    # Step 3: Write back to disk
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


# region Load config file
def load_config_with_defaults(config_path: Optional[str] = None, is_cnn: bool = False) -> Dict[str, Any]:
    """
    Load configuration from a YAML file with default values for missing fields.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses all defaults.
    
    Returns:
        Dictionary containing the configuration parameters
    """
    # Default configuration
    default_config = {
        # Dataset selection: 'mnist', 'shd', 'nmnist', 'dvs', 'ncars', 'smnist'
        'dataset': 'mnist',
        'filename': None,
        'collapse_units': True,
        'preserve_exact_times': False,

        'mode': 'training', # Choose either train or inference
        'use_bias': False, # Whether to use bias in the network
        'use_tanh': False, # Whether to apply tanh to the hidden state update
        'exact_rtrl': False, # Use exact RTRL traces (H^3 memory) instead of diagonal approximations
        # Network architecture: tuple of layer sizes (input, hidden1, hidden2, ..., output)
        # Examples:
        #   MNIST: (784, 256, 10) or (28*28, 128, 128, 10)
        #   SHD: (700, 256, 20)
        #   N-MNIST: (34*34*2, 256, 10)
        #   DVS Gesture: (128*128*2, 128, 11) or (64*64*2, 128, 11)
        'layer_sizes': (784, 256, 10),
        'recurrence':(None, None, None), # No recurrence by default
        # Training parameters
        'batch_size': 36,
        'num_epochs': 10,
        'learning_rate': 0.0001,
        'optimizer': 'adam',
        
        # Model loading/saving
        'load_file': False,  # Load pytorch pretrained weights
        'best': False,  # Old variable, not in use anymore
        'rerun': None,  # Path to a previous training JSON to continue/rerun
        "override_params": None,  # List of parameter names to override when rerunning from a previous file

        # Number of MPI processes per layer (tuple, must match number of layers and sum to MPI size)
        # Example: [1, 2, 1] for 3 layers with 4 total processes
        # If None, falls back to uniform split (requires MPI size % nb_layers == 0)
        'processes_per_layer': None,

        # Reset rate for each layer (tuple, must match number of layers)
        # Example: (1, 1, 1) for 3 layers, (2, 2, 1) for different reset rates
        'restrict': None,  # Will be auto-generated as (1,) * len(layer_sizes) if None
        
        # Threshold initialization
        'init_thresholds': 0.0,  # Initial threshold value for neurons
        
        # Neuron behavior parameters
        'shuffle_activations': False,  # Shuffle activations in hidden layers
        'shuffle_input': False,  # Shuffle input values
        'firing_nb': 10000,  # Max number of top values allowed to fire per layer
        'sync_rate': 1,  # Number of input values needed before firing
        
        # Learning and regularization
        'threshold_lr': 0.0,  # Threshold learning rate
        'sparsity_impact': (0.0, 0.0, 0.0, 0.0, 0.0),  # Sparsity loss impact per layer
        'w_reg': 0.0,  # Weight regularization impact
        
        # Advanced features
        'top_weights': -1,  # Top k weights to use (-1 for all)
        'history_size': 0,  # Number of output states to keep for plottingis the l
        'exploration_rate': 0.0,  # Probability of replacing a top-k fired neuron with a random non-top-k non-zero neuron
        'trace_event_timing': False,  # Capture one-batch event timing traces during inference
        'residual_connections': (),  # Residual skip connections: [(src_layer_idx, dst_layer_idx), ...]

        # FPTT parameters
        'cell_type': 'aed',         # "aed" or "minimalrnn"
        'fptt_parts': 1,            # Number of FPTT chunks (1 = no FPTT)
        'fptt_alpha': 0.1,
        'fptt_beta': 0.5,
        'fptt_lambda': 2.0,
        'fptt_rho': 0.0,
        'fptt_clip': 1.0,
        'fptt_warm_epochs': 1,
        'fptt_accumulate_logits': True,
        'fptt_avg_logits': False,
        'fptt_relu_output': False,
    }

    if is_cnn:
        default_config['layer_sizes'] = (
            [1, 28, 28], 
            [3, [3, 3], [1, 1], [1, 1], 'max'], 
            [128], 
            [10]
        )
        
    
    # If no config file provided, return defaults
    if config_path:
        config_file = Path(config_path)

        # Load YAML configuration
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
        with open(config_file, 'r') as f:
            user_config = yaml.safe_load(f)
            
            if user_config:
                default_config.update(user_config)
    
    # Convert all lists to tuples (especially layer_sizes)
    final_config = {k: make_hashable(v) for k, v in default_config.items()}

    if final_config['restrict'] is None:
        final_config['restrict'] = (1,) * len(final_config['layer_sizes'])

    return final_config

def make_hashable(item):
    if isinstance(item, (list, tuple)):
        return tuple(make_hashable(i) for i in item)
    return item

# region parse_unknown_args
def parse_unknown_args_and_overrides_config(unknown, config):
    overrides = {}
    key = None

    for item in unknown:
        if item.startswith("--"):
            key = item.lstrip("-")
        else:
            if key is None:
                raise ValueError(f"Unexpected value {item}")
            overrides[key] = item
            key = None

    for k, v in overrides.items():
        if k in config:
            # Auto-cast to correct type
            config[k] = type(config[k])(v)
        else:
            raise ValueError(f"Unknown config key: {k}")
    return config
