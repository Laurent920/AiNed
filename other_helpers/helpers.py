import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import dataclasses

def accuracy(batch_number, outputs, y, iterations, print):
    # Get predictions (indices of max values)
    predictions = jnp.argmax(outputs, axis=-1)
    
    # Calculate accuracy for this batch
    valid_mask = y != -1
    valid_y = y[valid_mask]
    valid_predictions = predictions[valid_mask]

    batch_correct = jnp.sum(valid_predictions == valid_y)
    if print:
        jax.debug.print("Batch {}: Predictions: {}, True: {}, Iterations avg: {}, Correct: {}/{}, last network output: {}",
                batch_number, valid_predictions, valid_y, jnp.mean(iterations), batch_correct, valid_y.shape[0], outputs[-1])
    return valid_y, batch_correct

#region DATACLASSES
@jax.tree_util.register_pytree_node_class
class NeuronStates:
    def __init__(self, 
                 values, 
                 thresholds, 
                 input_residuals, 
                 input_order, 
                 input_activity, 
                 layer_activity, 
                 output_activity, 
                 last_sent_iteration, 
                 input_vector, 
                 output_vector, 
                 values_history=None,
                 history_index=None,
                 weights_shape=None, 
                 is_conv=False):
        '''
        Shapes are referenced for a layer with weights of shape: (784, 128)

        values: jnp.ndarray             # Current state of the neurons in the layer, shape: (layer_sizes[rank],) __ (128,)
        thresholds: jnp.float32         # An array of thresholds, one per neuron, shape: (layer_sizes[rank],) __ (128,)
        input_residuals: jnp.ndarray    # Sum of all input neurons, shape: (layer_sizes[rank-1],) __ (784,)
        input order                     # Set input neuron to the iteration at which the input is received to record the order of input received, shape: (layer_sizes[rank-1],) __ (784,)
        input activity                  # Count the number of times a input neuron fired, shape: (layer_sizes[rank-1],) __ (784,)
        layer activity                  # Count the number of times a neuron activated in this layer, only used for restrict parameter and threshold, shape: (layer_sizes[rank],) __ (128,)
        output activity                 # For each input neuron stores the hidden neurons that fire, shape: (layer_sizes[rank-1], layer_sizes[rank]) __ (784, 128)  
        
        ____ Convolution fields
        weights_shape                   # Shape of the weights
        is_conv                         # Is convolution layer               
        '''
        self.values = values
        self.thresholds = thresholds
        self.input_residuals = input_residuals
        self.input_order = input_order
        self.input_activity = input_activity
        self.layer_activity = layer_activity
        self.output_activity = output_activity
        self.last_sent_iteration = last_sent_iteration
        self.input_vector = input_vector
        self.output_vector = output_vector
        self.values_history = values_history
        self.history_index = history_index
        self.weights_shape = weights_shape
        self.is_conv = is_conv

    # Tell JAX how to flatten this object
    def tree_flatten(self):
        children = (self.values, self.thresholds, self.input_residuals,
                    self.input_order, self.input_activity, self.layer_activity,
                    self.output_activity, self.last_sent_iteration, 
                    self.input_vector, self.output_vector, 
                    self.values_history, self.history_index,
                    self.weights_shape, self.is_conv)
        aux_data = None  # no extra static data
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    def replace(self, **updates):
        """Return a new NeuronStates object with some fields replaced."""
        return NeuronStates(
            values=updates.get("values", self.values),
            thresholds=updates.get("thresholds", self.thresholds),
            input_residuals=updates.get("input_residuals", self.input_residuals),
            input_order=updates.get("input_order", self.input_order),
            input_activity=updates.get("input_activity", self.input_activity),
            layer_activity=updates.get("layer_activity", self.layer_activity),
            output_activity=updates.get("output_activity", self.output_activity),
            last_sent_iteration=updates.get("last_sent_iteration", self.last_sent_iteration),
            input_vector=updates.get("input_vector", self.input_vector),
            output_vector=updates.get("output_vector", self.output_vector),
            values_history=updates.get("values_history", self.values_history),
            history_index=updates.get("history_index", self.history_index),
            weights_shape=updates.get("weights_shape", self.weights_shape),
            is_conv=updates.get("is_conv", self.is_conv),
        )
    
@dataclasses.dataclass(frozen=True)
class Params:
    dataset: str 
    random_seed: int
    layer_sizes: tuple[int, ...]
    init_thresholds: float      # Starting thresholds
    num_epochs: int 
    learning_rate: float
    batch_size: int
    load_file: bool
    shuffle_activations: bool   # Shuffle the activations in the network
    restrict: int               # The amount of times a single neuron can fire accross all inputs, if negative then no restriction
    firing_nb: int              # The maximum number of neurons that can fire for one input at each layer
    sync_rate: int              # The number of inputs that needs to be accumulated before firing  
    max_nonzero: int
    shuffle_input:bool          # Shuffle the input data 
    threshold_lr: float
    sparsity_impact: float
    rerun: str
    async_layer: int            # The layer that is training asynchronously while all other layers are training sync, if -1 then all layers are async
    history_size: int = 0       # Size of history you want to store
    max_kernel: int = None      # The maximum size of flattened kernel
    flat_layer_sizes: tuple[int, ...] = None

#region RERUN
def rerun_init(data_file_path, 
               mpi_config, 
               new_params,
               new_epoch_nb, 
               shuffle_activations=False,
               shuffle_input=False,
               firing_nb=False,
               sync_rate=False,
               batch_size=False,
               learning_rate=False,
               init_thresholds=False,
               restrict=False,
               sparsity_impact=False,
               threshold_lr=False,
               async_layer=False,
               ):
    '''
    Rerun from an existing file by replacing the fields marked as True with the values of new params 
    '''
    split_rank = mpi_config.split_rank
    last_rank = mpi_config.last_rank

    path = os.path.normpath(data_file_path).split(os.sep)
    assert path[1] == new_params.dataset, f"Rerun can only be used on the same dataset, got {path[1]} and {new_params.dataset}"
    
    with open(data_file_path, "r") as f:
        stored_data = json.load(f)

    load_file = stored_data["loadfile"]
    layer_sizes = list_to_tuple_deep(stored_data["layer_sizes"])
    assert layer_sizes == new_params.layer_sizes, f"Network structure must be the same to rerun, got {layer_sizes} and {new_params.layer_sizes}"
    
    # Use stored value if flag is False, otherwise use new_params value
    shuffle_activations_val = new_params.shuffle_activations if shuffle_activations else stored_data["shuffle activations"]
    shuffle_input_val = new_params.shuffle_input if shuffle_input else stored_data["shuffle input"]
    firing_nb_val = new_params.firing_nb if firing_nb else stored_data["firing number"]
    sync_rate_val = new_params.sync_rate if sync_rate else stored_data["synchronization rate"]
    batch_size_val = new_params.batch_size if batch_size else stored_data["batch_size"]
    learning_rate_val = new_params.learning_rate if learning_rate else stored_data["learning rate"]
    init_thresholds_val = new_params.init_thresholds if init_thresholds else extract_scalar(stored_data["thresholds"]["thresholds_1"])
    
    restrict_val = new_params.restrict if restrict else tuple(stored_data["restrict"])
    sparsity_impact_val = new_params.sparsity_impact if sparsity_impact else stored_data.get("sparsity impact", 0)
    threshold_lr_val = new_params.threshold_lr if threshold_lr else stored_data["threshold lr"]
    async_layer_val = new_params.async_layer if async_layer else stored_data.get("async layer", False)
    
    threshold_dict = stored_data["thresholds"]
    weights_dict = stored_data["weights"]

    params = Params(
        dataset=new_params.dataset,  # Assuming you want to keep the dataset from new_params
        random_seed=new_params.random_seed,  # Assuming you want to keep the random_seed from new_params
        layer_sizes=layer_sizes, 
        init_thresholds=init_thresholds_val, 
        num_epochs=new_epoch_nb, 
        learning_rate=learning_rate_val, 
        batch_size=batch_size_val,
        load_file=load_file,
        shuffle_activations=shuffle_activations_val,
        restrict=restrict_val,
        firing_nb=firing_nb_val,
        sync_rate=sync_rate_val,
        max_nonzero=new_params.max_nonzero,  # Assuming you want to keep max_nonzero from new_params
        shuffle_input=shuffle_input_val,
        threshold_lr=threshold_lr_val,
        sparsity_impact=sparsity_impact_val,
        rerun=data_file_path,
        async_layer=async_layer_val,
        max_kernel=new_params.max_kernel
    )
    
    if split_rank > 0:
        weights = jnp.array(weights_dict["layer_"+str(split_rank)])
        if split_rank < last_rank:
            thresholds = jnp.array(threshold_dict["thresholds_"+str(split_rank)])
        else:
            thresholds = jnp.zeros(layer_sizes[split_rank])
    else:
        l = layer_sizes[split_rank]
        if isinstance(l, int) or len(l) == 1:
            weights = thresholds = thresholds = jnp.zeros(layer_sizes[split_rank])
        else:
            weights = thresholds = jnp.zeros((1,1,1,1))
    return params, weights, thresholds

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
def store_training_data(size, network, mode, all_epoch_accuracies, all_validation_accuracies, test_accuracy, execution_time, all_iteration_mean, weights_dict, all_loss, thresholds_dict, optiname, network_type, all_history=None, total_batches=None): 
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
    elif mode == "inference":
        result_dir = os.path.join("network_results", params.dataset, "inference", network_type)
        filename = f"{params.random_seed}" + f"_load{params.load_file}" + filename_nn
        all_iteration_mean = np.array(all_iteration_mean).flatten().tolist()
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

    # Set up file path and changing the name if same name exists already 
    # filename = filename_header + "_".join(map(str, params.layer_sizes)) 
    filename += f"_acc{accuracy:.3f}" 
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
    
    if all_history is not None and len(all_history) > 0 and total_batches is not None:
        # Output history analysis  
        store_history(jnp.array(all_history), result_path, total_batches)

    # Store the results
    result_data = {
        "time": float(execution_time),
        "loadfile": params.load_file,
        "shuffle activations": params.shuffle_activations,
        "shuffle input": params.shuffle_input,
        "rerun": params.rerun,
        "processes": size,
        "firing number": params.firing_nb,
        "synchronization rate": params.sync_rate,
        "async layer": params.async_layer,
        "restrict": params.restrict,
        "threshold impact": params.sparsity_impact,
        "threshold lr": params.threshold_lr,
        "test accuracy": test_accuracy,
        "layer_sizes": params.layer_sizes,
        "batch_size": params.batch_size,
        "learning rate": params.learning_rate,
        "training accuracy": np.array(all_epoch_accuracies).tolist(),
        "validation accuracy": np.array(all_validation_accuracies).tolist(),
        "iterations mean": np.array(all_iteration_mean).tolist(),
        "loss": [float(loss) for loss in all_loss],
        "thresholds": thresholds_dict,
        "weights": weights_dict
    }

    with open(result_path + ".json", "w") as f:
        json.dump(result_data, f, indent=4)

    print(f"Results saved to {result_path}")

    if mode == "train":
        epochs = [i + 1 for i in range(len(all_epoch_accuracies))]        
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
            plt.plot(epochs, layer_values, marker='o', label=f'Layer {i} (last: {layer_values[-1]:.1f})')

        plt.xlabel("Epoch")
        plt.ylabel("Average Activation Values")
        plt.title("Average Activation Values per Layer")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(result_path + "_activations.png") 
        plt.close()
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
    