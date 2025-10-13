import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import numpy as np
import json

def pad_batch(batch_x, batch_y, batch_size):
    # Pad the x data with 0 and the y data with nan for the last batch
    current_size = batch_y.shape[0]
    if current_size < batch_size:
        pad_amount = batch_size - current_size
        pad_y = jnp.full((pad_amount,), -1.0, dtype=jnp.float32)
        pad_x = jnp.zeros((pad_amount,) + batch_x.shape[1:], dtype=batch_x.dtype)  
        # jax.debug.print("rank {}, has batch size: {} and pad batch size: {}", rank, current_size, pad_x.shape)
        batch_y = jnp.concatenate([batch_y, pad_y], axis=0)
        batch_x = jnp.concatenate([batch_x, pad_x], axis=0)
    
    return batch_x, batch_y

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


# region SAVING DATA
def store_training_data(size, network, mode, all_epoch_accuracies, all_validation_accuracies, test_accuracy, execution_time, all_iteration_mean, weights_dict, all_loss, thresholds_dict, optiname, network_type): 
    filename_add_on = ""
    if optiname is not None:
        filename_add_on = f"_{optiname}_"
    
    params = network.params

    # Choose the saving folder
    if mode == "train":
        result_dir = os.path.join("network_results", params.dataset, "training", network_type)
        filename = f"{params.random_seed}" + f"_ep{params.num_epochs}" + network.filename
    elif mode == "inference":
        result_dir = os.path.join("network_results", params.dataset, "inference", network_type)
        filename = f"{params.random_seed}" + f"_load{params.load_file}" + network.filename
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
def update_history(weight_residuals, new_value):
    history = weight_residuals["values_history"]
    index = weight_residuals["history_index"]

    # Replace value at current index
    history = history.at[index].set(new_value)

    # Increment index and wrap around
    new_index = (index + 1) % history.shape[0]

    updated = weight_residuals.copy()
    updated["values_history"] = history
    updated["history_index"] = new_index
    return updated

def get_ordered_history(weight_residuals):
    history = weight_residuals["values_history"]     # shape: (B, T, 10)
    index = weight_residuals["history_index"]        # shape: (B,)

    def reorder_single(h, idx):
        return jnp.roll(h, shift=-idx, axis=0)

    # Vectorize across batch
    return jax.vmap(reorder_single)(history, index)

def process_history(values_history, target_labels):
    '''
    values_history: (B, T, 10)
    target_labels: (B)
    
    return (T), (T)
    '''
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
    print("all history shape: ",all_history.shape)

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
    print(f"Flattened shape: {flat_history[0].shape}, {flat_history[1].shape}")
    
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
    