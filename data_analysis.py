import os
import json
import matplotlib.pyplot as plt
import numpy as np
# Define your folder path
parameter_values = "1_2_4_8_16_32_64_128"
folder_path = "network_results/training/firing_nb_" + parameter_values + "/"
data_field = "firing number"
data_field_label = "Firing number"

# parameter_values = "1_2_4_8_16_32_64_128"
# folder_path = "network_results/training/firing_nb_99_train_accuracy" + "/"
# data_field = "firing number"
# data_field_label = "Firing number"
# folder_add = "_99_train_accuracy"

# parameter_values = "1_2_4_8_16_32_64_128" #"None_1_2_4_8_16_32_64_128_256_512"
# folder_path = "network_results/training/restrict_fixed_" + parameter_values + "/"
# data_field = "restrict"
# data_field_label = "Number of times a neuron can fire in total"

# parameter_values = "1_2_4_8_16_32_64_128_256_784"
# folder_path = "network_results/training/sync_rate_" + parameter_values + "/"
# data_field = "synchronization rate"
# data_field_label = "Synchronization rate"

# parameter_values = "2_4_8" #"1_2_4_8_16_32_64_128_256_784"
# folder_path = "network_results/training/firing_nb_2_sync_rate_" + parameter_values + "/"
# data_field = "synchronization rate"
# data_field_label = "Synchronization rate with firing nb 2"

# Lists to hold the extracted values
data = []
train_accs = []
val_accs = []
test_accs = []
times = []
iterations_means = []
total_epoch = []
plot_epoch_nb = False

# Loop through all JSON files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            data_f = json.load(f)
            try:
                val = data_f[data_field]
                if val in data:
                    index = data.index(val)
                    print(val, data, index)

                    total_epoch[index] += len(data_f['loss']) # add retraining length 
                    # Extract the number after '_ep' in the filename
                    ep_number = int(filename.split('_ep')[1].split('_')[0])
                    if ep_number != len(data_f['loss']):
                        ep_number = len(data_f['loss'])
                        plot_epoch_nb = True
                    times[index] = (times[index] + (data_f['time'] / ep_number/60))/2
                    
                    iterations_means[index] = (iterations_means[index] + (data_f['iterations mean'][1]))/2
                    
                    if data_f.get('rerun') is not None:
                        train_accs[index] = data_f['training accuracy'][-1]                    
                        val_accs[index] = data_f['validation accuracy'][-1]
                        test_accs[index] = data_f['test accuracy']
                else:
                    data.append(val)
                    t_acc = data_f['training accuracy']
                    v_acc = data_f['validation accuracy']
                    train_accs.append(t_acc if type(t_acc) is float else t_acc[-1])
                    val_accs.append(v_acc if type(v_acc) is float else v_acc[-1])
                    test_accs.append(data_f['test accuracy'])
                    total_epoch.append(len(data_f['loss']))
                    # Extract the number after '_ep' in the filename
                    ep_number = int(filename.split('_ep')[1].split('_')[0])
                    if ep_number != total_epoch[-1]:
                        ep_number = total_epoch[-1]
                        plot_epoch_nb = True
                    times.append(data_f['time'] / ep_number/60)
                    
                    iterations_means.append(data_f['iterations mean'][1])
            except KeyError as e:
                print(f"Missing field {e} in file {filename}")
                
# Sort all lists by firing rate
sorted_data = sorted(zip(data, train_accs, val_accs, test_accs, times, iterations_means))
data, train_accs, val_accs, test_accs, times, iterations_means = map(list, zip(*sorted_data))

plot_folder = "Plots/"
os.makedirs(plot_folder, exist_ok=True)

# Create a single figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

epoch_nb = "for 8 epochs"
epoch_nb = ""
plt.suptitle("Results "+epoch_nb+" with parameters values: " + parameter_values, fontsize=16)

# Adjust layout to accommodate the global title
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Plot Accuracy
axs[0].plot(data, train_accs, marker='o', label="Train Accuracy")
axs[0].plot(data, val_accs, marker='o', label="Validation Accuracy")
axs[0].plot(data, test_accs, marker='o', label="Test Accuracy")
axs[0].set_xlabel(data_field_label)
axs[0].set_ylabel("Accuracy")
axs[0].set_title("Accuracy vs "+data_field_label)
axs[0].legend()
axs[0].grid(True)

# Plot Time
axs[1].plot(data, times, marker='o', color='purple', label="Time (min)")
axs[1].set_xlabel(data_field_label)
axs[1].set_ylabel("Time per epoch(min)")
axs[1].set_title("Time vs "+data_field_label)
axs[1].legend()
axs[1].grid(True)

if plot_epoch_nb:
    # Create a second y-axis on the right side
    ax2 = axs[1].twinx()

    # Example: add another line plot to the right axis
    # Replace 'other_values' and 'label' as needed
    ax2.plot(data, total_epoch, marker='s', color='green', label="Epoch number")
    ax2.set_ylabel("Epochs", color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Combine legends from both axes
    lines_1, labels_1 = axs[1].get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    axs[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

# Plot Iterations Mean
axs[2].plot(data, iterations_means, marker='o', label="Iterations Mean[1]")
axs[2].set_xlabel(data_field_label)
axs[2].set_ylabel("Iterations Mean")
axs[2].set_title("Iterations Mean vs "+data_field_label)
# axs[2].set_xscale("log")
axs[2].legend()
axs[2].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
if not plot_epoch_nb:
    folder_add = ""
plt.savefig(plot_folder + data_field+folder_add+".png")
plt.close()
