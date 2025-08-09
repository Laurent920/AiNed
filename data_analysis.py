import os
import json
import matplotlib.pyplot as plt
import numpy as np
# # Define your folder path
# parameter_values = "1_2_4_8_16_32_64_128"
# folder_path = "network_results/mnist/training/constant_layers_firing_nb_test"+ "/"
# data_field = "firing number"
# data_field_label = "Firing number for constant layers"

# # parameter_values = "1_2_4_8_16_32_64_128"
# # folder_path = "network_results/training/firing_nb_99_train_accuracy" + "/"
# # data_field = "firing number"
# # data_field_label = "Firing number"
# # folder_add = "_99_train_accuracy"

# # parameter_values = "1_2_4_8_16_32_64_128"
# # folder_path = "network_results/training/restrict_penalty_" + parameter_values + "/"
# # data_field = "restrict"
# # data_field_label = "Number of times a neuron can fire in total before applying a penalty"
# # folder_add = "_penalty"

# # parameter_values = "1_2_4_8_16_32_64_128_256_784"
# # folder_path = "network_results/training/sync_rate_" + parameter_values + "/"
# # data_field = "synchronization rate"
# # data_field_label = "Synchronization rate"

# # parameter_values = "2_4_8" #"1_2_4_8_16_32_64_128_256_784"
# # folder_path = "network_results/training/firing_nb_2_sync_rate_" + parameter_values + "/"
# # data_field = "synchronization rate"
# # data_field_label = "Synchronization rate with firing nb 2"

# # network_results/mnist/training/firing_nb_constant_layers

# # Lists to hold the extracted values
# data = []
# train_accs = []
# val_accs = []
# test_accs = []
# times = []
# iterations_means = []
# total_epoch = []
# plot_epoch_nb = False

# # Loop through all JSON files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith(".json"):
#         # Extract hidden layer part between '784_' and '_10'
#         try:
#             hidden_part = filename.split("_784_")[1].split("_10")[0]
#             hidden_layers = list(map(int, hidden_part.split("_")))
#         except Exception as e:
#             print(f"Skipping file {filename} due to parsing error: {e}")
#             continue

#         if len(hidden_layers) != 7 or hidden_layers[0] not in [32]:
#             continue  # only keep 7-layer networks with uniform hidden size of 32, 64, or 128

#         hidden_size = hidden_layers[0]
        
        
#         file_path = os.path.join(folder_path, filename)
#         with open(file_path, 'r') as f:
#             data_f = json.load(f)
#             try:
#                 val = data_f[data_field]
#                 if val in data:
#                     index = data.index(val)
#                     print(val, data, index)

#                     total_epoch[index] += len(data_f['loss']) # add retraining length 
#                     # Extract the number after '_ep' in the filename
#                     ep_number = int(filename.split('_ep')[1].split('_')[0])
#                     if ep_number != len(data_f['loss']):
#                         ep_number = len(data_f['loss'])
#                         plot_epoch_nb = True
#                     times[index] = (times[index] + (data_f['time'] / ep_number/60))/2
                    
#                     iterations_means[index] = (iterations_means[index] + (data_f['iterations mean'][1]))/2
                    
#                     if data_f.get('rerun') is not None:
#                         train_accs[index] = data_f['training accuracy'][-1]                    
#                         val_accs[index] = data_f['validation accuracy'][-1]
#                         test_accs[index] = data_f['test accuracy']
#                 else:
#                     data.append(val)
#                     t_acc = data_f['training accuracy']
#                     v_acc = data_f['validation accuracy']
#                     train_accs.append(t_acc if type(t_acc) is float else t_acc[-1])
#                     val_accs.append(v_acc if type(v_acc) is float else v_acc[-1])
#                     test_accs.append(data_f['test accuracy'])
#                     total_epoch.append(len(data_f['loss']))
#                     # Extract the number after '_ep' in the filename
#                     ep_number = int(filename.split('_ep')[1].split('_')[0])
#                     if ep_number != total_epoch[-1]:
#                         ep_number = total_epoch[-1]
#                         plot_epoch_nb = True
#                     times.append(data_f['time'] / ep_number/60)
                    
#                     all_iterations = data_f['iterations mean'][1:]
#                     avg_iterations = sum([sublist[-1] for sublist in all_iterations])/len(all_iterations)
#                     iterations_means.append(avg_iterations)
#                     print(filename, val, avg_iterations)
#             except KeyError as e:
#                 print(f"Missing field {e} in file {filename}")
                
# # Sort all lists by firing rate
# sorted_data = sorted(zip(data, train_accs, val_accs, test_accs, times, iterations_means))
# data, train_accs, val_accs, test_accs, times, iterations_means = map(list, zip(*sorted_data))

# print(data)
# plot_folder = "Plots/"
# os.makedirs(plot_folder, exist_ok=True)

# # Create a single figure with subplots
# fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# epoch_nb = "for 2 epochs"
# epoch_nb = ""
# plt.suptitle("Results "+epoch_nb+" with parameters values: " + parameter_values, fontsize=16)

# # Adjust layout to accommodate the global title
# plt.tight_layout(rect=[0, 0, 1, 0.96])

# # Plot Accuracy
# axs[0].plot(data, train_accs, marker='o', label="Train Accuracy")
# axs[0].plot(data, val_accs, marker='o', label="Validation Accuracy")
# axs[0].plot(data, test_accs, marker='o', label="Test Accuracy")
# axs[0].set_xlabel(data_field_label)
# axs[0].set_ylabel("Accuracy")
# axs[0].set_title("Accuracy vs "+data_field_label)
# axs[0].legend()
# axs[0].grid(True)

# # Plot Time
# axs[1].plot(data, times, marker='o', color='purple', label="Time (min)")
# axs[1].set_xlabel(data_field_label)
# axs[1].set_ylabel("Time per epoch(min)")
# axs[1].set_title("Time vs "+data_field_label)
# axs[1].legend()
# axs[1].grid(True)

# if plot_epoch_nb:
#     # Create a second y-axis on the right side
#     ax2 = axs[1].twinx()

#     # Example: add another line plot to the right axis
#     # Replace 'other_values' and 'label' as needed
#     ax2.plot(data, total_epoch, marker='s', color='green', label="Epoch number")
#     ax2.set_ylabel("Epochs", color='green')
#     ax2.tick_params(axis='y', labelcolor='green')

#     # Combine legends from both axes
#     lines_1, labels_1 = axs[1].get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()
#     axs[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

# # Plot Iterations Mean
# axs[2].plot(data, iterations_means, marker='o', label="Iterations Mean")
# axs[2].set_xlabel(data_field_label)
# axs[2].set_ylabel("Iterations Mean")
# axs[2].set_title("Iterations Mean vs "+data_field_label)
# # axs[2].set_xscale("log")
# axs[2].legend()
# axs[2].grid(True)

# # Adjust layout and save the figure
# plt.tight_layout()
# # if not plot_epoch_nb:
# folder_add = " for constant layers"
# plt.savefig(plot_folder + data_field+folder_add+"32.png")
# plt.close()


#____________________________________________________________________Plot for folders:mnist/training/constant_layers_activation_test_______________
from collections import defaultdict

# Path to your folder
folder_path = "network_results/mnist/training/constant_layers_activation_test"
folder_path = "network_results/mnist/training/constant_layer_activation_test_with_fnb2"
folder_path = "network_results/shd/training/constant_layer_threshold_test"

# Dictionary to collect results: {hidden_size: {num_layers: [averages]}}
results = defaultdict(lambda: defaultdict(list))

# Go through all JSON files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        parts = filename.split("_")
        # if parts[1] != 'ep2': # Only keep the same 2 epoch runs
        #     continue
        try:
            # Get the hidden layer part from filename
            idx_784 = parts.index("700")
            hidden_layer_part = []
            i = idx_784 + 1
            while parts[i] not in ("20"):  # stops at output or known suffix
                hidden_layer_part.append(parts[i])
                i += 1
            hidden_layers = list(map(int, hidden_layer_part))
            # if float(parts[-3][3:]) < 0.870 or (float(parts[-3][3:]) == 0.896 and hidden_layers[0]==64) :
            #     # print(float(parts[-3][3:]))
            #     continue
            print(filename, hidden_layers)
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
            continue

        hidden_size = hidden_layers[0]
        if not all(h == hidden_size for h in hidden_layers):
            # Skip mixed-size layers
            continue
        num_layers = len(hidden_layers)

        # Load JSON data
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path) as f:
                data = json.load(f)
                activations_list = data["iterations mean"][1:]
                print(activations_list, len(activations_list))
                avg_activation = sum(a[-1] for a in activations_list) / len(activations_list)
                results[hidden_size][num_layers].append(avg_activation)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

input_avg = 149.6296844482422
# Plotting
colors = {32: "blue", 64: "green", 128: "red"}
plt.figure(figsize=(10, 6))

for hidden_size in [32, 64, 128]:
    print(hidden_size, len(results[hidden_size]))
    if hidden_size not in results:
        continue
    x = sorted(results[hidden_size].keys())
    y = [sum(results[hidden_size][n]) / len(results[hidden_size][n]) for n in x]
    plt.plot(x, y, marker="o", label=f"{hidden_size}-unit layers", color=colors[hidden_size])

plt.xlabel("Number of Hidden Layers")
plt.ylabel("Average Layer Activations")
plt.title("Average Layer Activations vs. Number of Hidden Layers with Firing number = 2")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot to file
plot_folder = "Plots"
os.makedirs(plot_folder, exist_ok=True)
plot_path = os.path.join(plot_folder, "activation_vs_layers_shd.png")
plt.savefig(plot_path)
print(f"Plot saved to: {plot_path}")
plt.close()
