import os
import json
import matplotlib.pyplot as plt

# Define your folder path
parameter_values = "1_2_4_8_16_32_64_128"
folder_path = "network_results/training/firing_nb_" + parameter_values + "/"
data_field = "firing number"
data_field_label = "Firing number"

parameter_values = "1_2_4_8_16" #"None_1_2_4_8_16_32_64_128_256_512"
folder_path = "network_results/training/restrict_fixed_" + parameter_values + "/"
data_field = "restrict"
data_field_label = "Number of times a neuron can fire in total"

parameter_values = "1_2_4_8_16_32_64_128_256_784"
folder_path = "network_results/training/sync_rate_" + parameter_values + "/"
data_field = "synchronization rate"
data_field_label = "Synchronization rate"

# parameter_values = "2_4_8" #"1_2_4_8_16_32_64_128_256_784"
# folder_path = "network_results/training/firing_nb_2_sync_rate_" + parameter_values + "/"
# data_field = "synchronization rate"
# data_field_label = "Synchronization rate with firing nb 2"

# Lists to hold the extracted values
firing_rates = []
train_accs = []
val_accs = []
test_accs = []
times = []
iterations_means = []

# Loop through all JSON files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            try:
                firing_rates.append(data[data_field])
                train_accs.append(data['training accuracy'])
                val_accs.append(data['validation accuracy'])
                test_accs.append(data['test accuracy'])
                # Extract the number after '_ep' in the filename
                try:
                    ep_number = int(filename.split('_ep')[1].split('_')[0])
                    times.append(data['time'] / ep_number/60)
                except (IndexError, ValueError) as e:
                    print(f"Error extracting '_ep' number from filename {filename}: {e}")
                    
                iterations_means.append(data['iterations mean'][1])
            except KeyError as e:
                print(f"Missing field {e} in file {filename}")

# Sort all lists by firing rate
sorted_data = sorted(zip(firing_rates, train_accs, val_accs, test_accs, times, iterations_means))
firing_rates, train_accs, val_accs, test_accs, times, iterations_means = map(list, zip(*sorted_data))

plot_folder = "Plots/"
os.makedirs(plot_folder, exist_ok=True)

# Create a single figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

plt.suptitle("Results for 8 epochs with parameters values: " + parameter_values, fontsize=16)

# Adjust layout to accommodate the global title
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Plot Accuracy
axs[0].plot(firing_rates, train_accs, marker='o', label="Train Accuracy")
axs[0].plot(firing_rates, val_accs, marker='o', label="Validation Accuracy")
axs[0].plot(firing_rates, test_accs, marker='o', label="Test Accuracy")
axs[0].set_xlabel(data_field_label)
axs[0].set_ylabel("Accuracy")
axs[0].set_title("Accuracy vs "+data_field_label)
axs[0].legend()
axs[0].grid(True)

# Plot Time
axs[1].plot(firing_rates, times, marker='o', color='purple', label="Time (min)")
axs[1].set_xlabel(data_field_label)
axs[1].set_ylabel("Time per epoch(min)")
axs[1].set_title("Time vs "+data_field_label)
axs[1].legend()
axs[1].grid(True)

# Plot Iterations Mean
axs[2].plot(firing_rates, iterations_means, marker='o', label="Iterations Mean[1]")
axs[2].set_xlabel(data_field_label)
axs[2].set_ylabel("Iterations Mean")
axs[2].set_title("Iterations Mean vs "+data_field_label)
# axs[2].set_xscale("log")
axs[2].legend()
axs[2].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(plot_folder + data_field+".png")
plt.close()
