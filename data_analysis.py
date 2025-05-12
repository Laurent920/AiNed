import os
import json
import matplotlib.pyplot as plt

# Define your folder path
folder_path = "network_results/training/firing_nb_loadfileF_1_2_4_8_16_32_64_128/"

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
                firing_rates.append(data['firing number'])
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

# Plot Accuracy
axs[0].plot(firing_rates, train_accs, marker='o', label="Train Accuracy")
axs[0].plot(firing_rates, val_accs, marker='o', label="Validation Accuracy")
axs[0].plot(firing_rates, test_accs, marker='o', label="Test Accuracy")
axs[0].set_xlabel("Firing Rate")
axs[0].set_ylabel("Accuracy")
axs[0].set_title("Accuracy vs Firing Number")
axs[0].legend()
axs[0].grid(True)

# Plot Time
axs[1].plot(firing_rates, times, marker='o', color='purple', label="Time (min)")
axs[1].set_xlabel("Firing Rate")
axs[1].set_ylabel("Time per epoch(min)")
axs[1].set_title("Time vs Firing Number")
axs[1].legend()
axs[1].grid(True)

# Plot Iterations Mean
axs[2].plot(firing_rates, iterations_means, marker='o', label="Iterations Mean[1]")
axs[2].set_xlabel("Firing Rate")
axs[2].set_ylabel("Iterations Mean")
axs[2].set_title("Iterations Mean vs Firing Number")
# axs[2].set_yscale("log")
axs[2].legend()
axs[2].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(plot_folder + "combined_plots.png")
plt.close()
