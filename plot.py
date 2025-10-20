import numpy as np
import matplotlib.pyplot as plt

# Data from the table
layers = np.array([3, 4, 5, 6, 7])

# Different configurations: (Input activations, AA)
mnist_avg_150 = np.array([119, 125, 140, 142, 140])  # MNIST avg, AA=150
input_300_aa_150 = np.array([162, 147, 163, 162, 158])  # Input=300, AA=150
input_150_aa_300 = np.array([162, 206, 216, 223, 225])  # Input=150, AA=300
input_300_aa_300 = np.array([184, 207, 224, 231, 230])  # Input=300, AA=300
input_600_aa_600 = np.array([317, 372, 386, 393, 399])  # Input=600, AA=600

# Create the plot
plt.figure(figsize=(12, 7))

plt.plot(layers, mnist_avg_150, marker='o', linewidth=2, markersize=8, label='Input: 150, AA: 150')
plt.plot(layers, input_300_aa_150, marker='s', linewidth=2, markersize=8, label='Input: 300, AA: 150')
plt.plot(layers, input_150_aa_300, marker='^', linewidth=2, markersize=8, label='Input: 150, AA: 300')
plt.plot(layers, input_300_aa_300, marker='D', linewidth=2, markersize=8, label='Input: 300, AA: 300')
plt.plot(layers, input_600_aa_600, marker='*', linewidth=2, markersize=12, label='Input: 600, AA: 600')

plt.xlabel('Number of Layers', fontsize=12, fontweight='bold')
plt.ylabel('Time per Epoch (s)', fontsize=12, fontweight='bold')
plt.title('Framework Performance vs Number of Layers', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(layers)

plt.tight_layout()
plt.savefig('framework_performance_time_per_epoch.png', dpi=300, bbox_inches='tight')
plt.show()

# Also create the data as a dictionary for easy access
data_dict = {
    'layers': layers,
    'MNIST_avg_150': mnist_avg_150,
    '300_150': input_300_aa_150,
    '150_300': input_150_aa_300,
    '300_300': input_300_aa_300,
    '600_600': input_600_aa_600
}

# Print the arrays for verification
print("Data arrays:")
print(f"Layers: {layers}")
print(f"MNIST avg, AA=150: {mnist_avg_150}")
print(f"Input=300, AA=150: {input_300_aa_150}")
print(f"Input=150, AA=300: {input_150_aa_300}")
print(f"Input=300, AA=300: {input_300_aa_300}")
print(f"Input=600, AA=600: {input_600_aa_600}")


# Stack all configurations by layer
data = np.stack([mnist_avg_150, input_300_aa_300, input_600_aa_600], axis=1)  # shape (5 layers, 3 configs)

x = [150, 300, 600]

plt.figure(figsize=(8, 6))
# for i, layer in enumerate(layers):
#     plt.plot(x, data[i], marker='o', label=f'Layer {layer}')
plt.plot(x, data[0], marker='o', linewidth=2, markersize=8, label='Input: 150, AA: 150')
plt.plot(x, data[1], marker='D', linewidth=2, markersize=8, label='Input: 300, AA: 300')
plt.plot(x, data[2], marker='*', linewidth=2, markersize=12, label='Input: 600, AA: 600', color='purple')

plt.xlabel('Number of Activations (AA)')
plt.ylabel('Time')
plt.title('Relation between Number of Activations and Time per Layer')
plt.legend(title='Layers')
plt.grid(True)

plt.xticks(x)

plt.tight_layout()
plt.savefig('framework_performance_act_vs_time.png', dpi=300, bbox_inches='tight')
plt.show()
