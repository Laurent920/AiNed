try:
    from dataset_helpers.mnist_helper import mnist_loader_manual
except ModuleNotFoundError:
    from mnist_helper import mnist_loader_manual

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import time
import matplotlib.pyplot as plt 
import json

save = False
epochs = 10
batch_size = 36
# ==========================================================
# CNN MODEL
# ==========================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Define your layers
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False)
        # self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2), padding=0)

        self.conv2 = nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2), padding=0)
        # self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2,2), padding=0)

        # self.out = nn.Linear(28*28*3, 10, bias=False)

        # self.fc1 = nn.Linear(28 * 28 * 5, 128, bias=False)
        self.fc1 = nn.Linear((14 * 14 * 5), 128, bias=False)
        # self.fc1 = nn.Linear(28 * 28 * 64, 32, bias=False)
        self.out = nn.Linear(128, 10, bias=False)

        # Automatically collect all layers with parameters
        self.activation_stats = {
            **{
                name: [] for name, module in self.named_children()
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d))
            },
            "input": []
        }

    def forward(self, x):
        self._record_activation("input", x)

        x = F.relu(self.conv1(x))
        self._record_activation("conv1", x)
        # print(x.shape)

        # x = self.pool1(x)
        # self._record_activation("pool1", x)
        # print(x.shape)

        x = F.relu(self.conv2(x))
        self._record_activation("conv2", x)
 
        x = self.pool2(x)
        self._record_activation("pool2", x)
        
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)

        # fc1
        x = F.relu(self.fc1(x))
        self._record_activation("fc1", x)

        # output layer
        x = self.out(x)
        self._record_activation("out", x)
        return x

    def _record_activation(self, layer_name, x):
        """Helper to record average nonzero activations per sample."""
        nonzero = (x != 0).sum().item() / x.size(0)
        self.activation_stats[layer_name].append(nonzero)

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()

        # ----- Block 1 -----
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.pool1   = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----- Block 2 -----
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pool2   = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----- Block 3 -----
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.pool3   = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----- Block 4 -----
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.pool4   = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----- Block 5 -----
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.pool5   = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----- Fully Connected Layers -----
        self.fc1 = nn.Linear(512 * 7 * 7, 4096, bias=False)
        self.fc2 = nn.Linear(4096, 4096, bias=False)
        self.out = nn.Linear(4096, num_classes, bias=False)

        # ----- Activation Stats -----
        self.activation_stats = {
            name: [] for name, module in self.named_children()
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d))
        }
        self.activation_stats["input"] = []

        # ----- Weight Initialization -----
        self._initialize_weights()

    def forward(self, x):
        self._record_activation("input", x)

        # Block 1
        x = F.relu(self.conv1_1(x)); self._record_activation("conv1_1", x)
        x = F.relu(self.conv1_2(x)); self._record_activation("conv1_2", x)
        x = self.pool1(x);           self._record_activation("pool1", x)

        # Block 2
        x = F.relu(self.conv2_1(x)); self._record_activation("conv2_1", x)
        x = F.relu(self.conv2_2(x)); self._record_activation("conv2_2", x)
        x = self.pool2(x);           self._record_activation("pool2", x)

        # Block 3
        x = F.relu(self.conv3_1(x)); self._record_activation("conv3_1", x)
        x = F.relu(self.conv3_2(x)); self._record_activation("conv3_2", x)
        x = F.relu(self.conv3_3(x)); self._record_activation("conv3_3", x)
        x = self.pool3(x);           self._record_activation("pool3", x)

        # Block 4
        x = F.relu(self.conv4_1(x)); self._record_activation("conv4_1", x)
        x = F.relu(self.conv4_2(x)); self._record_activation("conv4_2", x)
        x = F.relu(self.conv4_3(x)); self._record_activation("conv4_3", x)
        x = self.pool4(x);           self._record_activation("pool4", x)

        # Block 5
        x = F.relu(self.conv5_1(x)); self._record_activation("conv5_1", x)
        x = F.relu(self.conv5_2(x)); self._record_activation("conv5_2", x)
        x = F.relu(self.conv5_3(x)); self._record_activation("conv5_3", x)
        x = self.pool5(x);           self._record_activation("pool5", x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC
        x = F.relu(self.fc1(x)); self._record_activation("fc1", x)
        x = F.relu(self.fc2(x)); self._record_activation("fc2", x)
        x = self.out(x);         self._record_activation("out", x)
        return x

    def _record_activation(self, layer_name, x):
        """Record average nonzero activations per sample."""
        nonzero = (x != 0).sum().item() / x.size(0)
        self.activation_stats[layer_name].append(nonzero)

    def _initialize_weights(self):
        """He (Kaiming) initialization for all conv and linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
# ==========================================================
# TRAINING AND EVALUATION
# ==========================================================
def train_model(train_loader, val_loader, test_loader, total_train_batches, total_val_batches, total_test_batches, device, epochs=10, lr=0.0001):
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_accs, val_accs = [], []
    # (training_generator_p, total_train_batches_p), (validation_generator_p, total_val_batches_p), (test_generator_p, total_test_batches_p), max_nonzero = mnist_loader_manual(batch_size, shuffle=False, preprocess=True, CNN_preproces=True)
    # p_train_iter = iter(training_generator_p)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(iter(train_loader)):
            inputs = torch.tensor(inputs, dtype=torch.float32).view(-1, 1, 28, 28).to(device)
            targets = torch.tensor(targets, dtype=torch.long).to(device)

            # UNCOMMENT to check if the two dataloader implementation are equal
            # p_inputs, p_targets = next(p_train_iter)
            # for i in range(16):
            #     reconstructed = np.zeros((28, 28))
            #     for tup in p_inputs[i]:  # tup = (0, x, y, value)
            #         # print(tup)
            #         if int(tup[3]) > 0:
            #             reconstructed[int(tup[1]), int(tup[2])] = tup[3]
            #     if not np.array_equal(reconstructed, inputs[i].squeeze(0).cpu().numpy()):
            #         print(reconstructed, inputs[i])

            #         print("Different!")
            #     else:
            #         # print(reconstructed, inputs[i])
            #         pass

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        train_accs.append(train_acc)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/total_train_batches:.4f} Train Acc: {train_acc:.2f}%")

        print("Average activations per layer across training:")
        for layer, stats in model.activation_stats.items():
            avg_activations = sum(stats) / len(stats)
            print(f"  {layer}: {avg_activations:.2f}")
        
        # Validation step
        val_acc = evaluate(model, val_loader, total_val_batches, device)
        val_accs.append(val_acc)
        print(f"Validation Acc: {val_acc:.2f}%")

    end_time = time.time()

    # Final Test Evaluation
    val_start = time.time()
    test_acc = evaluate(model, test_loader, total_test_batches, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    val_end = time.time()

    inference_time = val_end - val_start
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.6f} seconds")
    print(f"Inference Execution Time: {inference_time:.6f} seconds")
    input_shape = (1, 28, 28)
    if save: save_cnn_weights(model, input_shape, batch_size, epochs, train_accs, val_accs, test_acc, 
                     execution_time=execution_time, inference_time=inference_time)

    return model


def save_cnn_weights(network, input_shape, batch_size, epochs, train_accs, val_accs, test_acc, 
                     execution_time=None, inference_time=None, plot=False, output_dir="tensor_data/CNN"):
    os.makedirs(output_dir, exist_ok=True)

    # ----- Build architecture string -----
    layers_repr = ["C" + "x".join(map(str, input_shape))]
    for layer in network.children():
        if isinstance(layer, nn.Conv2d):
            layers_repr.append(f"C{layer.out_channels}x{layer.in_channels}x{layer.kernel_size[0]}x{layer.kernel_size[1]}")
        elif isinstance(layer, nn.MaxPool2d):
            layers_repr.append(f"P{layer.kernel_size[0]}x{layer.kernel_size[0]}")
        elif isinstance(layer, nn.AvgPool2d):
            layers_repr.append(f"AvgP{layer.kernel_size[0]}x{layer.kernel_size[0]}")
        elif isinstance(layer, nn.Linear):
            layers_repr.append(f"L{layer.out_features}")

    filename_base = f"tensor_data_b{batch_size}_" + "_".join(layers_repr)

    # ----- Save weights (.npz) -----
    tensor_data = [param.detach().cpu().numpy() for param in network.parameters()]
    np.savez(os.path.join(output_dir, f"{filename_base}.npz"), *tensor_data)

    # ----- Compute average activations -----
    activation_stats = {}
    if hasattr(network, "activation_stats"):
        for layer, stats in network.activation_stats.items():
            if layer == "out": continue
            if len(stats) > 0:
                avg_activations = sum(stats) / len(stats)
                activation_stats[layer] = avg_activations
                print(f"  {layer}: {avg_activations:.2f}")
            else:
                activation_stats[layer] = 0.0

    # ----- Save training/validation accuracy plot (.png) -----
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, epochs + 1), train_accs, label="Train Accuracy")
        plt.plot(range(1, epochs + 1), val_accs, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Final Train: {train_accs[-1]:.2f}% | Val: {val_accs[-1]:.2f}% | Test: {test_acc:.2f}%")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename_base}.png"))
        plt.close()
        print(f"Saved training curve: {filename_base}.png")

    # ----- Save metrics to JSON -----
    results_dict = {
        "epochs": epochs,
        "batch_size": batch_size,
        "train_accuracy": train_accs,
        "validation_accuracy": val_accs,
        "test_accuracy": test_acc,
        "final_train_accuracy": train_accs[-1],
        "final_val_accuracy": val_accs[-1],
        "execution_time_sec": execution_time,
        "inference_time_sec": inference_time,
        "average_activations": activation_stats,
    }

    with open(os.path.join(output_dir, f"{filename_base}.json"), "w") as f:
        json.dump(results_dict, f, indent=4)

    print(f"Saved CNN weights: {filename_base}.npz")
    print(f"Saved training log: {filename_base}.json")



def evaluate(model, loader, total_batches, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in iter(loader):
            inputs = torch.tensor(inputs, dtype=torch.float32).view(-1, 1, 28, 28).to(device)
            targets = torch.tensor(targets, dtype=torch.long).to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total



# ==========================================================
# LOAD AND PARSE PRETRAINED FILES
# ==========================================================
def parse_cnn_filename(filename):
    """
    Parse CNN filename and return list of layer descriptions.
    Example: "tensor_data_b128_C1x28x28_C32x1x3x3_P2x2_L128_L10.npz"
    """
    # Strip directory and extension
    base = filename.split("/")[-1].replace(".npz", "")
    parts = base.split("_")

    # Ignore first 3 parts: 'tensor', 'data', 'eX', 'bY'
    layer_tokens = parts[3:]

    layers = []
    for token in layer_tokens:
        if token.startswith("C"):  # Conv or input
            dims = list(map(int, token[1:].split("x")))
            if len(dims) == 3:  # input layer: CxHxW
                layers.append(("input", tuple(dims)))
            elif len(dims) == 4:  # conv layer: CoutxCinxfhxfw
                layers.append(("conv", tuple(dims)))
        elif token.startswith("P"):  # Pool
            k = int(token[1])
            layers.append(("pool", (k, k)))
        elif token.startswith("L"):  # Linear
            out = int(token[1:])
            layers.append(("linear", out))
    return layers


def get_weights_for_rank(filename, rank):
    """
    Given filename and MPI rank, return weights corresponding to that rank.
    Bias terms are omitted since the CNN is trained without biases.
    """
    layers = parse_cnn_filename(filename)
    npz_data = np.load(filename)

    # Build mapping from layer description to npz index
    weights = []
    param_idx = 0
    for ltype, dims in layers:
        if ltype == "conv":
            cout, cin, kh, kw = dims
            w = npz_data[f"arr_{param_idx}"]  # conv weights
            weights.append(w)
            param_idx += 1
        elif ltype == "linear":
            out = dims
            w = npz_data[f"arr_{param_idx}"]  # fc weights
            weights.append(w)
            param_idx += 1
        elif ltype in ("input", "pool"):
            # No weights for input or pooling layers
            continue

    # Return weights for this rank
    w = weights[rank-1]
    if len(w.shape) == 2:
        w = w.transpose()
    print(f"Rank {rank} loading weights {w.shape} from {filename}")

    return w 

# ==========================================================
# MAIN SCRIPT
# ==========================================================
if __name__ == "__main__":
    (train_loader, total_train_batches), (val_loader, total_val_batches), (test_loader, total_test_batches), max_nonzero = mnist_loader_manual(batch_size, preprocess=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_model(train_loader, val_loader, test_loader, total_train_batches, total_val_batches, total_test_batches, device, epochs=epochs)

    