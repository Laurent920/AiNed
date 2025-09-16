from mnist_helper import torch_mnist_loader_manual


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import time

# ==========================================================
# CNN MODEL
# ==========================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv layer: 1 input channel (MNIST is grayscale), 32 filters
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) 
        # Second conv layer: 32 -> 64 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)  # Downsample
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Fully connected
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))       # (N, 32, 28, 28)
        x = self.pool(F.relu(self.conv2(x))) # (N, 64, 14, 14)
        x = x.view(x.size(0), -1)       # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ==========================================================
# TRAINING AND EVALUATION
# ==========================================================
def train_model(train_loader, val_loader, test_loader, total_train_batches, total_val_batches, total_test_batches, device, epochs=10, lr=0.001):
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(iter(train_loader)):
            inputs = torch.tensor(inputs, dtype=torch.float32).view(-1, 1, 28, 28).to(device)
            targets = torch.tensor(targets, dtype=torch.long).to(device)

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
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/total_train_batches:.4f} Train Acc: {train_acc:.2f}%")

        # Validation step
        val_acc = evaluate(model, val_loader, total_val_batches, device)
        print(f"Validation Acc: {val_acc:.2f}%")
    end_time = time.time()

    # Final Test Evaluation
    val_start = time.time()
    test_acc = evaluate(model, test_loader, total_test_batches, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    val_end = time.time()

    input_shape = (1, 28, 28)
    save_cnn_weights(model, input_shape, batch_size)

    inference_time = val_end - val_start
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.6f} seconds")
    print(f"Inference Execution Time: {inference_time:.6f} seconds")
    return model


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



def save_cnn_weights(network, input_shape, batch_size, output_dir="tensor_data/CNN"):
    os.makedirs(output_dir, exist_ok=True)

    # Build filename: tensor_data_(in)_(layer1)_(layer2)_..._batch{batch}
    # Example: tensor_data_(1x28x28)_(32x3x3)_(64x3x3)_(128)_(10)_batch128
    layer_shapes = []
    for param in network.parameters():
        layer_shapes.append("x".join(map(str, list(param.shape))))

    filename = "tensor_data_" + f"({ 'x'.join(map(str, input_shape)) })_" \
               + "_".join(f"({s})" for s in layer_shapes) \
               + f"_batch{batch_size}"

    # Save parameters to .npz
    tensor_data = [param.detach().cpu().numpy() for param in network.parameters()]
    np.savez(os.path.join(output_dir, f"{filename}.npz"), *tensor_data)

    print(f"Saved CNN weights to {os.path.join(output_dir, f'{filename}.npz')}")

# ==========================================================
# MAIN SCRIPT
# ==========================================================
if __name__ == "__main__":
    batch_size = 128
    (train_loader, total_train_batches), (val_loader, total_val_batches), (test_loader, total_test_batches), max_nonzero = torch_mnist_loader_manual(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_model(train_loader, val_loader, test_loader, total_train_batches, total_val_batches, total_test_batches, device, epochs=10)

    