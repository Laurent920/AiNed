import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
# -------------------------------------------------------
# 1. Your MNIST loader
# -------------------------------------------------------
from dataset_helpers.mnist_helper import mnist_loader_manual


# -------------------------------------------------------
# 2. Flexible MLP model
# -------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim=10):
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h

        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------
# 2. Gradient Check and Error Computation Logic (NEW)
# -------------------------------------------------------

def compute_relative_error(analytic_grad, numeric_grad):
    """Computes the robust relative error element-wise using numpy."""
    # Ensure they are numpy arrays
    w_grad = analytic_grad.numpy().flatten()
    g_num = numeric_grad.numpy().flatten()

    # Numerator: Absolute difference
    numerator = np.abs(w_grad - g_num)
    
    # Denominator: np.maximum for element-wise max, plus smoothing term (1e-12)
    denominator = np.maximum(np.abs(w_grad), np.abs(g_num)) + 1e-12 
    
    relative_errors = numerator / denominator
    
    return relative_errors

def pytorch_gradient_check(model, criterion, x_batch, y_batch, epsilon=1e-5, threshold=1e-4):
    """
    Performs a finite difference gradient check on a PyTorch model.
    
    CRITICAL FIX: Converts everything to float64 for accurate gradient checking.
    Float32 has machine epsilon ~1e-7, so you can't expect errors below that!
    
    Args:
        model: PyTorch model to check
        criterion: Loss function
        x_batch: Input batch
        y_batch: Target batch
        epsilon: Perturbation size (1e-5 works well for float64)
        threshold: Maximum acceptable relative error (1e-4 to 1e-5 for float64)
    """
    print("\n" + "=" * 70)
    print(f"GRADIENT CHECK STARTING")
    print(f"Epsilon={epsilon:.1e}, Threshold={threshold:.1e}")
    print("=" * 70)
    
    device = next(model.parameters()).device
    
    # CRITICAL: Convert everything to float64 for numerical precision
    print("Converting model and data to float64 for precision...")
    model_f64 = copy.deepcopy(model).double()
    x = x_batch.clone().detach().double().to(device)
    y = y_batch.clone().detach().to(device)
    
    # Get Analytic Gradients (Backprop)
    model_f64.zero_grad()
    loss = criterion(model_f64(x), y)
    loss.backward()
    
    # Compute Numeric Gradients (Finite Difference)
    max_relative_error = 0.0
    total_errors = []
    param_count = 0
    worst_offender = (0.0, "", 0.0, 0.0)

    print(f"Checking parameters...")
    for name, p in model_f64.named_parameters():        
        if not p.requires_grad:
            continue
        
        analytic_grad = p.grad.clone()
        
        # Iterate over every element in the parameter tensor
        for idx in np.ndindex(p.shape):
            param_count += 1
            
            # Store original value
            original_value = p.data[idx].item()
            
            # --- Perturbation for L+ ---
            with torch.no_grad():
                p.data[idx] = original_value + epsilon
            L_plus = criterion(model_f64(x), y).item()
            
            # --- Perturbation for L- ---
            with torch.no_grad():
                p.data[idx] = original_value - epsilon
            L_minus = criterion(model_f64(x), y).item()
            
            # --- Restore original value ---
            with torch.no_grad():
                p.data[idx] = original_value
            
            # Numeric Gradient (Centered difference)
            g_num_scalar = (L_plus - L_minus) / (2 * epsilon)
            
            # Analytic Gradient
            g_analytic_scalar = analytic_grad[idx].item()
            
            # Compute Relative Error
            abs_diff  = abs(g_analytic_scalar - g_num_scalar)
            max_magnitude = max(abs(g_analytic_scalar), abs(g_num_scalar))
            # Special case: both essentially zero
            if max_magnitude < 1e-10:
                # Use absolute error when both are tiny
                rel_error = abs_diff 
            else:
                # Normal relative error
                rel_error = abs_diff / (max_magnitude + 1e-12)
            
            total_errors.append(rel_error)
            
            # Track worst offender
            if rel_error > max_relative_error:
                max_relative_error = rel_error
                worst_offender = (
                    rel_error, 
                    f"{name}{idx}", 
                    g_analytic_scalar, 
                    g_num_scalar
                )
            
            # Progress indicator every 1000 params
            if param_count % 100000 == 0:
                print(f"  Checked {param_count} parameters... (max error: {max_relative_error:.2e})")

    # Report Results
    mean_relative_error = np.mean(total_errors) if total_errors else 0.0
    
    print("\n" + "=" * 70)
    print("GRADIENT CHECK REPORT")
    print("=" * 70)
    print(f"Parameters checked: {param_count}")
    print(f"Max Relative Error: {max_relative_error:.2e}")
    print(f"Mean Relative Error: {mean_relative_error:.2e}")
    
    if max_relative_error > threshold:
        print("\n✗ GRADIENT CHECK FAILED!")
        print(f"   Max error ({max_relative_error:.2e}) exceeds threshold ({threshold:.1e})")
        print("-" * 50)
        print(f"Worst Offender ({worst_offender[1]}):")
        print(f"  Analytic Grad: {worst_offender[2]:.8f}")
        print(f"  Numeric Grad:  {worst_offender[3]:.8f}")
        print(f"  Relative Error: {worst_offender[0]:.2e}")
    else:
        print(f"\n✓ GRADIENT CHECK PASSED! (Max error < {threshold:.1e})")
    print("=" * 70)
    
    return max_relative_error < threshold


# BONUS: Fast random sampling version for quick checks
def pytorch_gradient_check_fast(model, criterion, x_batch, y_batch, 
                                 num_checks=100, epsilon=1e-5, threshold=1e-4):
    """
    Fast gradient check that randomly samples parameters instead of checking all.
    Use this for quick sanity checks on large models.
    """
    print("\n" + "=" * 70)
    print(f"FAST GRADIENT CHECK (sampling {num_checks} random parameters)")
    print(f"Epsilon={epsilon:.1e}, Threshold={threshold:.1e}")
    print("=" * 70)
    
    device = next(model.parameters()).device
    
    # Convert to float64
    model_f64 = copy.deepcopy(model).double()
    x = x_batch.clone().detach().double().to(device)
    y = y_batch.clone().detach().to(device)
    
    # Get Analytic Gradients
    model_f64.zero_grad()
    loss = criterion(model_f64(x), y)
    loss.backward()
    
    # Collect all parameters
    params_list = []
    for name, p in model_f64.named_parameters():
        if p.requires_grad:
            for idx in np.ndindex(p.shape):
                params_list.append((name, p, idx, p.grad[idx].item()))
    
    # Randomly sample
    np.random.seed(42)
    sampled_params = np.random.choice(len(params_list), 
                                      size=min(num_checks, len(params_list)), 
                                      replace=False)
    
    max_relative_error = 0.0
    total_errors = []
    worst_offender = (0.0, "", 0.0, 0.0)
    
    for i, param_idx in enumerate(sampled_params):
        name, p, idx, g_analytic_scalar = params_list[param_idx]
        
        original_value = p.data[idx].item()
        
        # L+
        with torch.no_grad():
            p.data[idx] = original_value + epsilon
        L_plus = criterion(model_f64(x), y).item()
        
        # L-
        with torch.no_grad():
            p.data[idx] = original_value - epsilon
        L_minus = criterion(model_f64(x), y).item()
        
        # Restore
        with torch.no_grad():
            p.data[idx] = original_value
        
        g_num_scalar = (L_plus - L_minus) / (2 * epsilon)
        
        # Compute Relative Error
        abs_diff  = abs(g_analytic_scalar - g_num_scalar)
        max_magnitude = max(abs(g_analytic_scalar), abs(g_num_scalar))
        # Special case: both essentially zero
        if max_magnitude < 1e-10:
            # Use absolute error when both are tiny
            rel_error = abs_diff 
        else:
            # Normal relative error
            rel_error = abs_diff / (max_magnitude + 1e-12)
        
        total_errors.append(rel_error)
        
        if rel_error > max_relative_error:
            max_relative_error = rel_error
            worst_offender = (rel_error, f"{name}{idx}", g_analytic_scalar, g_num_scalar)
        
        # if (i + 1) % 25 == 0:
        #     print(f"  Checked {i + 1}/{len(sampled_params)} samples... (max error: {max_relative_error:.2e})")
    
    mean_relative_error = np.mean(total_errors)
    
    print("\n" + "=" * 70)
    print("FAST GRADIENT CHECK REPORT")
    print("=" * 70)
    print(f"Parameters sampled: {len(sampled_params)}/{len(params_list)}")
    print(f"Max Relative Error: {max_relative_error:.2e}")
    print(f"Mean Relative Error: {mean_relative_error:.2e}")
    
    if max_relative_error > threshold:
        print("\n✗ GRADIENT CHECK FAILED!")
        print(f"   Max error ({max_relative_error:.2e}) exceeds threshold ({threshold:.1e})")
        print("-" * 50)
        print(f"Worst Offender ({worst_offender[1]}):")
        print(f"  Analytic Grad: {worst_offender[2]:.8f}")
        print(f"  Numeric Grad:  {worst_offender[3]:.8f}")
        print(f"  Relative Error: {worst_offender[0]:.2e}")
    else:
        print(f"\n✓ GRADIENT CHECK PASSED! (Max error < {threshold:.1e})")
    print("=" * 70)
    
    return max_relative_error < threshold
    
# -------------------------------------------------------
# 3. Get MNIST data from your custom loader
# -------------------------------------------------------
BATCH_SIZE = 36
(train_loader, train_batches), (val_loader, val_batches), (test_loader, test_batches), max_nonzero = \
    mnist_loader_manual(
        batch_size=BATCH_SIZE,
        shuffle=False,
        preprocess=False,
        CNN_preproces=False,
        downsample=False,
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device {device}")
# Determine input dimension from the first batch
sample_x, _ = next(iter(train_loader))
INPUT_DIM = sample_x.shape[1]     # e.g., 784 for MNIST


# -------------------------------------------------------
# 4. Create model + optimizer + loss
# -------------------------------------------------------
HIDDEN = [256, 256]

model = MLP(INPUT_DIM, HIDDEN).to(device)
for param in model.parameters():
    print(param)
criterion = nn.CrossEntropyLoss()

OPT = "adam"
# OPT = "sgd"
if OPT == "adam":
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
elif OPT == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)
else:
    raise ValueError("Unknown optimizer")
print(f"Chose optimizer {OPT}")

# -------------------------------------------------------
# 5. Gradient Check Execution (NEW)
# -------------------------------------------------------

# Get a single batch of data for the check
X_check, Y_check = next(iter(train_loader))
X_check = torch.tensor(X_check, dtype=torch.float32)
Y_check = torch.tensor(Y_check, dtype=torch.long)

# Only check a small batch (first 5 samples) and a sub-set of parameters 
# for speed, as a full check is extremely slow.
X_check = X_check[:1].to(device) 
Y_check = Y_check[:1].to(device)

# Full check (slow but thorough)
pytorch_gradient_check(
    model, 
    criterion, 
    X_check, 
    Y_check, 
    epsilon=1e-4,
    threshold=1e-4  # Reasonable for float64
)

# Fast check (recommended for development)
# pytorch_gradient_check_fast(
#     model, 
#     criterion, 
#     X_check, 
#     Y_check,
#     num_checks=100,  # Check 100 random parameters
#     epsilon=1e-5,
#     threshold=1e-4
# )

# -------------------------------------------------------
# Helper: evaluate accuracy on any loader
# -------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x = torch.tensor(x, dtype=torch.float32, device=device)
            y = torch.tensor(y, dtype=torch.long, device=device)

            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total * 100

# -------------------------------------------------------
# 5. Training
# -------------------------------------------------------
# EPOCHS = 5
# for epoch in range(EPOCHS):
#     model.train()
#     correct, total = 0, 0
#     total_loss = 0

#     for x, y in tqdm(train_loader, total=train_batches):
#         x = torch.tensor(x, dtype=torch.float32, device=device)
#         y = torch.tensor(y, dtype=torch.long, device=device)

#         optimizer.zero_grad()
#         out = model(x)
#         loss = criterion(out, y)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         preds = out.argmax(dim=1)
#         correct += (preds == y).sum().item()
#         total += y.size(0)

#     train_acc = correct / total * 100
#     val_acc = evaluate(model, val_loader)

#     print(f"Epoch {epoch+1}/{EPOCHS} | "
#           f"Loss={total_loss/train_batches:.4f} | "
#           f"Train Acc={train_acc:.2f}% | "
#           f"Val Acc={val_acc:.2f}%")

# # -------------------------------------------------------
# # 6. Final Test accuracy
# # -------------------------------------------------------
# test_acc = evaluate(model, test_loader)
# print(f"\nTest accuracy: {test_acc:.2f}%")
