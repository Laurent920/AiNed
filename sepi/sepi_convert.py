"""
sepi_convert.py
===============
Convert weights saved from sepi_train.ipynb (asynctorch / PyTorch)
into the JSON format expected by sepi_tmlr_28052026.py.

Usage
-----
  python sepi_convert.py

Output
------
  weight/pretrained_mnist_128_10_converted.json
"""

import torch
import json
import numpy as np
import os

# ── Config — must match layer_sizes in tmlr_sepi_conf.yaml ─────────────────
CHECKPOINT  = "weight/asynctorch_mnist_linear.pt"
OUTPUT_JSON = "weight/pretrained_mnist_128_10_converted.json"
LAYER_SIZES = [784, 128, 10]
# ───────────────────────────────────────────────────────────────────────────

print(f"Loading checkpoint: {CHECKPOINT}")
state_dict = torch.load(CHECKPOINT, map_location="cpu")
print("Keys found:", list(state_dict.keys()))

n_weight_layers = len(LAYER_SIZES) - 1  # 2 for [784, 128, 10]

# asynctorch AsyncNetwork key names are NOT numeric-prefixed.
# Keys are: 'input_layer.module.weight', 'layers.0.module.weight', ...
# We must order them explicitly: input_layer first, then layers.0, layers.1, ...
# The old sort (int(k.split(".")[0]) if isdigit() else 0) gave both keys
# sort-key 0, relying on dict insertion order — fragile.
# Fix: extract the layer index explicitly from the key name.

def asynctorch_layer_index(key):
    """Return the layer order index for an asynctorch state_dict key."""
    if key.startswith("input_layer"):
        return 0
    if key.startswith("layers."):
        # 'layers.0.module.weight' → 1, 'layers.1.module.weight' → 2, ...
        return int(key.split(".")[1]) + 1
    return 999  # unknown keys sorted last

weight_keys = sorted(
    [k for k in state_dict if k.endswith(".weight") and state_dict[k].ndim == 2],
    key=asynctorch_layer_index
)
print(f"Weight matrices (in order): {weight_keys}")

assert len(weight_keys) == n_weight_layers, (
    f"Expected {n_weight_layers} weight matrices for layer_sizes={LAYER_SIZES}, "
    f"but found {len(weight_keys)}. Keys: {weight_keys}"
)

weights_dict    = {}
thresholds_dict = {}

for module_idx, key in enumerate(weight_keys):
    sepi_layer_idx = module_idx + 1          # sepi_tmlr uses layer_1, layer_2, ...
    w = state_dict[key].cpu().float().numpy()  # shape: [out, in]
    w = w.T                                    # → [in, out]  ✓

    expected_shape = (LAYER_SIZES[module_idx], LAYER_SIZES[module_idx + 1])
    assert w.shape == expected_shape, (
        f"layer_{sepi_layer_idx}: expected shape {expected_shape}, got {w.shape}\n"
        f"Check that LAYER_SIZES={LAYER_SIZES} matches the trained network."
    )

    weights_dict[f"layer_{sepi_layer_idx}"]         = w.tolist()
    thresholds_dict[f"thresholds_{sepi_layer_idx}"] = \
        [0.3] * LAYER_SIZES[module_idx + 1]

    print(f"  layer_{sepi_layer_idx}: {w.shape}  "
          f"min={w.min():.4f}  max={w.max():.4f}  mean={w.mean():.6f}")

output = {
    "params": {
        "dataset":      "mnist",
        "layer_sizes":  LAYER_SIZES,
        "note":         "pretrained via sepi_train.ipynb (asynctorch)"
    },
    "weights":    weights_dict,
    "thresholds": thresholds_dict,
    "accuracies": {"train": [-1], "val": [-1], "test": -1},
    "loss":       [],
    "iterations": []
}

os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f)

size_mb = os.path.getsize(OUTPUT_JSON) / 1e6
print(f"\n✓  Saved → {OUTPUT_JSON}  ({size_mb:.2f} MB)")
print("   Weight keys: ", list(weights_dict.keys()))