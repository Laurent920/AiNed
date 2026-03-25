"""
MNIST Trainer
=============
Data loading  : mnist_loader_manual  (original network_helper DataLoader)
Training      : PyTorch
Architectures : MLP | LSTM | GRU | RNN   (switch via CONFIG below)

Pixel-by-pixel sequence for RNNs: 784 steps × 1 feature
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp

try:
    import dataset_helpers.network_helper as network_helper
except ModuleNotFoundError:
    import network_helper

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  –  only thing you need to change between runs
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = dict(
    # ── model ──────────────────────────────────────────────────
    model_type     = "LSTM",       # "MLP" | "LSTM" | "GRU" | "RNN"

    # MLP-specific
    hidden_sizes   = [256, 256],  # hidden layers (input/output added automatically)

    # RNN-specific
    rnn_hidden     = 100,         # hidden size of recurrent layer
    rnn_layers     = 1,           # number of stacked recurrent layers

    # ── training ───────────────────────────────────────────────
    epochs         = 10,
    batch_size     = 64,
    lr             = 1e-3,

    # ── data ───────────────────────────────────────────────────
    dataset        = "smnist",    # "mnist" | "smnist"
    data_dir       = "",          # root dir passed to mnist_loader_manual
    shuffle        = True,
    permuted       = False,

    # ── misc ───────────────────────────────────────────────────
    output_dir     = "results",
    seed           = 42,
)
# ──────────────────────────────────────────────────────────────────────────────


# ── reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ─────────────────────────────── DATA ────────────────────────────────────────

def download_mnist_csv(dataset_folder: str) -> None:
    """Download MNIST via torchvision and save as CSV (train / test)."""
    from torchvision import datasets as tv_datasets

    os.makedirs(dataset_folder, exist_ok=True)
    train_csv = os.path.join(dataset_folder, "mnist_train.csv")
    test_csv  = os.path.join(dataset_folder, "mnist_test.csv")

    if os.path.exists(train_csv) and os.path.exists(test_csv):
        print("MNIST CSV files already exist – skipping download.")
        return

    print("Downloading MNIST …")
    torch_tmp = os.path.join(dataset_folder, "torch_tmp")

    for split, path in [(True, train_csv), (False, test_csv)]:
        ds     = tv_datasets.MNIST(root=torch_tmp, train=split, download=True)
        images = ds.data.numpy().reshape(-1, 28 * 28)
        labels = ds.targets.numpy()
        pd.DataFrame(np.column_stack([labels, images])).to_csv(path, index=False, header=False)
        print(f"  saved {path}")

    import shutil
    shutil.rmtree(torch_tmp, ignore_errors=True)
    print("Done.")


def downsample_14x14(x: np.ndarray) -> np.ndarray:
    x = x.reshape(-1, 28, 28)
    x = x.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))
    return x.reshape(-1, 196)


MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081


def mnist_loader_preprocessed_single(x, max_nonzero, sequential):
    """
    Preprocess a single MNIST sample (1D vector).
    Stores (index, value) for non-zero pixels up to max_nonzero.
    """
    processed_data = np.full((max_nonzero, 2), -2.0, dtype=np.float32)
    j = 0
    for i, val in enumerate(x):
        if sequential:
            processed_data[j] = [0, ((float(val) / 255.0) - MNIST_MEAN) / MNIST_STD]
            j += 1
        else:
            if val != 0:
                processed_data[j] = [i, val]
                j += 1
                if j >= max_nonzero:
                    break
    return jnp.array(processed_data)


def preprocess_dataset(dataset_x, max_nonzero, sequential):
    """
    Apply preprocessing to the whole dataset.
    dataset_x: shape (N, 784)
    Returns:   shape (N, max_nonzero, 2)
    """
    N = dataset_x.shape[0]
    processed_dataset = np.zeros((N, max_nonzero, 2), dtype=np.float32)
    for n in range(N):
        processed_dataset[n] = mnist_loader_preprocessed_single(dataset_x[n], max_nonzero, sequential)
    return jnp.array(processed_dataset)


def mnist_loader_preprocessed_single_CNN(x, max_nonzero, downsample=False):
    """
    Preprocess a single MNIST sample (1D vector).
    Stores (0, row, col, value) for non-zero pixels up to max_nonzero.
    """
    input_dimension = 14 if downsample else 28
    processed_data  = np.full((max_nonzero, 4), -2.0, dtype=np.float32)
    j = 0
    for i, val in enumerate(x):
        if val != 0:
            row = i // input_dimension
            col = i %  input_dimension
            processed_data[j] = [0, row, col, val]
            j += 1
            if j >= max_nonzero:
                break
    return processed_data


def preprocess_dataset_CNN(dataset_x, max_nonzero, downsample=False):
    """
    Apply CNN preprocessing to the whole dataset.
    dataset_x: shape (N, 784)
    Returns:   shape (N, max_nonzero, 4)  — (channel=0, row, col, value)
    """
    N = dataset_x.shape[0]
    processed_dataset = np.zeros((N, max_nonzero, 4), dtype=np.float32)
    for n in range(N):
        processed_dataset[n] = mnist_loader_preprocessed_single_CNN(dataset_x[n], max_nonzero, downsample)
    return processed_dataset


# ── original loader — unchanged signature and return format ──────────────────
def mnist_loader_manual(batch_size,
                        shuffle        = False,
                        preprocess     = True,
                        CNN_preprocess = False,
                        downsample     = False,
                        sequential     = False,
                        permuted       = False,
                        data_dir       = "",
                        cache_dir      = "./cache/mnist"):

    max_nonzero    = 784 if sequential else 351
    dataset_folder = os.path.join(data_dir, "data/mnist/")
    cache_dir      = os.path.join(data_dir, cache_dir)

    download_mnist_csv(dataset_folder)

    if preprocess:
        cache_dir_add  = "/async_CNN" if CNN_preprocess else "/async_MLP"
        cache_dir_add += "_14"               if downsample else ""
        cache_dir_add += "_sequential_mean0" if sequential else ""
        cache_dir_add += "_permuted"         if permuted   else ""
        cache_dir += cache_dir_add

    os.makedirs(cache_dir, exist_ok=True)
    train_cache_path = os.path.join(cache_dir, "train.npz")
    test_cache_path  = os.path.join(cache_dir, "test.npz")

    if preprocess and os.path.exists(train_cache_path):
        print("Loading cached MNIST dataset")
        data              = np.load(train_cache_path)
        mnist_data_x      = data["x"];  mnist_data_y = data["y"]
        data              = np.load(test_cache_path)
        mnist_data_x_test = data["x"];  mnist_data_y_test = data["y"]
    else:
        mnist_data        = pd.read_csv(os.path.join(dataset_folder, "mnist_train.csv"), header=None)
        mnist_data_x      = mnist_data.iloc[:, 1:].values.astype("float")
        mnist_data_y      = mnist_data.iloc[:, 0].values

        mnist_data        = pd.read_csv(os.path.join(dataset_folder, "mnist_test.csv"), header=None)
        mnist_data_x_test = mnist_data.iloc[:, 1:].values.astype("float")
        mnist_data_y_test = mnist_data.iloc[:, 0].values

        if permuted:
            np.random.seed(42)
            permutation       = np.random.permutation(mnist_data_x.shape[1])
            mnist_data_x      = mnist_data_x[:, permutation]
            mnist_data_x_test = mnist_data_x_test[:, permutation]

        if downsample:
            print("Downsampling MNIST images to 14×14")
            mnist_data_x      = downsample_14x14(mnist_data_x)
            mnist_data_x_test = downsample_14x14(mnist_data_x_test)

        if preprocess:
            if CNN_preprocess:
                print("Preprocess MNIST dataset for CNN")
                mnist_data_x      = preprocess_dataset_CNN(mnist_data_x,      max_nonzero, downsample)
                mnist_data_x_test = preprocess_dataset_CNN(mnist_data_x_test, max_nonzero, downsample)
            else:
                print("Preprocessing MNIST dataset")
                mnist_data_x      = preprocess_dataset(mnist_data_x,      max_nonzero, sequential)
                mnist_data_x_test = preprocess_dataset(mnist_data_x_test, max_nonzero, sequential)

        np.savez_compressed(train_cache_path, x=mnist_data_x,      y=mnist_data_y)
        np.savez_compressed(test_cache_path,  x=mnist_data_x_test, y=mnist_data_y_test)

    # ── split and wrap in original network_helper DataLoader ─────────────────
    train_indices, val_indices = network_helper.train_validate_split(
        mnist_data_y, val_ratio=0.2, shuffle=shuffle)

    train_dataloader = network_helper.DataLoader(
        mnist_data_x, mnist_data_y, batch_size, train_indices, shuffle=shuffle)
    val_dataloader   = network_helper.DataLoader(
        mnist_data_x, mnist_data_y, batch_size, val_indices, shuffle=shuffle)

    test_indices, _  = network_helper.train_validate_split(
        mnist_data_y_test, val_ratio=0, shuffle=shuffle)
    test_dataloader  = network_helper.DataLoader(
        mnist_data_x_test, mnist_data_y_test, batch_size, test_indices)

    total_train_batches = network_helper.get_total_batches(batch_size, train_indices)
    total_val_batches   = network_helper.get_total_batches(batch_size, val_indices)
    total_test_batches  = network_helper.get_total_batches(batch_size, test_indices)

    return (
        (train_dataloader, total_train_batches),
        (val_dataloader,   total_val_batches),
        (test_dataloader,  total_test_batches),
        max_nonzero,
    )


def load_mnist_arrays(batch_size, data_dir, dataset="smnist"):
    """
    Load MNIST/S-MNIST arrays using mnist_loader_manual, matching
    the data loading pattern from async_RNN_fptt.py.

    For S-MNIST: returns x as (N, 784, 1) — sequential pixel values,
                 normalised via (pixel/255 - 0.1307) / 0.3081.
    For MNIST:   returns x as (N, 784) — flat pixel vectors, normalised.

    Returns x_train, y_train, x_val, y_val, x_test, y_test.
    """
    sequential = (dataset == "smnist")
    (train_loader, _), (val_loader, _), (test_loader, _), _ = mnist_loader_manual(
        batch_size=batch_size,
        shuffle=True,
        preprocess=True,
        CNN_preprocess=False,
        downsample=False,
        sequential=sequential,
        permuted=False,
        data_dir=data_dir,
    )

    # Extract full arrays from loaders (matching async_RNN_fptt.py pattern)
    x_train = np.asarray(train_loader.X[train_loader.indices], dtype=np.float32)
    y_train = np.asarray(train_loader.Y[train_loader.indices], dtype=np.int64)
    x_val   = np.asarray(val_loader.X[val_loader.indices],     dtype=np.float32)
    y_val   = np.asarray(val_loader.Y[val_loader.indices],     dtype=np.int64)
    x_test  = np.asarray(test_loader.X[test_loader.indices],   dtype=np.float32)
    y_test  = np.asarray(test_loader.Y[test_loader.indices],   dtype=np.int64)

    if sequential:
        # x shape: (N, 784, 2) with (neuron_idx=0, value) — extract values only
        x_train = x_train[:, :, 1:2]  # (N, 784, 1)
        x_val   = x_val[:, :, 1:2]
        x_test  = x_test[:, :, 1:2]
    else:
        # MNIST: (N, max_nonzero, 2) — flatten to raw normalised pixels
        # Reload without preprocess for flat format
        (train_loader, _), (val_loader, _), (test_loader, _), _ = mnist_loader_manual(
            batch_size=batch_size, shuffle=True, preprocess=False,
            CNN_preprocess=False, downsample=False, sequential=False,
            permuted=False, data_dir=data_dir,
        )
        x_train = np.asarray(train_loader.X[train_loader.indices], dtype=np.float32)
        y_train = np.asarray(train_loader.Y[train_loader.indices], dtype=np.int64)
        x_val   = np.asarray(val_loader.X[val_loader.indices],     dtype=np.float32)
        y_val   = np.asarray(val_loader.Y[val_loader.indices],     dtype=np.int64)
        x_test  = np.asarray(test_loader.X[test_loader.indices],   dtype=np.float32)
        y_test  = np.asarray(test_loader.Y[test_loader.indices],   dtype=np.int64)
        # Normalise
        x_train = (x_train / 255.0 - MNIST_MEAN) / MNIST_STD
        x_val   = (x_val   / 255.0 - MNIST_MEAN) / MNIST_STD
        x_test  = (x_test  / 255.0 - MNIST_MEAN) / MNIST_STD

    return x_train, y_train, x_val, y_val, x_test, y_test


# ─────────────────────────────── MODELS ──────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, n_classes: int = 10):
        super().__init__()
        dims   = [input_size] + hidden_sizes + [n_classes]
        layers = []
        for in_d, out_d in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(in_d, out_d), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RNNClassifier(nn.Module):
    """Wraps LSTM / GRU / vanilla-RNN with an identical classification head."""

    _CELL = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}

    def __init__(self, model_type: str, hidden_size: int,
                 num_layers: int, n_classes: int = 10):
        super().__init__()
        cell_cls = self._CELL[model_type.upper()]
        self.rnn = cell_cls(
            input_size  = 1,           # pixel-by-pixel: one value per step
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
        )
        self.head = nn.Linear(hidden_size, n_classes)

        # Vanilla RNN: orthogonal init on hidden weights helps gradients survive 784 steps
        if model_type.upper() == "RNN":
            for name, p in self.rnn.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(p)

        # LSTM: default forget gate bias=0 → sigmoid(0)=0.5 → cell state halved
        # every step → 0.5^784 ≈ 0. Setting bias to 1.0 keeps gradients alive.
        if model_type.upper() == "LSTM":
            for name, p in self.rnn.named_parameters():
                if "bias_ih" in name or "bias_hh" in name:
                    # bias layout: [input, forget, cell, output] gates, each hidden_size
                    nn.init.zeros_(p)
                    p.data[hidden_size:2*hidden_size].fill_(1.0)  # forget gate bias → 1

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])  # last time-step → logits


def build_model(cfg: dict, input_size: int) -> nn.Module:
    mt = cfg["model_type"].upper()
    if mt == "MLP":
        return MLP(input_size, cfg["hidden_sizes"])
    elif mt in ("LSTM", "GRU", "RNN"):
        return RNNClassifier(mt,
                             hidden_size = cfg["rnn_hidden"],
                             num_layers  = cfg["rnn_layers"])
    else:
        raise ValueError(
            f"Unknown model_type '{cfg['model_type']}'. Choose: MLP | LSTM | GRU | RNN")


# ─────────────────────────────── TRAINING ────────────────────────────────────

def prepare_batch(x, y, model_type: str, device: str):
    """
    Convert pre-normalised numpy arrays into tensors shaped for each model.

      MLP : x (batch, 784)      — flat pixels
      RNN : x (batch, 784, 1)   — sequential pixels
    """
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    if model_type.upper() in ("LSTM", "GRU", "RNN"):
        x = x.reshape(x.shape[0], -1, 1)   # (batch, 784, 1)
    else:
        x = x.reshape(x.shape[0], -1)      # (batch, 784)

    return torch.tensor(x).to(device), torch.tensor(y).to(device)


def run_epoch(model, loader, criterion, model_type: str,
              optimiser=None, device="cpu"):
    """One pass over a PyTorch DataLoader. optimiser=None → eval mode."""
    training = optimiser is not None
    model.train(training)

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # Shape for model: RNN needs (B, 784, 1), MLP needs (B, 784)
            if model_type.upper() in ("LSTM", "GRU", "RNN"):
                x = x.reshape(x.shape[0], -1, 1)
            else:
                x = x.reshape(x.shape[0], -1)
            logits = model(x)
            loss   = criterion(logits, y)

            if training:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total


def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    is_rnn = cfg["model_type"].upper() in ("LSTM", "GRU", "RNN")
    dataset = cfg.get("dataset", "smnist")

    # ── data — load arrays via load_mnist_arrays (async_RNN_fptt.py style) ───
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_arrays(
        batch_size=cfg["batch_size"],
        data_dir=cfg["data_dir"],
        dataset=dataset,
    )
    print(f"Dataset: {dataset}  |  x_train: {x_train.shape}  y_train: {y_train.shape}")

    # Wrap in PyTorch DataLoaders
    from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

    def make_loader(x, y, shuffle):
        ds = TensorDataset(torch.tensor(x), torch.tensor(y))
        return TorchDataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle)

    tr_loader  = make_loader(x_train, y_train, shuffle=cfg["shuffle"])
    val_loader = make_loader(x_val,   y_val,   shuffle=False)
    te_loader  = make_loader(x_test,  y_test,  shuffle=False)

    # ── infer input size ─────────────────────────────────────────────────────
    input_size = 1 if is_rnn else x_train.shape[1]

    # ── model ─────────────────────────────────────────────────────────────────
    model    = build_model(cfg, input_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model : {cfg['model_type']}  ({n_params:,} params)")

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    mt        = cfg["model_type"]

    # ── training loop ─────────────────────────────────────────────────────────
    os.makedirs(cfg["output_dir"], exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'Epoch':>6}  {'Tr-loss':>8}  {'Tr-acc':>7}  {'Val-loss':>9}  {'Val-acc':>8}  {'Time':>6}")
    print("─" * 60)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss,  tr_acc  = run_epoch(model, tr_loader,  criterion, mt, optimiser, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, mt, None,      device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  "
              f"{val_loss:>9.4f}  {val_acc:>8.4f}  {elapsed:>5.1f}s")

    # ── test ──────────────────────────────────────────────────────────────────
    te_loss, te_acc = run_epoch(model, te_loader, criterion, mt, None, device)
    print(f"\nTest  loss={te_loss:.4f}  acc={te_acc:.4f}")

    # ── save checkpoint ───────────────────────────────────────────────────────
    tag      = cfg["model_type"].lower()
    base     = os.path.join(cfg["output_dir"], tag)
    ckpt     = base + "_model.pt"
    plot_png = base + "_accuracy.png"

    torch.save(model.state_dict(), ckpt)
    print(f"Model saved → {ckpt}")

    # ── plot ──────────────────────────────────────────────────────────────────
    epochs = range(1, cfg["epochs"] + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], "o-", label="Train")
    plt.plot(epochs, history["val_acc"],   "s-", label="Val")
    plt.axhline(te_acc, color="red", linestyle="--", label=f"Test {te_acc:.4f}")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title(f"{cfg['model_type']} — train {history['train_acc'][-1]:.4f} | "
              f"val {history['val_acc'][-1]:.4f} | test {te_acc:.4f}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(plot_png); plt.close()
    print(f"Plot  saved → {plot_png}")

    return model, history


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="MLP | LSTM | GRU | RNN  (overrides CONFIG)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None,
                        help="mnist | smnist")
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.model:
        cfg["model_type"] = args.model
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.lr:
        cfg["lr"] = args.lr

    train(cfg)