import argparse
import time

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn

from async_RNN_pytorch_gradient_check import (
    TorchRuleRNN,
    init_feedforward_weights,
    init_recurrent_weight,
)
from dataset_helpers.mnist_helper import mnist_loader_manual


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_base_mnist_arrays(batch_size: int, data_dir: str):
    (train_loader, _), (_, _), (test_loader, _), _ = mnist_loader_manual(
        batch_size=batch_size,
        shuffle=True,
        preprocess=False,
        CNN_preprocess=False,
        downsample=False,
        sequential=False,
        permuted=False,
        data_dir=data_dir,
    )

    x_train = np.asarray(train_loader.X[train_loader.indices], dtype=np.float32)
    y_train = np.asarray(train_loader.Y[train_loader.indices], dtype=np.int64)
    x_test = np.asarray(test_loader.X[test_loader.indices], dtype=np.float32)
    y_test = np.asarray(test_loader.Y[test_loader.indices], dtype=np.int64)
    return x_train, y_train, x_test, y_test


def transform_dataset(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mnist":
        return x
    if mode == "smnist":
        # Sequential MNIST input is the same pixel order, typically standardized.
        return (x - 0.1307) / 0.3081
    raise ValueError(f"Unsupported mode: {mode}")


def accuracy(model, x: np.ndarray, y: np.ndarray, batch_size: int) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for s in range(0, x.shape[0], batch_size):
            e = min(s + batch_size, x.shape[0])
            xb = torch.from_numpy(x[s:e]).to(dtype=torch.float32)
            yb = torch.from_numpy(y[s:e]).to(dtype=torch.long)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += yb.shape[0]
    return float(correct / total) if total > 0 else 0.0


def train_one(
    mode: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    hidden_size: int,
    epochs: int,
    train_batch: int,
    eval_batch: int,
    lr: float,
    seed: int,
    sync_rate: int,
    firing_nb: int,
    train_samples: int,
    use_tanh: bool,
):
    x_train_mode = transform_dataset(x_train, mode)
    x_test_mode = transform_dataset(x_test, mode)

    if train_samples > 0:
        x_train_mode = x_train_mode[:train_samples]
        y_train_mode = y_train[:train_samples]
    else:
        y_train_mode = y_train

    layer_sizes = (x_train_mode.shape[1], hidden_size, 10)
    weights = init_feedforward_weights(layer_sizes, seed)
    w_ih, w_out = weights[0], weights[1]
    w_hh = init_recurrent_weight(hidden_size, seed, gain=0.5)
    bias_h = jnp.zeros((hidden_size,), dtype=jnp.float32)

    model = TorchRuleRNN(
        w_ih=w_ih,
        w_hh=w_hh,
        w_out=w_out,
        bias_h=bias_h,
        sync_rate=sync_rate,
        firing_nb=firing_nb,
        use_tanh=use_tanh,
    )

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = x_train_mode.shape[0]
    epoch_logs = []
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        perm = np.random.permutation(n)

        running_loss = 0.0
        seen = 0
        for s in range(0, n, train_batch):
            e = min(s + train_batch, n)
            idx = perm[s:e]
            xb = torch.from_numpy(x_train_mode[idx]).to(dtype=torch.float32)
            yb = torch.from_numpy(y_train_mode[idx]).to(dtype=torch.long)

            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

            bsz = int(yb.shape[0])
            running_loss += float(loss.item()) * bsz
            seen += bsz

        train_acc = accuracy(model, x_train_mode, y_train_mode, eval_batch)
        test_acc = accuracy(model, x_test_mode, y_test, eval_batch)
        dt = time.time() - t0
        avg_loss = running_loss / max(seen, 1)
        epoch_logs.append((ep + 1, avg_loss, train_acc, test_acc, dt))

    return model, epoch_logs


def main():
    parser = argparse.ArgumentParser(
        description="Train PyTorch async-style RNN on MNIST and sMNIST and report accuracy."
    )
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sync-rate", type=int, default=1)
    parser.add_argument("--firing-nb", type=int, default=10000)
    parser.add_argument(
        "--use-tanh",
        action="store_true",
        help="If set, use z_t = tanh(inner_t) before ReLU/output.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=3000,
        help="Number of training samples to use per dataset; -1 or 0 means full train split.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    x_train, y_train, x_test, y_test = load_base_mnist_arrays(args.train_batch, args.data_dir)

    train_samples = args.train_samples
    if train_samples <= 0:
        train_samples = x_train.shape[0]

    print(
        f"seed={args.seed} hidden={args.hidden_size} epochs={args.epochs} "
        f"train_batch={args.train_batch} eval_batch={args.eval_batch} lr={args.lr} "
        f"sync_rate={args.sync_rate} firing_nb={args.firing_nb} use_tanh={args.use_tanh} "
        f"train_samples={train_samples}"
    )

    for mode in ("mnist", "smnist"):
        _, logs = train_one(
            mode=mode,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            train_batch=args.train_batch,
            eval_batch=args.eval_batch,
            lr=args.lr,
            seed=args.seed,
            sync_rate=args.sync_rate,
            firing_nb=args.firing_nb,
            train_samples=train_samples,
            use_tanh=args.use_tanh,
        )

        for ep, avg_loss, train_acc, test_acc, dt in logs:
            print(
                f"{mode} epoch={ep} loss={avg_loss:.6f} "
                f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} time_s={dt:.2f}"
            )


if __name__ == "__main__":
    main()
