"""
FPTT (Forward Propagation Through Time) training for the custom async RNN.

Based on: "Training RNNs via Forward Propagation Through Time"
Reference implementation: https://github.com/anilkagak2/FPTT

Core idea
---------
The sequence of length T is split into PARTS chunks.  For chunk p (0-indexed):

    loss_p = beta_p * CE(output_p, y)  +  (1 - beta_p) * oracle_loss_p
    beta_p = (p+1) / PARTS

    oracle_loss_p = -mean( oracle_prob_p · log_softmax(logit_p) )

    oracle_prob_p:
        p < PARTS-1  →  estimate_class_distribution[y, p]   (previous-epoch estimate)
        p == PARTS-1 →  one_hot(y)                          (true label at the end)

A consensus regularizer is added to every chunk loss:

    regularizer = (rho - 1) * sum(param · lm)
                + lambda * 0.5 * alpha * sum(||param - sm||²)

where (sm, lm) are per-parameter running statistics updated *after* each
optimizer step via post_optimizer_updates:

    lm  +=  -alpha * (param - sm)
    sm   =  (1-beta)*sm + beta*param - (beta/alpha)*lm

At the start of each epoch, sm is reset to the current param values
(reset_named_params), so the regularizer pulls params toward their
epoch-start values, preventing large intra-epoch drifts.

Forward-pass equations (your custom RNN, one hidden layer):

    inner_t = W_ih[t] * x_t  +  z_{t-1}  -  o_{t-1}  +  o_{t-1} @ W_hh  +  b_h
    z_t     = tanh(inner_t)   if use_tanh else inner_t
    o_t     = ReLU(z_t)  (then sync_rate + firing_nb masks)
    logit   = sum_t  o_t @ W_out         (accumulated over the chunk)
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from async_RNN_pytorch_gradient_check import (
    TorchRuleRNN,
    init_feedforward_weights,
    init_recurrent_weight,
    keep_top_k_batch_torch,
)
from dataset_helpers.mnist_helper import mnist_loader_manual


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_mnist_arrays(batch_size: int, data_dir: str, dataset: str = "mnist"):
    """
    Load MNIST pixel arrays and normalize.

    dataset="mnist"  → normalize by train mean/std (data-derived)
    dataset="smnist" → normalize by standard MNIST constants (0.1307, 0.3081),
                       matching the sequential-MNIST benchmark convention
    """
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
    x_test  = np.asarray(test_loader.X[test_loader.indices],  dtype=np.float32)
    y_test  = np.asarray(test_loader.Y[test_loader.indices],  dtype=np.int64)

    if dataset == "smnist":
        # Standard sMNIST normalization
        x_train = (x_train - 0.1307) / 0.3081
        x_test  = (x_test  - 0.1307) / 0.3081
    else:
        # Normalize using train statistics
        mean = x_train.mean()
        std  = x_train.std() + 1e-8
        x_train = (x_train - mean) / std
        x_test  = (x_test  - mean) / std
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# FPTT parameter-state helpers  (direct port of the reference implementation)
# ---------------------------------------------------------------------------

def get_stats_named_params(model):
    """
    For each trainable parameter create (param, sm, lm) where:
        sm  = shadow copy (running average, initialised to param)
        lm  = momentum term (initialised to zero)
    """
    named_params = {}
    for name, param in model.named_parameters():
        sm = param.detach().clone()
        lm = torch.zeros_like(param.detach())
        named_params[name] = (param, sm, lm)
    return named_params


def post_optimizer_updates(named_params, alpha, beta):
    """
    Called after each optimizer.step().
        lm  +=  -alpha * (param - sm)
        sm   =  (1-beta)*sm + beta*param - (beta/alpha)*lm
    """
    for name in named_params:
        param, sm, lm = named_params[name]
        lm.data.add_(-alpha * (param.data - sm.data))
        sm.data.mul_(1.0 - beta)
        sm.data.add_(beta * param.data - (beta / alpha) * lm.data)


def reset_named_params(named_params):
    """Reset sm = param, lm = 0 at the start of each epoch."""
    for name in named_params:
        param, sm, lm = named_params[name]
        sm.data.copy_(param.data)
        lm.data.zero_()


def get_regularizer(named_params, alpha, lmbda, rho=0.0):
    """regularizer = (rho-1)*sum(param·lm) + lambda*0.5*alpha*sum(||param-sm||²)"""
    reg = torch.zeros([], device=next(iter(named_params.values()))[0].device)
    for name in named_params:
        param, sm, lm = named_params[name]
        reg = reg + (rho - 1.0) * torch.sum(param * lm)
        reg = reg + lmbda * 0.5 * alpha * torch.sum((param - sm) ** 2)
    return reg


# ---------------------------------------------------------------------------
# PyTorch RNN model for FPTT
# ---------------------------------------------------------------------------

class FPTTRuleRNN(nn.Module):
    """
    Same dynamics as TorchRuleRNN, but exposes a chunk-level forward so that
    the hidden state can be detached between chunks (truncated BPTT within
    each chunk, FPTT across chunks).
    """
    def __init__(self, w_ih, w_hh, w_out, bias_h, sync_rate, firing_nb, use_tanh=False):
        super().__init__()
        self.w_ih   = nn.Parameter(torch.tensor(np.asarray(w_ih),   dtype=torch.float32))
        self.w_hh   = nn.Parameter(torch.tensor(np.asarray(w_hh),   dtype=torch.float32))
        self.w_out  = nn.Parameter(torch.tensor(np.asarray(w_out),  dtype=torch.float32))
        self.bias_h = nn.Parameter(torch.tensor(np.asarray(bias_h), dtype=torch.float32))
        self.sync_rate = int(sync_rate)
        self.firing_nb = int(firing_nb)
        self.use_tanh  = bool(use_tanh)

    def init_hidden(self, batch_size):
        H = self.w_ih.shape[1]
        device = self.w_ih.device
        return (
            torch.zeros(batch_size, H, device=device),
            torch.zeros(batch_size, H, device=device),
        )

    def forward_chunk(self, x_chunk, hidden, t_offset):
        """
        Run the RNN over x_chunk (B, chunk_len) starting at absolute step t_offset.
        Returns accumulated logits (B, C) and updated hidden state.

        Speedup: all input projections x_t * W_ih[t] are precomputed in a
        single batched multiply before the sequential recurrence loop.
        """
        z_prev, o_prev = hidden
        batch_size, chunk_len = x_chunk.shape
        C = self.w_out.shape[1]
        device = self.w_ih.device

        # Precompute all input projections in one op: (B, chunk_len, H)
        w_ih_chunk = self.w_ih[t_offset : t_offset + chunk_len]      # (chunk_len, H)
        x_proj = x_chunk.unsqueeze(2) * w_ih_chunk.unsqueeze(0)      # (B, chunk_len, H)

        logits = torch.zeros(batch_size, C, device=device)

        for local_t in range(chunk_len):
            t = t_offset + local_t

            inner = (
                x_proj[:, local_t, :]
                + z_prev
                - o_prev
                + o_prev @ self.w_hh
                + self.bias_h.unsqueeze(0)
            )
            z_t = torch.tanh(inner) if self.use_tanh else inner
            o_t = torch.relu(z_t)

            sync_fire = 1.0 if ((t + 1) % self.sync_rate == 0) else 0.0
            o_t = o_t * sync_fire
            o_t = keep_top_k_batch_torch(o_t, self.firing_nb)

            logits = logits + o_t @ self.w_out
            z_prev = z_t
            o_prev = o_t

        return logits, (z_prev, o_prev)


# ---------------------------------------------------------------------------
# FPTT training
# ---------------------------------------------------------------------------

def train_one_epoch(
    x_train, y_train,
    model, optimizer, named_params,
    estimate_class_distribution,
    epoch,
    PARTS,
    alpha, beta, lmbda, rho,
    batch_size,
    clip,
    n_classes,
    warm_epochs=1,
):
    """Train for one epoch using FPTT. Returns average surrogate loss."""
    model.train()
    n = x_train.shape[0]
    T = x_train.shape[1]
    step = T // PARTS
    _PARTS = PARTS if PARTS * step >= T else PARTS + 1

    rng = np.random.default_rng(epoch)
    perm = rng.permutation(n)

    total_loss = 0.0
    total_batches = 0

    for s in range(0, n, batch_size):
        idx = perm[s : s + batch_size]
        xb  = torch.tensor(x_train[idx], dtype=torch.float32)
        yb  = torch.tensor(y_train[idx], dtype=torch.long)
        B   = yb.shape[0]

        hidden = model.init_hidden(B)

        for p in range(_PARTS):
            start = p * step
            end   = min(start + step, T)
            if start >= T:
                break
            x_chunk = xb[:, start:end]

            # Detach hidden state: no gradient flow between chunks
            h_detached = (hidden[0].detach(), hidden[1].detach())

            # Oracle distribution for this chunk
            if p < _PARTS - 1:
                if epoch <= warm_epochs:
                    oracle_prob = torch.full((B, n_classes), 1.0 / n_classes)
                else:
                    oracle_prob = estimate_class_distribution[yb, p]   # (B, C)
            else:
                oracle_prob = F.one_hot(yb, num_classes=n_classes).float()

            optimizer.zero_grad()
            chunk_logits, hidden = model.forward_chunk(x_chunk, h_detached, t_offset=start)

            beta_p    = (p + 1) / _PARTS
            log_probs = F.log_softmax(chunk_logits, dim=1)

            clf_loss    = beta_p         * F.nll_loss(log_probs, yb)
            oracle_loss = (1.0 - beta_p) * torch.mean(-torch.sum(oracle_prob * log_probs, dim=1))
            regularizer = get_regularizer(named_params, alpha=alpha, lmbda=lmbda, rho=rho)
            loss = clf_loss + oracle_loss + regularizer

            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            post_optimizer_updates(named_params, alpha=alpha, beta=beta)

            total_loss    += float(loss.item())
            total_batches += 1

            # Update the reference distribution estimate for future epochs.
            # Only update when the model is wrong, once per class per batch.
            if p < _PARTS - 1:
                with torch.no_grad():
                    probs = torch.softmax(chunk_logits, dim=1)
                    filled = [False] * n_classes
                    for j in range(B):
                        c = int(yb[j].item())
                        if not filled[c] and torch.argmax(probs[j]).item() != c:
                            estimate_class_distribution[c, p] = probs[j].detach()
                            filled[c] = True
                        if all(filled):
                            break

    return total_loss / max(total_batches, 1)


def evaluate(x, y, model, batch_size, n_classes):
    """Full-sequence accuracy evaluation (no weight updates)."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for s in range(0, x.shape[0], batch_size):
            xb = torch.tensor(x[s : s + batch_size], dtype=torch.float32)
            B  = xb.shape[0]
            hidden = model.init_hidden(B)
            logits, _ = model.forward_chunk(xb, hidden, t_offset=0)
            all_preds.append(logits.argmax(dim=1).numpy())
    preds = np.concatenate(all_preds)
    return float(np.mean(preds == y))


# ---------------------------------------------------------------------------
# Top-level training run
# ---------------------------------------------------------------------------

def train_fptt(
    x_train, y_train,
    x_test,  y_test,
    hidden_size,
    n_classes,
    epochs,
    batch_size,
    lr,
    alpha,
    beta,
    lmbda,
    rho,
    PARTS,
    clip,
    seed,
    sync_rate,
    firing_nb,
    use_tanh,
    train_samples,
    warm_epochs,
    optim_name,
):
    if train_samples > 0:
        x_train = x_train[:train_samples]
        y_train = y_train[:train_samples]

    T = x_train.shape[1]
    layer_sizes = (T, hidden_size, n_classes)

    weights = init_feedforward_weights(layer_sizes, seed)
    w_ih, w_out = weights[0], weights[1]
    w_hh   = init_recurrent_weight(hidden_size, seed, gain=0.5)
    bias_h = np.zeros(hidden_size, dtype=np.float32)

    model = FPTTRuleRNN(
        w_ih=w_ih, w_hh=w_hh, w_out=w_out, bias_h=bias_h,
        sync_rate=sync_rate, firing_nb=firing_nb, use_tanh=use_tanh,
    )

    if optim_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    named_params = get_stats_named_params(model)

    # Reference distributions: shape (n_classes, PARTS, n_classes), init uniform
    estimate_class_distribution = torch.full(
        (n_classes, PARTS, n_classes), 1.0 / n_classes
    )

    logs = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        reset_named_params(named_params)

        avg_loss = train_one_epoch(
            x_train, y_train,
            model, optimizer, named_params,
            estimate_class_distribution,
            epoch=ep,
            PARTS=PARTS,
            alpha=alpha, beta=beta, lmbda=lmbda, rho=rho,
            batch_size=batch_size,
            clip=clip,
            n_classes=n_classes,
            warm_epochs=warm_epochs,
        )

        # Accuracy from a proper full-sequence pass (up to 1000 train samples)
        train_acc = evaluate(x_train[:1000], y_train[:1000], model,
                             batch_size=batch_size * 4, n_classes=n_classes)
        test_acc  = evaluate(x_test, y_test, model,
                             batch_size=batch_size * 4, n_classes=n_classes)
        dt = time.time() - t0
        logs.append((ep, avg_loss, train_acc, test_acc, dt))
        print(
            f"epoch={ep}  loss={avg_loss:.6f}  "
            f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  time={dt:.1f}s"
        )

    return model, logs


# ---------------------------------------------------------------------------
# Smoke test  (uses synthetic T=40 data — fast, no MNIST loading needed)
# ---------------------------------------------------------------------------

def smoke_test(data_dir=""):
    """Quick correctness check using T=40 synthetic data."""
    print("=" * 60)
    print("Smoke test: 5 epochs, synthetic data (T=40, H=32, C=4)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N, T, C = 400, 40, 4
    x_train = rng.normal(0, 1, (N, T)).astype(np.float32)
    y_train = rng.integers(0, C, N, dtype=np.int64)
    x_test  = rng.normal(0, 1, (80, T)).astype(np.float32)
    y_test  = rng.integers(0, C, 80, dtype=np.int64)

    _, logs = train_fptt(
        x_train=x_train, y_train=y_train,
        x_test=x_test,   y_test=y_test,
        hidden_size=32,
        n_classes=C,
        epochs=5,
        batch_size=64,
        lr=1e-3,
        alpha=0.1,
        beta=0.5,
        lmbda=2.0,
        rho=0.0,
        PARTS=4,
        clip=1.0,
        seed=42,
        sync_rate=1,
        firing_nb=10000,
        use_tanh=False,
        train_samples=0,
        warm_epochs=1,
        optim_name="adam",
    )

    first_loss = logs[0][1]
    last_loss  = logs[-1][1]
    last_train = logs[-1][2]

    print(f"\nFirst epoch loss : {first_loss:.6f}")
    print(f"Last  epoch loss : {last_loss:.6f}")
    print(f"Last  train  acc : {last_train:.4f}")

    assert last_loss < first_loss, (
        f"Surrogate loss did not decrease: {first_loss:.6f} -> {last_loss:.6f}"
    )
    assert last_train > 0.10, f"Train accuracy below chance: {last_train:.4f}"
    print("Smoke test PASSED.")
    return logs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FPTT training for the custom async RNN on MNIST"
    )
    parser.add_argument("--data-dir",       type=str,   default="")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--hidden-size",    type=int,   default=128)
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch-size",     type=int,   default=128)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--alpha",          type=float, default=0.1,
                        help="Consensus regularisation strength.")
    parser.add_argument("--beta",           type=float, default=0.5,
                        help="Running-average decay for sm update.")
    parser.add_argument("--lmbda",          type=float, default=2.0,
                        help="Scale for the quadratic consensus penalty.")
    parser.add_argument("--rho",            type=float, default=0.0,
                        help="Coefficient on the lm·param term.")
    parser.add_argument("--parts",          type=int,   default=10,
                        help="Number of chunks to split the sequence into.")
    parser.add_argument("--clip",           type=float, default=1.0,
                        help="Gradient clipping (0 = disabled).")
    parser.add_argument("--warm-epochs",    type=int,   default=1,
                        help="Epochs to use uniform oracle before learned distribution.")
    parser.add_argument("--sync-rate",      type=int,   default=1)
    parser.add_argument("--firing-nb",      type=int,   default=10000)
    parser.add_argument("--use-tanh",       action="store_true", default=False)
    parser.add_argument("--train-samples",  type=int,   default=3000,
                        help="Number of training samples; 0 = full dataset.")
    parser.add_argument("--optim",          type=str,   default="adam",
                        choices=["adam", "sgd"])
    parser.add_argument("--dataset",        type=str,   default="mnist",
                        choices=["mnist", "smnist"],
                        help="Dataset variant: 'mnist' (data-derived norm) or 'smnist' (0.1307/0.3081 norm).")
    parser.add_argument("--smoke-test",     action="store_true", default=False,
                        help="Run the built-in smoke test and exit.")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(data_dir=args.data_dir)
        return

    x_train, y_train, x_test, y_test = load_mnist_arrays(args.batch_size, args.data_dir, dataset=args.dataset)

    print(
        f"FPTT training  dataset={args.dataset}\n"
        f"  seed={args.seed}  hidden={args.hidden_size}  epochs={args.epochs}\n"
        f"  batch={args.batch_size}  lr={args.lr}  alpha={args.alpha}  beta={args.beta}\n"
        f"  lmbda={args.lmbda}  rho={args.rho}  parts={args.parts}  clip={args.clip}\n"
        f"  sync_rate={args.sync_rate}  firing_nb={args.firing_nb}  use_tanh={args.use_tanh}\n"
        f"  train_samples={args.train_samples}  optim={args.optim}"
    )

    train_fptt(
        x_train=x_train,  y_train=y_train,
        x_test=x_test,    y_test=y_test,
        hidden_size=args.hidden_size,
        n_classes=10,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        lmbda=args.lmbda,
        rho=args.rho,
        PARTS=args.parts,
        clip=args.clip,
        seed=args.seed,
        sync_rate=args.sync_rate,
        firing_nb=args.firing_nb,
        use_tanh=args.use_tanh,
        train_samples=args.train_samples,
        warm_epochs=args.warm_epochs,
        optim_name=args.optim,
    )


if __name__ == "__main__":
    main()

# Quick commands:
# python async_RNN_fptt.py --smoke-test
# python async_RNN_fptt.py --epochs 10 --hidden-size 128 --train-samples 3000 --parts 10
# python async_RNN_fptt.py --epochs 20 --hidden-size 128 --use-tanh --firing-nb 1 --train-samples 0
