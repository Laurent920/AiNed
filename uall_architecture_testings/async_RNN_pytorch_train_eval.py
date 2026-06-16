import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

# Initialize PyTorch CUDA before JAX touches the GPU
_CUDA_AVAILABLE = torch.cuda.is_available()

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax.numpy as jnp  # noqa: E402

from async_RNN_pytorch_gradient_check import (  # noqa: E402
    TorchRuleRNN,
    init_feedforward_weights,
    init_recurrent_weight,
    keep_top_k_batch_torch,
)
from dataset_helpers.mnist_helper import mnist_loader_manual  # noqa: E402
from other_helpers.init_weights import init_params  # noqa: E402


class RTRLModel:
    """RNN with RTRL trace-based gradient computation (no autograd).

    Maintains parameters as plain tensors and computes gradients via
    forward-mode RTRL traces during the forward pass.  Supports:
      - exact W_hh trace  (H, H, H) per sample
      - exact bias trace  (H, H) per sample
      - W_ih: diagonal approximation (H^2 exact trace is too large for MNIST)
      - W_out: exact (computed from output-layer residuals, same as autograd)

    All computation is pure PyTorch so it runs on GPU.
    """

    def __init__(self, w_ih, w_hh, w_out, bias_h, sync_rate, firing_nb,
                 use_tanh=False, is_smnist=False, device="cpu"):
        self.device = torch.device(device)
        self.w_ih = torch.tensor(np.asarray(w_ih), dtype=torch.float32, device=self.device)
        self.w_hh = torch.tensor(np.asarray(w_hh), dtype=torch.float32, device=self.device)
        self.w_out = torch.tensor(np.asarray(w_out), dtype=torch.float32, device=self.device)
        self.bias_h = torch.tensor(np.asarray(bias_h), dtype=torch.float32, device=self.device)
        self.sync_rate = int(sync_rate)
        self.firing_nb = int(firing_nb)
        self.use_tanh = bool(use_tanh)
        self.is_smnist = bool(is_smnist)

        # Collect params for optimizer
        self.params_list = [self.w_ih, self.w_hh, self.w_out, self.bias_h]

    def forward_with_rtrl(self, x):
        """Forward pass that returns logits AND RTRL gradient accumulators."""
        B, in_dim = x.shape
        H = self.w_ih.shape[1]
        C = self.w_out.shape[1]
        dev = x.device

        state = torch.zeros(B, H, device=dev)
        o_prev = torch.zeros(B, H, device=dev)
        logits = torch.zeros(B, C, device=dev)
        out_residuals = torch.zeros(B, H, device=dev)  # for W_out gradient

        eye_h = torch.eye(H, device=dev)
        w_hh_diag = torch.diag(self.w_hh)  # (H,)

        # W_ih trace: diagonal approx (B, n_input, H)
        n_input = 1 if self.is_smnist else in_dim
        ih_running = torch.zeros(B, n_input, H, device=dev)
        ih_total = torch.zeros(B, n_input, H, device=dev)

        # Bias trace: exact (B, H, H)
        bias_running = torch.zeros(B, H, H, device=dev)
        bias_total = torch.zeros(B, H, H, device=dev)

        # W_hh trace: exact (B, H, H, H)
        P_hh = torch.zeros(B, H, H, H, device=dev)
        T_hh = torch.zeros(B, H, H, H, device=dev)

        prev_active = torch.zeros(B, H, device=dev)
        prev_td = torch.ones(B, H, device=dev)

        num_steps = in_dim
        for t in range(num_steps):
            x_t = x[:, t:t+1]  # (B, 1)
            if self.is_smnist:
                w_t = self.w_ih[0].unsqueeze(0)  # (1, H)
                neuron_t = 0
            else:
                w_t = self.w_ih[t].unsqueeze(0)  # (1, H)
                neuron_t = t

            activations = x_t * w_t + state + self.bias_h.unsqueeze(0) + (o_prev @ self.w_hh)

            if self.use_tanh:
                tanh_out = torch.tanh(activations)
                td = (tanh_out > 0).float() * (1.0 - tanh_out ** 2)
                o_t = torch.relu(tanh_out)
            else:
                td = torch.ones_like(activations)
                o_t = torch.relu(activations)

            sync_fire = 1.0 if ((t + 1) % self.sync_rate == 0) else 0.0
            o_t = o_t * sync_fire
            o_t = keep_top_k_batch_torch(o_t, self.firing_nb)

            m_t = (o_t > 0).float()
            m_t_eff = m_t * td  # (B, H)

            logits = logits + o_t @ self.w_out
            out_residuals = out_residuals + o_t

            # --- A matrix: (1 - prev_m_eff)*I + prev_m_eff*W_hh ---
            prev_m_eff = prev_active * prev_td  # (B, H)
            A = (1.0 - prev_m_eff).unsqueeze(2) * eye_h.unsqueeze(0) + prev_m_eff.unsqueeze(2) * self.w_hh.unsqueeze(0)
            # A shape: (B, H, H)

            # --- W_ih trace (diagonal approx) ---
            A_diag = (1.0 - prev_m_eff) + prev_m_eff * w_hh_diag.unsqueeze(0)  # (B, H)
            ih_running = ih_running * A_diag.unsqueeze(1)  # (B, n_input, H)
            ih_running[:, neuron_t, :] = ih_running[:, neuron_t, :] + x[:, t].unsqueeze(1)
            ih_total = ih_total + ih_running * m_t_eff.unsqueeze(1)

            # --- Bias trace (exact, H x H) ---
            bias_running = torch.bmm(bias_running, A) + eye_h.unsqueeze(0)
            bias_total = bias_total + bias_running * m_t_eff.unsqueeze(1)

            # --- W_hh trace (exact, H x H x H) ---
            # P_hh[b,m,:,:] @ A[b,:,:] for each b,m — use matmul broadcasting
            # P_hh: (B, H, H, H), A: (B, H, H) -> A: (B, 1, H, H)
            P_hh = torch.matmul(P_hh, A.unsqueeze(1))  # (B, H, H, H)
            P_hh = P_hh + o_prev.unsqueeze(2).unsqueeze(3) * eye_h.unsqueeze(0).unsqueeze(0)
            T_hh = T_hh + P_hh * m_t_eff.unsqueeze(1).unsqueeze(2)

            state = activations - o_t
            o_prev = o_t
            prev_active = m_t
            prev_td = td

        return logits, out_residuals, ih_total, T_hh, bias_total

    def compute_grads(self, x, y):
        """Run forward + compute all parameter gradients via RTRL. Returns loss and grads dict."""
        B = x.shape[0]
        C = self.w_out.shape[1]

        logits, out_residuals, ih_total, T_hh, bias_total = self.forward_with_rtrl(x)

        # --- Loss + dlogits ---
        log_probs = torch.log_softmax(logits, dim=1)
        loss = torch.nn.functional.cross_entropy(logits, y)
        targets_oh = torch.zeros_like(logits)
        targets_oh.scatter_(1, y.unsqueeze(1), 1.0)
        dlogits = (torch.softmax(logits, dim=1) - targets_oh) / B  # (B, C)

        # --- W_out gradient ---
        # out_residuals (B, H), dlogits (B, C)
        grad_out = torch.einsum("bh,bc->hc", out_residuals, dlogits)  # (H, C)

        # --- next_grad: dL/dactivations through W_out ---
        out_grad = dlogits @ self.w_out.T  # (B, H)

        # --- W_hh gradient (exact) ---
        grad_hh = torch.einsum("bj,bmnj->mn", out_grad, T_hh)  # (H, H)

        # --- W_ih gradient (diagonal approx) ---
        grad_ih = torch.einsum("bj,bkj->kj", out_grad, ih_total)  # (n_input, H)

        # --- Bias gradient (exact) ---
        grad_bias = torch.einsum("bj,bnj->n", out_grad, bias_total)  # (H,)

        return loss, {"w_ih": grad_ih, "w_hh": grad_hh, "w_out": grad_out, "bias_h": grad_bias}

    @torch.no_grad()
    def predict(self, x):
        """Inference-only forward (no traces)."""
        B, in_dim = x.shape
        H = self.w_ih.shape[1]
        C = self.w_out.shape[1]
        dev = x.device

        state = torch.zeros(B, H, device=dev)
        o_prev = torch.zeros(B, H, device=dev)
        logits = torch.zeros(B, C, device=dev)

        for t in range(in_dim):
            x_t = x[:, t:t+1]
            w_t = self.w_ih[0].unsqueeze(0) if self.is_smnist else self.w_ih[t].unsqueeze(0)
            activations = x_t * w_t + state + self.bias_h.unsqueeze(0) + (o_prev @ self.w_hh)
            if self.use_tanh:
                tanh_out = torch.tanh(activations)
                o_t = torch.relu(tanh_out)
            else:
                o_t = torch.relu(activations)
            sync_fire = 1.0 if ((t + 1) % self.sync_rate == 0) else 0.0
            o_t = o_t * sync_fire
            o_t = keep_top_k_batch_torch(o_t, self.firing_nb)
            logits = logits + o_t @ self.w_out
            state = activations - o_t
            o_prev = o_t

        return logits


def accuracy_rtrl(model, x: np.ndarray, y: np.ndarray, batch_size: int, device: str) -> float:
    dev = torch.device(device)
    correct = 0
    total = 0
    for s in range(0, x.shape[0], batch_size):
        e = min(s + batch_size, x.shape[0])
        xb = torch.from_numpy(x[s:e]).to(dtype=torch.float32, device=dev)
        yb = torch.from_numpy(y[s:e]).to(dtype=torch.long, device=dev)
        logits = model.predict(xb)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += yb.shape[0]
    return float(correct / total) if total > 0 else 0.0


def train_one_rtrl(
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
    device: str,
):
    dev = torch.device(device)
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

    model = RTRLModel(
        w_ih=w_ih, w_hh=w_hh, w_out=w_out, bias_h=bias_h,
        sync_rate=sync_rate, firing_nb=firing_nb,
        use_tanh=use_tanh, device=device,
    )

    # Adam optimizer state (manual, since we don't use nn.Module)
    adam_states = {}
    for name, p in zip(["w_ih", "w_hh", "w_out", "bias_h"], model.params_list):
        adam_states[name] = {
            "m": torch.zeros_like(p),
            "v": torch.zeros_like(p),
            "t": 0,
        }

    def adam_step(name, param, grad, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        s = adam_states[name]
        s["t"] += 1
        s["m"] = beta1 * s["m"] + (1 - beta1) * grad
        s["v"] = beta2 * s["v"] + (1 - beta2) * grad ** 2
        m_hat = s["m"] / (1 - beta1 ** s["t"])
        v_hat = s["v"] / (1 - beta2 ** s["t"])
        param.sub_(lr * m_hat / (v_hat.sqrt() + eps))

    n = x_train_mode.shape[0]
    epoch_logs = []
    for ep in range(epochs):
        t0 = time.time()
        perm = np.random.permutation(n)
        running_loss = 0.0
        seen = 0

        for s in range(0, n, train_batch):
            e = min(s + train_batch, n)
            idx = perm[s:e]
            xb = torch.from_numpy(x_train_mode[idx]).to(dtype=torch.float32, device=dev)
            yb = torch.from_numpy(y_train_mode[idx]).to(dtype=torch.long, device=dev)

            loss, grads = model.compute_grads(xb, yb)

            adam_step("w_ih", model.w_ih, grads["w_ih"], lr)
            adam_step("w_hh", model.w_hh, grads["w_hh"], lr)
            adam_step("w_out", model.w_out, grads["w_out"], lr)
            adam_step("bias_h", model.bias_h, grads["bias_h"], lr)

            bsz = int(yb.shape[0])
            running_loss += float(loss.item()) * bsz
            seen += bsz

        train_acc = accuracy_rtrl(model, x_train_mode, y_train_mode, eval_batch, device)
        test_acc = accuracy_rtrl(model, x_test_mode, y_test, eval_batch, device)
        dt = time.time() - t0
        avg_loss = running_loss / max(seen, 1)
        epoch_logs.append((ep + 1, avg_loss, train_acc, test_acc, dt))
        print(
            f"[rtrl] epoch={ep+1}/{epochs} loss={avg_loss:.6f} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} time_s={dt:.2f}",
            flush=True,
        )

    return model, epoch_logs


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


def accuracy(model, x: np.ndarray, y: np.ndarray, batch_size: int, device: str = "cpu") -> float:
    dev = torch.device(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for s in range(0, x.shape[0], batch_size):
            e = min(s + batch_size, x.shape[0])
            xb = torch.from_numpy(x[s:e]).to(dtype=torch.float32, device=dev)
            yb = torch.from_numpy(y[s:e]).to(dtype=torch.long, device=dev)
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
    device: str = "cpu",
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

    dev = torch.device(device)
    model = TorchRuleRNN(
        w_ih=w_ih,
        w_hh=w_hh,
        w_out=w_out,
        bias_h=bias_h,
        sync_rate=sync_rate,
        firing_nb=firing_nb,
        use_tanh=use_tanh,
    ).to(dev)

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
            xb = torch.from_numpy(x_train_mode[idx]).to(dtype=torch.float32, device=dev)
            yb = torch.from_numpy(y_train_mode[idx]).to(dtype=torch.long, device=dev)

            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

            bsz = int(yb.shape[0])
            running_loss += float(loss.item()) * bsz
            seen += bsz

        train_acc = accuracy(model, x_train_mode, y_train_mode, eval_batch, device)
        test_acc = accuracy(model, x_test_mode, y_test, eval_batch, device)
        dt = time.time() - t0
        avg_loss = running_loss / max(seen, 1)
        epoch_logs.append((ep + 1, avg_loss, train_acc, test_acc, dt))
        print(
            f"[autograd] epoch={ep+1}/{epochs} loss={avg_loss:.6f} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} time_s={dt:.2f}",
            flush=True,
        )

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
    parser.add_argument(
        "--rtrl",
        action="store_true",
        help="Use RTRL trace-based gradients instead of PyTorch autograd.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', or 'auto' (use GPU if available).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="mnist,smnist",
        help="Comma-separated list of dataset modes to train on (default: mnist,smnist).",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    set_seed(args.seed)
    x_train, y_train, x_test, y_test = load_base_mnist_arrays(args.train_batch, args.data_dir)

    train_samples = args.train_samples
    if train_samples <= 0:
        train_samples = x_train.shape[0]

    modes = [m.strip() for m in args.modes.split(",")]
    grad_method = "rtrl" if args.rtrl else "autograd"

    print(
        f"seed={args.seed} hidden={args.hidden_size} epochs={args.epochs} "
        f"train_batch={args.train_batch} eval_batch={args.eval_batch} lr={args.lr} "
        f"sync_rate={args.sync_rate} firing_nb={args.firing_nb} use_tanh={args.use_tanh} "
        f"train_samples={train_samples} grad={grad_method} device={device}",
        flush=True,
    )

    for mode in modes:
        train_fn = train_one_rtrl if args.rtrl else train_one
        _, logs = train_fn(
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
            device=device,
        )

        for ep, avg_loss, train_acc, test_acc, dt in logs:
            print(
                f"[{grad_method}] {mode} epoch={ep} loss={avg_loss:.6f} "
                f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} time_s={dt:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
# python async_RNN_pytorch_train_eval.py --use-tanh --epochs 5 --hidden-size 128

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
