import argparse

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn

from dataset_helpers.mnist_helper import mnist_loader_manual
from other_helpers.backpropagation import RNN_back_prop
from other_helpers.helpers import NeuronStates, Params, load_config_with_defaults
from other_helpers.loss_functions import loss_bpp, loss_func


def random_layer_params(m, n, key, scale=1e-2):
    w_key, _ = jax.random.split(key)
    return scale * jax.random.normal(w_key, (n, m))


def init_feedforward_weights(layer_sizes, seed):
    key = jax.random.key(seed)
    keys = jax.random.split(key, len(layer_sizes))
    weights = []
    for i in range(1, len(layer_sizes)):
        fan_in = layer_sizes[i - 1]
        std = jnp.sqrt(2.0 / fan_in)
        w = random_layer_params(layer_sizes[i], layer_sizes[i - 1], keys[i], scale=std)
        weights.append(w)
    return weights


def init_recurrent_weight(hidden_size, seed, gain=0.5):
    key = jax.random.key(seed + 12345)
    w = jax.random.normal(key, shape=(hidden_size, hidden_size), dtype=jnp.float32)
    return w * (gain / jnp.sqrt(hidden_size))


def load_mnist_inputs(num_inputs, batch_size, data_dir):
    (train_loader, _), _, _, _ = mnist_loader_manual(
        batch_size=batch_size,
        shuffle=False,
        preprocess=False,
        CNN_preprocess=False,
        downsample=False,
        sequential=False,
        permuted=False,
        data_dir=data_dir,
    )

    xs, ys = [], []
    for bx, by in train_loader:
        xs.append(np.asarray(bx, dtype=np.float32))
        ys.append(np.asarray(by, dtype=np.int64))
        if sum(arr.shape[0] for arr in xs) >= num_inputs:
            break

    x = np.concatenate(xs, axis=0)[:num_inputs]
    y = np.concatenate(ys, axis=0)[:num_inputs]
    return x, y


def one_hot(y, num_classes):
    return jax.nn.one_hot(jnp.asarray(y), num_classes=num_classes)


def keep_top_k_batch_jax(x, k):
    if k < 0 or k >= x.shape[1]:
        return x
    _, idx = jax.lax.top_k(x, k)
    mask = jnp.sum(jax.nn.one_hot(idx, x.shape[1], dtype=x.dtype), axis=1)
    return x * mask


def keep_top_k_batch_torch(x, k):
    if k < 0 or k >= x.shape[1]:
        return x
    _, idx = torch.topk(x, k=k, dim=1)
    mask = torch.zeros_like(x)
    mask.scatter_(1, idx, 1.0)
    return x * mask


# region custom (your backprop path)
def custom_rule_grads(
    x,
    y,
    w_ih,
    w_hh,
    w_out,
    bias_h,
    sync_rate,
    firing_nb,
    params,
    num_classes,
    grad_mode="trace",
    trace_source_mode="full",
    trace_lowrank_rank=8,
    use_tanh=False,
):
    batch_size, in_dim = x.shape
    hidden_dim = w_ih.shape[1]

    z_prev = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
    o_prev = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)

    logits = jnp.zeros((batch_size, num_classes), dtype=jnp.float32)
    out_input_residuals = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)

    # Accumulators expected by RNN_back_prop
    rnn_running_sum = jnp.zeros((batch_size, in_dim, hidden_dim), dtype=jnp.float32)
    rnn_total_sum = jnp.zeros((batch_size, in_dim, hidden_dim), dtype=jnp.float32)
    # Bias trace: same as W_ih trace but input is always 1, so shape (B, H)
    bias_running_sum = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
    bias_total_sum = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
    rnn_running_product = jnp.zeros((batch_size, hidden_dim, hidden_dim), dtype=jnp.float32)
    rnn_total_product_sum = jnp.zeros((batch_size, hidden_dim, hidden_dim), dtype=jnp.float32)
    prev_active = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
    prev_output = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)

    eye_h = jnp.eye(hidden_dim, dtype=jnp.float32)
    use_lowrank_trace = trace_source_mode in ("lowrank_full", "lowrank_full_prev_active")
    if use_lowrank_trace:
        if trace_lowrank_rank < 1:
            raise ValueError("trace_lowrank_rank must be >= 1 for lowrank trace mode")
        p_factors = jnp.zeros((batch_size, hidden_dim, trace_lowrank_rank), dtype=jnp.float32)
        q_factors = jnp.zeros((batch_size, hidden_dim, trace_lowrank_rank), dtype=jnp.float32)

    m_list = []
    tanh_deriv_list = []
    o_prev_list = []
    prev_active_eff = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
    for t in range(in_dim):
        x_t = x[:, t : t + 1]

        # inner_t = W_Ih[t] * x_t + z_{t-1} - O_{t-1} + W_hh * O_{t-1}
        inner_t = x_t * w_ih[t][None, :] + z_prev - o_prev + (o_prev @ w_hh) + bias_h[None, :]

        # z_t = tanh(inner_t) or inner_t; tanh_deriv used to scale Jacobian
        if use_tanh:
            z_t = jnp.tanh(inner_t)
            tanh_deriv = 1.0 - z_t ** 2  # (B, H), tanh'(inner_t)
        else:
            z_t = inner_t
            tanh_deriv = jnp.ones_like(z_t)

        # O_t = ReLU(z_t), then async constraints (sync + firing_nb)
        o_t = jax.nn.relu(z_t)
        sync_fire = jnp.asarray(1.0 if ((t + 1) % sync_rate == 0) else 0.0, dtype=o_t.dtype)
        o_t = o_t * sync_fire
        o_t = keep_top_k_batch_jax(o_t, firing_nb)

        logits = logits + o_t @ w_out
        out_input_residuals = out_input_residuals + o_t

        m_t = (o_t > 0).astype(jnp.float32)  # ReLU'(z_t) with sync/firing mask
        # effective derivative: ReLU'(z_t) * tanh'(inner_t)
        m_t_eff = m_t * tanh_deriv  # (B, H)
        m_list.append(m_t)
        tanh_deriv_list.append(tanh_deriv)
        o_prev_list.append(o_prev)

        # A_hat[b,j,l] = tanh'[b,j] * ((1-m[b,j])*eye[j,l] + m[b,j]*w_hh[j,l])
        # Use prev_active (binary m) and prev_tanh_deriv separately so non-firing units
        # get their identity row correctly scaled by tanh' (not left as 1).
        prev_tanh_deriv = tanh_deriv_list[t - 1] if t > 0 else jnp.ones((batch_size, hidden_dim))
        A_prev = prev_tanh_deriv[:, :, None] * (
            (1.0 - prev_active)[:, :, None] * eye_h[None]
            + prev_active[:, :, None] * w_hh[None]
        )
        w_hh_diag = jnp.diag(w_hh)  # shape (H,)
        # Diagonal: tanh'[j] * ((1-m[j]) + m[j]*w_hh[j,j])
        A_prev_diag = prev_tanh_deriv * ((1.0 - prev_active) + prev_active * w_hh_diag[None, :])
        # Each W_ih[t0,j] only directly affects z_{t0}[j], so propagate only along unit j's axis
        rnn_running_sum = rnn_running_sum * A_prev_diag[:, None, :]
        rnn_running_sum = rnn_running_sum.at[:, t, :].add(x[:, t][:, None])
        rnn_total_sum = rnn_total_sum + rnn_running_sum * m_t_eff[:, None, :]
        # Bias trace: same diagonal propagation, input is always 1
        bias_running_sum = bias_running_sum * A_prev_diag + 1.0  # (B, H)
        bias_total_sum = bias_total_sum + bias_running_sum * m_t_eff  # (B, H)

        # Exact recurrent-weight trace:
        # U_i = sum_{k=0}^{i-1} diag(R_k) * prod_{j=k+1}^{i-1} A_j
        # A_j = I + diag(ReLU'(z_j)) @ (W_hh - I)
        # Recurrence form: U_i = diag(R_{i-1}) + U_{i-1} @ A_{i-1}
        if use_lowrank_trace:
            # Fixed-rank compact trace:
            # Keep only the most recent `trace_lowrank_rank` source terms.
            # U_i approx sum_{r=1..R} p_r q_r^T, with q_r propagated by A_{i-1}^T.
            A_prev_t = A_prev.transpose(0, 2, 1)
            q_evolved = jnp.einsum("bnl,blr->bnr", A_prev_t, q_factors)
            if trace_source_mode == "lowrank_full":
                q_new = jnp.ones((batch_size, hidden_dim), dtype=prev_output.dtype)
            else:  # lowrank_full_prev_active
                q_new = prev_active
            p_new = prev_output
            if trace_lowrank_rank == 1:
                p_factors = p_new[:, :, None]
                q_factors = q_new[:, :, None]
            else:
                p_factors = jnp.concatenate(
                    [p_new[:, :, None], p_factors[:, :, : trace_lowrank_rank - 1]], axis=2
                )
                q_factors = jnp.concatenate(
                    [q_new[:, :, None], q_evolved[:, :, : trace_lowrank_rank - 1]], axis=2
                )
            recurrent_running_sum = jnp.einsum("bmr,bnr->bmn", p_factors, q_factors)
        else:
            if trace_source_mode == "diag":
                recurrent_source = jax.vmap(jnp.diag)(prev_output)
            elif trace_source_mode == "full":
                recurrent_source = prev_output[:, :, None] * jnp.ones(
                    (batch_size, 1, hidden_dim), dtype=prev_output.dtype
                )
            elif trace_source_mode == "full_prev_active":
                recurrent_source = prev_output[:, :, None] * prev_active[:, None, :]
            elif trace_source_mode == "full_current":
                recurrent_source = o_t[:, :, None] * jnp.ones(
                    (batch_size, 1, hidden_dim), dtype=o_t.dtype
                )
            elif trace_source_mode == "full_current_active":
                recurrent_source = o_t[:, :, None] * m_t[:, None, :]
            else:
                raise ValueError(f"Unknown trace_source_mode={trace_source_mode}")

            recurrent_running_sum = (
                jnp.einsum("bmh,bhl->bml", rnn_running_product, A_prev)
                + recurrent_source
            )

        rnn_total_product_sum = rnn_total_product_sum + recurrent_running_sum * m_t_eff[:, None, :]
        if not use_lowrank_trace:
            rnn_running_product = recurrent_running_sum

        z_prev = z_t
        o_prev = o_t
        prev_active = m_t
        prev_active_eff = m_t_eff  # scaled by tanh' for Jacobian
        prev_output = o_t

    targets = one_hot(y, num_classes)
    loss, dlogits = jax.value_and_grad(loss_func)(logits, targets)

    out_state = NeuronStates(input_residuals=out_input_residuals)
    out_grad, out_w_grad = jax.vmap(loss_bpp, in_axes=(None, 0, 0))(w_out, out_state, dlogits)
    grad_out = jnp.sum(out_w_grad, axis=0)

    if grad_mode == "trace":
        hidden_state = NeuronStates(
            rnn_total_sum=rnn_total_sum,
            rnn_total_product_sum=rnn_total_product_sum,
        )
        grad_ih, grad_hh, _ = RNN_back_prop(params, hidden_state, out_grad, layer_idx=1)
        # bias_total_sum[b,j] * out_grad[b,j], summed over batch
        grad_bias = jnp.sum(bias_total_sum * out_grad, axis=0)  # (H,)
    elif grad_mode == "reverse_bptt":
        # Exact reverse-time recurrence for:
        # inner_t = x_t * W_ih[t] + z_{t-1} - O_{t-1} + O_{t-1} @ W_hh
        # z_t = tanh(inner_t) if use_tanh else inner_t
        # O_t = ReLU(z_t)
        # Jacobian: A_hat_t = tanh'(inner_t) * ((1 - m_t)*I + m_t * W_hh)
        delta_next = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
        grad_ih = jnp.zeros((in_dim, hidden_dim), dtype=jnp.float32)
        grad_hh = jnp.zeros((hidden_dim, hidden_dim), dtype=jnp.float32)
        grad_bias = jnp.zeros((hidden_dim,), dtype=jnp.float32)

        for t in range(in_dim - 1, -1, -1):
            m_t = m_list[t]
            td = tanh_deriv_list[t]  # (B, H), tanh'(inner_t); ones when use_tanh=False
            # A_hat_t[b,h,l] = td[b,h] * ((1-m_t[b,h])*eye[h,l] + m_t[b,h]*W_hh[h,l])
            A_hat_t = td[:, :, None] * (
                (1.0 - m_t)[:, :, None] * eye_h[None]
                + m_t[:, :, None] * w_hh[None]
            )
            # m_t_eff = ReLU'(z_t) * tanh'(inner_t)
            m_t_eff = m_t * td
            delta_t = out_grad * m_t_eff + jnp.einsum(
                "bh,bhl->bl", delta_next, A_hat_t.transpose(0, 2, 1)
            )
            grad_ih = grad_ih.at[t, :].set(
                jnp.sum(delta_t * x[:, t][:, None], axis=0)
            )
            grad_hh = grad_hh + jnp.einsum("bm,bn->mn", o_prev_list[t], delta_t)
            grad_bias = grad_bias + jnp.sum(delta_t, axis=0)  # sum over batch
            delta_next = delta_t
    else:
        raise ValueError(f"Unknown grad_mode={grad_mode}")

    return float(loss), grad_ih, grad_hh, grad_out, grad_bias


# region pytorch autograd reference
class TorchRuleRNN(nn.Module):
    def __init__(self, w_ih, w_hh, w_out, bias_h, sync_rate, firing_nb, use_tanh=False):
        super().__init__()
        self.w_ih = nn.Parameter(torch.tensor(np.asarray(w_ih), dtype=torch.float32))
        self.w_hh = nn.Parameter(torch.tensor(np.asarray(w_hh), dtype=torch.float32))
        self.w_out = nn.Parameter(torch.tensor(np.asarray(w_out), dtype=torch.float32))
        self.bias_h = nn.Parameter(
            torch.tensor(np.asarray(bias_h), dtype=torch.float32), requires_grad=True
        )
        self.sync_rate = int(sync_rate)
        self.firing_nb = int(firing_nb)
        self.use_tanh = bool(use_tanh)

    def forward(self, x):
        batch_size, in_dim = x.shape
        hidden_dim = self.w_ih.shape[1]
        out_dim = self.w_out.shape[1]

        z_prev = torch.zeros((batch_size, hidden_dim), dtype=x.dtype, device=x.device)
        o_prev = torch.zeros((batch_size, hidden_dim), dtype=x.dtype, device=x.device)
        logits = torch.zeros((batch_size, out_dim), dtype=x.dtype, device=x.device)

        for t in range(in_dim):
            x_t = x[:, t : t + 1]
            inner_t = x_t * self.w_ih[t].unsqueeze(0) + z_prev - o_prev + (o_prev @ self.w_hh)
            inner_t = inner_t + self.bias_h.unsqueeze(0)
            z_t = torch.tanh(inner_t) if self.use_tanh else inner_t

            o_t = torch.relu(z_t)
            sync_fire = 1.0 if ((t + 1) % self.sync_rate == 0) else 0.0
            o_t = o_t * sync_fire
            o_t = keep_top_k_batch_torch(o_t, self.firing_nb)

            logits = logits + o_t @ self.w_out
            z_prev = z_t
            o_prev = o_t

        return logits


def pytorch_grads(
    x_np, y_np, w_ih, w_hh, w_out, bias_h, sync_rate, firing_nb, use_tanh=False
):
    model = TorchRuleRNN(
        w_ih=w_ih,
        w_hh=w_hh,
        w_out=w_out,
        bias_h=bias_h,
        sync_rate=sync_rate,
        firing_nb=firing_nb,
        use_tanh=use_tanh,
    )

    x_t = torch.tensor(x_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)

    logits = model(x_t)
    loss = nn.CrossEntropyLoss()(logits, y_t)
    loss.backward()

    grad_ih = model.w_ih.grad.detach().cpu().numpy()
    grad_hh = model.w_hh.grad.detach().cpu().numpy()
    grad_out = model.w_out.grad.detach().cpu().numpy()
    grad_bias = model.bias_h.grad.detach().cpu().numpy()
    return float(loss.item()), grad_ih, grad_hh, grad_out, grad_bias


# region metrics / utils
def similarity_metrics(a, b, eps=1e-12):
    a = a.reshape(-1)
    b = b.reshape(-1)
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    cosine = dot / (na * nb + eps)
    rel_l2 = float(np.linalg.norm(a - b) / (nb + eps))
    scale = dot / (float(np.dot(a, a)) + eps)
    rel_l2_scaled = float(np.linalg.norm((a * scale) - b) / (nb + eps))
    mae = float(np.mean(np.abs(a - b)))
    max_abs = float(np.max(np.abs(a - b)))
    return cosine, rel_l2, rel_l2_scaled, scale, mae, max_abs


def _parse_sync_rates(sync_rates_arg, sync_rate_arg, in_dim):
    if sync_rates_arg:
        vals = [int(v.strip()) for v in sync_rates_arg.split(",") if v.strip()]
    elif sync_rate_arg is not None:
        vals = [int(sync_rate_arg)]
    else:
        vals = [1, 2, 4, 8, in_dim]

    out = []
    seen = set()
    for v in vals:
        if v <= 0:
            continue
        if v > in_dim:
            continue
        if v not in seen:
            out.append(v)
            seen.add(v)
    if not out:
        out = [1, in_dim]
    return out


def build_params(cfg, layer_sizes, seed, batch_size, firing_nb, sync_rate, use_tanh=False):
    restrict = cfg.get("restrict", None)
    if restrict is None:
        restrict_tuple = tuple(-1.0 for _ in range(len(layer_sizes)))
    elif not isinstance(restrict, (tuple, list)):
        restrict_tuple = tuple(float(restrict) for _ in range(len(layer_sizes)))
    else:
        restrict_tuple = tuple(float(v) if v is not None else -1.0 for v in restrict)
        if len(restrict_tuple) < len(layer_sizes):
            restrict_tuple = restrict_tuple + (restrict_tuple[-1],) * (len(layer_sizes) - len(restrict_tuple))
        if len(restrict_tuple) > len(layer_sizes):
            restrict_tuple = restrict_tuple[: len(layer_sizes)]

    sparsity_impact = tuple(0.0 for _ in range(len(layer_sizes)))
    recurrence = cfg.get("recurrence", None)
    recurrence = tuple(recurrence) if recurrence is not None else None

    return Params(
        dataset=cfg["dataset"],
        random_seed=seed,
        layer_sizes=layer_sizes,
        init_thresholds=float(cfg.get("init_thresholds", 0.0)),
        num_epochs=1,
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        batch_size=batch_size,
        load_file=False,
        shuffle_activations=False,
        restrict=restrict_tuple,
        firing_nb=firing_nb,
        sync_rate=sync_rate,
        max_nonzero=layer_sizes[0],
        shuffle_input=False,
        threshold_lr=0.0,
        sparsity_impact=sparsity_impact,
        w_reg=0.0,
        rerun="",
        top_weights=int(cfg.get("top_weights", -1)),
        history_size=0,
        recurrence=recurrence,
        use_bias=False,
        use_tanh=use_tanh,
    )


def print_comparison(
    sync_rate,
    firing_nb,
    grad_mode,
    trace_source_mode,
    trace_lowrank_rank,
    custom_loss,
    torch_loss,
    g_ih,
    g_hh,
    g_out,
    g_bias,
    tg_ih,
    tg_hh,
    tg_out,
    tg_bias,
):
    print(f"\n{'=' * 78}")
    print(
        f"sync_rate={sync_rate}  firing_nb={firing_nb}  grad_mode={grad_mode}  "
        f"trace_source_mode={trace_source_mode}  trace_lowrank_rank={trace_lowrank_rank}"
    )
    print(f"custom_loss={custom_loss:.8f}, torch_loss={torch_loss:.8f}")
    print("Per-parameter gradient similarity (custom vs PyTorch):")

    metrics = [
        ("W_ih",   np.asarray(g_ih),   tg_ih),
        ("W_hh",   np.asarray(g_hh),   tg_hh),
        ("W_out",  np.asarray(g_out),  tg_out),
        ("bias_h", np.asarray(g_bias), tg_bias),
    ]
    cosines = []
    for name, c, t in metrics:
        cosine, rel_l2, rel_l2_scaled, scale, mae, max_abs = similarity_metrics(c, t)
        cosines.append(cosine)
        print(
            f"{name}: cosine={cosine:.8f}, rel_l2={rel_l2:.8e}, "
            f"rel_l2_scaled={rel_l2_scaled:.8e}, scale={scale:.8e}, "
            f"mae={mae:.8e}, max_abs={max_abs:.8e}"
        )

    print(
        "Overall similarity:\n"
        f"mean_cosine={float(np.mean(cosines)):.8f}, min_cosine={float(np.min(cosines)):.8f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Gradient check for async_RNN recurrence rules vs PyTorch autograd"
    )
    parser.add_argument("--config", type=str, default="configs/RNN_config.yaml")
    parser.add_argument("--num-inputs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--input-source", type=str, default="synthetic", choices=["synthetic", "mnist"])
    parser.add_argument("--layer-sizes", type=str, default=None, help="Override, e.g. '784,128,10'")
    parser.add_argument("--firing-nb", type=int, default=10000)
    parser.add_argument(
        "--grad-mode",
        type=str,
        default="trace",
        choices=["trace", "reverse_bptt", "both"],
        help="Custom JAX gradient mode used for comparison.",
    )
    parser.add_argument(
        "--trace-source-mode",
        type=str,
        default="full",
        choices=[
            "diag",
            "full",
            "full_prev_active",
            "full_current",
            "full_current_active",
            "lowrank_full",
            "lowrank_full_prev_active",
        ],
        help="Compact trace source term for W_hh when grad_mode includes 'trace'.",
    )
    parser.add_argument(
        "--trace-lowrank-rank",
        type=int,
        default=8,
        help="Rank used by lowrank trace modes.",
    )
    parser.add_argument("--sync-rate", type=int, default=None, help="Single sync rate to test")
    parser.add_argument(
        "--sync-rates",
        type=str,
        default=None,
        help="Comma-separated list, e.g. '1,2,4,8,16,784'. If set, overrides --sync-rate.",
    )
    parser.add_argument(
        "--use-tanh",
        action="store_true",
        default=False,
        help="Apply tanh before ReLU (both custom and PyTorch paths).",
    )
    args = parser.parse_args()

    cfg = load_config_with_defaults(args.config)
    layer_sizes = tuple(cfg["layer_sizes"])
    if args.layer_sizes is not None:
        layer_sizes = tuple(int(v.strip()) for v in args.layer_sizes.split(",") if v.strip())

    if len(layer_sizes) != 3:
        raise ValueError(
            f"Checker supports one hidden recurrent layer (input, hidden, output). Got {layer_sizes}."
        )

    in_dim, hidden_dim, out_dim = layer_sizes

    if args.input_source == "mnist":
        if in_dim != 784:
            raise ValueError(f"MNIST input-source expects in_dim=784, got {in_dim}")
        x_np, y_np = load_mnist_inputs(
            args.num_inputs,
            batch_size=max(64, args.num_inputs),
            data_dir=args.data_dir,
        )
    else:
        rng = np.random.default_rng(args.seed)
        x_np = rng.normal(size=(args.num_inputs, in_dim)).astype(np.float32)
        y_np = rng.integers(low=0, high=out_dim, size=(args.num_inputs,), dtype=np.int64)

    x_jnp = jnp.asarray(x_np, dtype=jnp.float32)

    weights = init_feedforward_weights(layer_sizes, args.seed)
    w_ih, w_out = weights[0], weights[1]
    w_hh = init_recurrent_weight(hidden_dim, args.seed, gain=0.5)
    rng_bias = np.random.default_rng(args.seed + 1)
    bias_h = jnp.asarray(rng_bias.normal(size=(hidden_dim,)).astype(np.float32) * 0.1)

    sync_rates = _parse_sync_rates(args.sync_rates, args.sync_rate, in_dim)
    params = build_params(
        cfg=cfg,
        layer_sizes=layer_sizes,
        seed=args.seed,
        batch_size=args.num_inputs,
        firing_nb=args.firing_nb,
        sync_rate=1,
        use_tanh=args.use_tanh,
    )

    print(
        f"tested_inputs={args.num_inputs}\n"
        f"input_source={args.input_source}\n"
        f"layer_sizes={layer_sizes}\n"
        f"sync_rates={sync_rates}\n"
        f"firing_nb={args.firing_nb}\n"
        f"use_tanh={args.use_tanh}"
    )

    grad_modes = ["trace", "reverse_bptt"] if args.grad_mode == "both" else [args.grad_mode]
    for sync_rate in sync_rates:
        torch_loss, tg_ih, tg_hh, tg_out, tg_bias = pytorch_grads(
            x_np=x_np,
            y_np=y_np,
            w_ih=w_ih,
            w_hh=w_hh,
            w_out=w_out,
            bias_h=bias_h,
            sync_rate=sync_rate,
            firing_nb=args.firing_nb,
            use_tanh=args.use_tanh,
        )

        for grad_mode in grad_modes:
            custom_loss, g_ih, g_hh, g_out, g_bias = custom_rule_grads(
                x=x_jnp,
                y=y_np,
                w_ih=w_ih,
                w_hh=w_hh,
                w_out=w_out,
                bias_h=bias_h,
                sync_rate=sync_rate,
                firing_nb=args.firing_nb,
                params=params,
                num_classes=out_dim,
                grad_mode=grad_mode,
                trace_source_mode=args.trace_source_mode,
                trace_lowrank_rank=args.trace_lowrank_rank,
                use_tanh=args.use_tanh,
            )

            print_comparison(
                sync_rate=sync_rate,
                firing_nb=args.firing_nb,
                grad_mode=grad_mode,
                trace_source_mode=args.trace_source_mode,
                trace_lowrank_rank=args.trace_lowrank_rank,
                custom_loss=custom_loss,
                torch_loss=torch_loss,
                g_ih=g_ih,
                g_hh=g_hh,
                g_out=g_out,
                g_bias=g_bias,
                tg_ih=tg_ih,
                tg_hh=tg_hh,
                tg_out=tg_out,
                tg_bias=tg_bias,
            )


if __name__ == "__main__":
    main()
# python async_RNN_pytorch_gradient_check.py   --num-inputs 320   --input-source mnist   --layer-sizes 784,128,10   --sync-rates 1,4   --firing-nb 10000   --grad-mode trace   --trace-source-mode full
