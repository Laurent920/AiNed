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


def load_smnist_inputs(num_inputs, batch_size, data_dir):
    """Load SMNIST: returns x shape (N, 784) raw pixel values, y shape (N,).
    The sequential preprocessing (index, value) pairs is handled in the forward loop.
    """
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

    x = np.concatenate(xs, axis=0)[:num_inputs]  # (N, 784), values in [0, 1]
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
    use_tanh=False,
    is_smnist=False,
    exact_hh_trace=False,  # If True, maintain full H^3 RTRL trace for W_hh (exact but expensive)
    exact_ih_trace=False,  # If True, maintain full (n_input, H, H) RTRL trace for W_ih + (H, H) for bias
):
    batch_size, in_dim = x.shape
    hidden_dim = w_ih.shape[1]
    # For SMNIST: x is (B, T=784) pixel values, w_ih is (1, H).
    # The loop runs T steps, always using w_ih[0].
    # rnn_running_sum tracks only 1 input neuron, shape (B, 1, H).
    num_input_neurons = w_ih.shape[0]  # 1 for smnist, in_dim for mnist

    # State mirrors async_RNN: values = activations - penalty (pre-tanh residual)
    state = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
    o_prev = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)

    logits = jnp.zeros((batch_size, num_classes), dtype=jnp.float32)
    out_input_residuals = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)

    # Accumulators expected by RNN_back_prop — diagonal approx, shape (B, n_input, n_hidden)
    rnn_running_sum = jnp.zeros((batch_size, num_input_neurons, hidden_dim), dtype=jnp.float32)
    rnn_total_sum = jnp.zeros((batch_size, num_input_neurons, hidden_dim), dtype=jnp.float32)
    # Bias trace: same as W_ih trace but input is always 1, so shape (B, H)
    bias_running_sum = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
    bias_total_sum = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
    rnn_running_product = jnp.zeros((batch_size, hidden_dim, hidden_dim), dtype=jnp.float32)
    rnn_total_product_sum = jnp.zeros((batch_size, hidden_dim, hidden_dim), dtype=jnp.float32)
    # Exact H^3 RTRL trace for W_hh: P_t[b, m, n, j] = dactivations_t[j]/dW_hh[m,n]
    if exact_hh_trace:
        P_trace = jnp.zeros((batch_size, hidden_dim, hidden_dim, hidden_dim), dtype=jnp.float32)
        T3_accum = jnp.zeros((batch_size, hidden_dim, hidden_dim, hidden_dim), dtype=jnp.float32)
    # Exact (n_input, H, H) trace for W_ih and (H, H) for bias
    if exact_ih_trace:
        # Q_t[b, k, n, j] = dactivations_t[j] / dW_ih[k, n]
        Q_ih_trace = jnp.zeros((batch_size, num_input_neurons, hidden_dim, hidden_dim), dtype=jnp.float32)
        T_ih_accum = jnp.zeros((batch_size, num_input_neurons, hidden_dim, hidden_dim), dtype=jnp.float32)
        # R_t[b, n, j] = dactivations_t[j] / dbias[n]
        Q_bias_trace = jnp.zeros((batch_size, hidden_dim, hidden_dim), dtype=jnp.float32)
        T_bias_accum = jnp.zeros((batch_size, hidden_dim, hidden_dim), dtype=jnp.float32)
    prev_active = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
    prev_output = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)

    eye_h = jnp.eye(hidden_dim, dtype=jnp.float32)

    m_list = []
    tanh_deriv_list = []
    o_prev_list = []
    prev_active_eff = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
    # For SMNIST: loop over T=784 timesteps, always use w_ih[0] (neuron_idx=0)
    # For MNIST: loop over in_dim input neurons, use w_ih[t] for timestep t
    num_steps = in_dim  # number of loop iterations
    for t in range(num_steps):
        if is_smnist:
            x_t = x[:, t : t + 1]   # (B, 1) pixel value at timestep t
            w_t = w_ih[0][None, :]   # (1, H) — always neuron 0
            neuron_t = 0             # rnn_running_sum index
        else:
            x_t = x[:, t : t + 1]   # (B, 1)
            w_t = w_ih[t][None, :]   # (1, H)
            neuron_t = t

        # async_RNN dynamics: activations = x*w + state + bias + o_prev @ W_hh
        activations = x_t * w_t + state + bias_h[None, :] + (o_prev @ w_hh)

        # z_t = tanh(activations) or activations; tanh_deriv at current step
        if use_tanh:
            tanh_out = jnp.tanh(activations)
            # tanh_deriv_curr: derivative of relu(tanh(z)) — nonzero only where tanh > 0
            tanh_deriv = (tanh_out > 0).astype(tanh_out.dtype) * (1.0 - tanh_out ** 2)
            o_t = jax.nn.relu(tanh_out)
        else:
            tanh_deriv = jnp.ones_like(activations)
            o_t = jax.nn.relu(activations)

        sync_fire = jnp.asarray(1.0 if ((t + 1) % sync_rate == 0) else 0.0, dtype=o_t.dtype)
        o_t = o_t * sync_fire
        o_t = keep_top_k_batch_jax(o_t, firing_nb)

        logits = logits + o_t @ w_out
        out_input_residuals = out_input_residuals + o_t

        m_t = (o_t > 0).astype(jnp.float32)
        # effective derivative: ReLU'(z_t) * tanh'(activations) — matches active_tanh in async_RNN
        m_t_eff = m_t * tanh_deriv  # (B, H)
        m_list.append(m_t)
        tanh_deriv_list.append(tanh_deriv)
        o_prev_list.append(o_prev)

        # A matrix: Jacobian dactivations_t / dactivations_{t-1}
        # Exact: A[k,j] = (1 - m_eff[k])*delta_{kj} + m_eff[k]*W_hh[k,j]
        # where m_eff = prev_active * prev_tanh_deriv (derivative of relu(tanh) w.r.t. activations)
        prev_td = tanh_deriv_list[t - 1] if t > 0 else jnp.ones((batch_size, hidden_dim))
        prev_m_eff = prev_active * prev_td  # (B, H)
        A_prev = (
            (1.0 - prev_m_eff)[:, :, None] * eye_h[None]
            + prev_m_eff[:, :, None] * w_hh[None]
        )
        w_hh_diag = jnp.diag(w_hh)  # shape (H,)
        A_prev_diag = (1.0 - prev_m_eff) + prev_m_eff * w_hh_diag[None, :]
        # W_ih trace: diagonal approx, shape (B, n_input, H)
        rnn_running_sum = rnn_running_sum * A_prev_diag[:, None, :]
        rnn_running_sum = rnn_running_sum.at[:, neuron_t, :].add(x[:, t][:, None])
        rnn_total_sum = rnn_total_sum + rnn_running_sum * m_t_eff[:, None, :]
        # Bias trace: same diagonal propagation, input is always 1
        bias_running_sum = bias_running_sum * A_prev_diag + 1.0  # (B, H)
        bias_total_sum = bias_total_sum + bias_running_sum * m_t_eff  # (B, H)

        # W_hh compact trace: U_i = source(R_{i-1}) + U_{i-1} @ A_{i-1}
        if trace_source_mode == "diag":
            recurrent_source = jax.vmap(jnp.diag)(prev_output)
        elif trace_source_mode == "full":
            recurrent_source = prev_output[:, :, None] * jnp.ones(
                (batch_size, 1, hidden_dim), dtype=prev_output.dtype
            )
        else:
            raise ValueError(f"Unknown trace_source_mode={trace_source_mode}")

        recurrent_running_sum = (
            jnp.einsum("bmh,bhl->bml", rnn_running_product, A_prev)
            + recurrent_source
        )

        rnn_total_product_sum = rnn_total_product_sum + recurrent_running_sum * m_t_eff[:, None, :]
        rnn_running_product = recurrent_running_sum

        # Exact H^3 RTRL for W_hh
        if exact_hh_trace:
            # P_t[m,n,j] = dactivations_t[j] / dW_hh[m,n]
            # = sum_k P_{t-1}[m,n,k] * A_prev[k,j] + o_{t-1}[m] * delta_{nj}
            P_prop = jnp.einsum("bmnk,bkj->bmnj", P_trace, A_prev)
            source_exact = prev_output[:, :, None, None] * eye_h[None, None, :, :]  # (B, H, H, H)
            P_trace = P_prop + source_exact
            # Accumulate: T3[b,m,n,j] += m_t_eff[b,j] * P_t[b,m,n,j]
            T3_accum = T3_accum + P_trace * m_t_eff[:, None, None, :]

        # Exact (n_input, H, H) RTRL for W_ih and (H, H) for bias
        if exact_ih_trace:
            # Q_t[k, n, j] = dactivations_t[j] / dW_ih[k, n]
            # = x_t * delta_{k=neuron_t} * delta_{nj} + sum_l Q_{t-1}[k,n,l] * A_prev[l,j]
            Q_ih_trace = jnp.einsum("bknl,blj->bknj", Q_ih_trace, A_prev)
            # Source: x_t[b] * delta_{k=neuron_t} * I[n,j]
            x_t_val = x[:, t]  # (B,)
            src_ih = x_t_val[:, None, None] * eye_h[None, :, :]  # (B, H, H)
            Q_ih_trace = Q_ih_trace.at[:, neuron_t, :, :].add(src_ih)
            T_ih_accum = T_ih_accum + Q_ih_trace * m_t_eff[:, None, None, :]

            # R_t[n, j] = dactivations_t[j] / dbias[n]
            # = delta_{nj} + sum_l R_{t-1}[n,l] * A_prev[l,j]
            Q_bias_trace = jnp.einsum("bnl,blj->bnj", Q_bias_trace, A_prev) + eye_h[None, :, :]
            T_bias_accum = T_bias_accum + Q_bias_trace * m_t_eff[:, None, :]

        # async_RNN: new_values = activations - penalty (restrict=1 => penalty = o_t)
        state = activations - o_t
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
            bias_total_sum=bias_total_sum,
        )
        grad_ih, grad_hh, _, grad_bias = RNN_back_prop(params, hidden_state, out_grad, layer_idx=1)
        if exact_hh_trace:
            # Override W_hh gradient with exact H^3 RTRL result
            grad_hh = jnp.einsum("bj,bmnj->mn", out_grad, T3_accum)
        if exact_ih_trace:
            # Override W_ih gradient: grad_ih[k,n] = sum_b sum_j out_grad[b,j] * T_ih_accum[b,k,n,j]
            grad_ih = jnp.einsum("bj,bknj->kn", out_grad, T_ih_accum)
            # Override bias gradient: grad_bias[n] = sum_b sum_j out_grad[b,j] * T_bias_accum[b,n,j]
            grad_bias = jnp.einsum("bj,bnj->n", out_grad, T_bias_accum)
    elif grad_mode == "reverse_bptt":
        # Exact reverse-time recurrence for async_RNN dynamics:
        # activations_t = x_t * W_ih[neuron_t] + state_{t-1} + bias + o_{t-1} @ W_hh
        # tanh_out_t = tanh(activations_t)
        # o_t = topk(sync(relu(tanh_out_t)))
        # state_t = activations_t - o_t
        # Jacobian: A_t[j,l] = (1 - m_eff_t[j])*delta_{jl} + m_eff_t[j]*W_hh[j,l]
        #   where m_eff_t = m_t * tanh_deriv_t
        delta_next = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
        grad_ih = jnp.zeros((num_input_neurons, hidden_dim), dtype=jnp.float32)
        grad_hh = jnp.zeros((hidden_dim, hidden_dim), dtype=jnp.float32)
        grad_bias = jnp.zeros((hidden_dim,), dtype=jnp.float32)

        for t in range(num_steps - 1, -1, -1):
            neuron_t = 0 if is_smnist else t
            m_t = m_list[t]
            td = tanh_deriv_list[t]  # (B, H), relu'(tanh)*tanh'(act); ones when use_tanh=False
            m_t_eff = m_t * td
            # A_t[j,l] = (1 - m_eff[j])*I[j,l] + m_eff[j]*W_hh[j,l]
            A_hat_t = (
                (1.0 - m_t_eff)[:, :, None] * eye_h[None]
                + m_t_eff[:, :, None] * w_hh[None]
            )
            delta_t = out_grad * m_t_eff + jnp.einsum(
                "bh,bhl->bl", delta_next, A_hat_t.transpose(0, 2, 1)
            )
            grad_ih = grad_ih.at[neuron_t, :].add(
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
    def __init__(self, w_ih, w_hh, w_out, bias_h, sync_rate, firing_nb, use_tanh=False, is_smnist=False):
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
        self.is_smnist = bool(is_smnist)

    def forward(self, x):
        batch_size, in_dim = x.shape
        hidden_dim = self.w_ih.shape[1]
        out_dim = self.w_out.shape[1]

        # State mirrors async_RNN: values = activations - penalty (pre-tanh residual)
        state = torch.zeros((batch_size, hidden_dim), dtype=x.dtype, device=x.device)
        o_prev = torch.zeros((batch_size, hidden_dim), dtype=x.dtype, device=x.device)
        logits = torch.zeros((batch_size, out_dim), dtype=x.dtype, device=x.device)

        num_steps = in_dim
        for t in range(num_steps):
            x_t = x[:, t : t + 1]
            # SMNIST: always use w_ih[0] (single input neuron)
            w_t = self.w_ih[0].unsqueeze(0) if self.is_smnist else self.w_ih[t].unsqueeze(0)

            # async_RNN: activations = x*w + values + bias + o_prev @ W_hh
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

            # async_RNN: new_values = activations - penalty (restrict=1 => penalty = o_t)
            state = activations - o_t
            o_prev = o_t

        return logits


def pytorch_grads(
    x_np, y_np, w_ih, w_hh, w_out, bias_h, sync_rate, firing_nb, use_tanh=False, is_smnist=False
):
    model = TorchRuleRNN(
        w_ih=w_ih,
        w_hh=w_hh,
        w_out=w_out,
        bias_h=bias_h,
        sync_rate=sync_rate,
        firing_nb=firing_nb,
        use_tanh=use_tanh,
        is_smnist=is_smnist,
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
        f"trace_source_mode={trace_source_mode}"
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
    parser.add_argument("--input-source", type=str, default="synthetic", choices=["synthetic", "mnist", "smnist"])
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
        choices=["diag", "full"],
        help="Compact trace source term for W_hh when grad_mode includes 'trace'.",
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
    parser.add_argument(
        "--exact-hh-trace",
        action="store_true",
        default=False,
        help="Use exact H^3 RTRL trace for W_hh (very expensive, for validation only).",
    )
    parser.add_argument(
        "--exact-ih-trace",
        action="store_true",
        default=False,
        help="Use exact (n_input, H, H) RTRL trace for W_ih and (H, H) for bias.",
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

    is_smnist = args.input_source == "smnist"

    if args.input_source == "mnist":
        if in_dim != 784:
            raise ValueError(f"MNIST input-source expects in_dim=784, got {in_dim}")
        x_np, y_np = load_mnist_inputs(
            args.num_inputs,
            batch_size=max(64, args.num_inputs),
            data_dir=args.data_dir,
        )
    elif args.input_source == "smnist":
        if in_dim != 1:
            raise ValueError(f"SMNIST input-source expects in_dim=1 (single input neuron), got {in_dim}")
        x_np, y_np = load_smnist_inputs(
            args.num_inputs,
            batch_size=max(64, args.num_inputs),
            data_dir=args.data_dir,
        )
        # x_np shape: (N, 784) — 784 pixel values per sample, used sequentially
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
        f"use_tanh={args.use_tanh}\n"
        f"exact_hh_trace={args.exact_hh_trace}\n"
        f"exact_ih_trace={args.exact_ih_trace}"
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
            is_smnist=is_smnist,
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
                use_tanh=args.use_tanh,
                is_smnist=is_smnist,
                exact_hh_trace=args.exact_hh_trace,
                exact_ih_trace=args.exact_ih_trace,
            )

            print_comparison(
                sync_rate=sync_rate,
                firing_nb=args.firing_nb,
                grad_mode=grad_mode,
                trace_source_mode=args.trace_source_mode,
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
# python async_RNN_pytorch_gradient_check.py --num-inputs 320 --input-source mnist --layer-sizes 784,128,10 --sync-rates 1,4 --firing-nb 10000 --grad-mode trace --trace-source-mode full --use-tanh
