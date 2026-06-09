import argparse
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn

from dataset_helpers.mnist_helper import mnist_loader_manual
from forward_backward_pass.backpropagation import MLP_back_prop
from other_helpers.helpers import Params, NeuronStates, load_config_with_defaults
from forward_backward_pass.loss_functions import loss_bpp, loss_func
from other_helpers.init_weights import init_params


def random_layer_params(m, n, key, scale=1e-2):
    w_key, _ = jax.random.split(key)
    return scale * jax.random.normal(w_key, (n, m))


def init_weights_from_async(layer_sizes, seed):
    key = jax.random.key(seed)
    keys = jax.random.split(key, len(layer_sizes))
    weights = []
    for i in range(1, len(layer_sizes)):
        fan_in = layer_sizes[i - 1]
        std = jnp.sqrt(2.0 / fan_in)
        w = random_layer_params(layer_sizes[i], layer_sizes[i - 1], keys[i], scale=std)
        weights.append(w)
    return weights


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


def forward_with_states(x, weights, biases, sync_rate):
    activations = x
    hidden_states = []

    for li, (w, b) in enumerate(zip(weights, biases)):
        z = activations @ w + b
        is_last = li == (len(weights) - 1)

        if is_last:
            logits = z
            out_state = NeuronStates(input_residuals=activations)
            return logits, hidden_states, out_state

        a = jnp.maximum(z, 0.0)
        bsz, in_dim = activations.shape

        state = NeuronStates(
            input_residuals=activations,
            input_vector=jnp.broadcast_to(jnp.arange(1, in_dim + 1), (bsz, in_dim)),
            output_vector=jnp.where(a > 0, sync_rate, 0),
            layer_activity=(a > 0).astype(jnp.int32),
            thresholds=jnp.zeros_like(a),
        )
        hidden_states.append(state)
        activations = a

    raise RuntimeError("Unexpected empty weight list")


def propagate_next_grad(next_grad, w, weight_res, current_state, prev_state, mode):
    base = jnp.dot(next_grad, w.T)
    input_mask = (~jnp.all(weight_res == 0, axis=2)).astype(base.dtype)
    cur_relu_mask = (current_state.output_vector > 0).astype(next_grad.dtype)
    prev_relu_mask = (prev_state.output_vector > 0).astype(base.dtype)

    if mode == "current":
        return base * input_mask
    if mode == "no_mask":
        return base
    if mode == "input_mask":
        return base * input_mask
    if mode == "relu_only":
        return jnp.dot(next_grad * cur_relu_mask, w.T)
    if mode == "relu_plus_input_mask":
        return jnp.dot(next_grad * cur_relu_mask, w.T) * input_mask
    if mode == "relu_prev":
        return base * prev_relu_mask
    raise ValueError(
        f"Unknown propagation mode '{mode}'. "
        "Choose from: current,no_mask,input_mask,relu_only,relu_plus_input_mask,relu_prev"
    )


def custom_async_like_grads(x, y, weights, biases, layer_sizes, params, prop_mode="current"):
    logits, hidden_states, out_state = forward_with_states(x, weights, biases, params.sync_rate)
    targets = one_hot(y, layer_sizes[-1])

    loss, dlogits = jax.value_and_grad(loss_func)(logits, targets)

    out_grad, out_w_grad = jax.vmap(loss_bpp, in_axes=(None, 0, 0))(weights[-1], out_state, dlogits)
    w_grads = [None] * len(weights)
    b_grads = [None] * len(biases)
    w_grads[-1] = jnp.sum(out_w_grad, axis=0)
    # bias grad for output layer: sum over batch of dlogits
    b_grads[-1] = jnp.sum(dlogits, axis=0)
    custom_deltas = {}

    next_grad = out_grad
    for hidden_pos in range(len(hidden_states) - 1, -1, -1):
        layer_idx = hidden_pos + 1
        custom_deltas[layer_idx] = next_grad
        weight_grad, _th_grad, weight_res = MLP_back_prop(params, hidden_states[hidden_pos], next_grad, layer_idx)
        w_grads[hidden_pos] = weight_grad[0]
        # bias grad: next_grad masked by whether the neuron fired (output_vector > 0), summed over batch
        neuron_fired = (hidden_states[hidden_pos].output_vector > 0).astype(next_grad.dtype)  # (B, out_dim)
        b_grads[hidden_pos] = jnp.sum(next_grad * neuron_fired, axis=0)

        if hidden_pos > 0:
            next_grad = propagate_next_grad(
                next_grad=next_grad,
                w=weights[hidden_pos],
                weight_res=weight_res,
                current_state=hidden_states[hidden_pos],
                prev_state=hidden_states[hidden_pos - 1],
                mode=prop_mode,
            )

    return float(loss), w_grads, b_grads, custom_deltas


class TorchMLP(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.layers = nn.ModuleList()
        for w in weights:
            in_dim, out_dim = w.shape
            layer = nn.Linear(in_dim, out_dim, bias=True)
            self.layers.append(layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def pytorch_grads(x_np, y_np, weights, biases):
    torch_model = TorchMLP(weights)
    with torch.no_grad():
        for layer, w, b in zip(torch_model.layers, weights, biases):
            layer.weight.copy_(torch.tensor(np.asarray(w.T), dtype=torch.float32))
            layer.bias.copy_(torch.tensor(np.asarray(b), dtype=torch.float32))

    x_t = torch.tensor(x_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)
    hidden_outputs = []
    x_cur = x_t
    for i, layer in enumerate(torch_model.layers):
        x_cur = layer(x_cur)
        if i < len(torch_model.layers) - 1:
            x_cur = torch.relu(x_cur)
            x_cur.retain_grad()
            hidden_outputs.append(x_cur)
    logits = x_cur
    loss = nn.CrossEntropyLoss()(logits, y_t)
    loss.backward()

    w_grads = []
    b_grads = []
    for layer in torch_model.layers:
        w_grads.append(layer.weight.grad.detach().cpu().numpy().T)
        b_grads.append(layer.bias.grad.detach().cpu().numpy())
    torch_deltas = {i + 1: h.grad.detach().cpu().numpy() for i, h in enumerate(hidden_outputs)}
    return float(loss.item()), w_grads, b_grads, torch_deltas


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


def main():
    parser = argparse.ArgumentParser(description="Compare async_MLP gradients to PyTorch gradients")
    parser.add_argument("--config", type=str, default="configs/MLP_config.yaml")
    parser.add_argument("--num-inputs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--debug-deltas", action="store_true", help="Compare hidden-layer dL/dh deltas")
    parser.add_argument(
        "--prop-modes",
        type=str,
        default="current",
        help="Comma-separated backprop propagation modes "
             "(current,no_mask,input_mask,relu_only,relu_plus_input_mask,relu_prev)",
    )
    parser.add_argument(
        "--layer-sizes",
        type=str,
        default=None,
        help="Optional override, e.g. '784,256,256,10'",
    )
    args = parser.parse_args()

    cfg = load_config_with_defaults(args.config)
    layer_sizes = tuple(cfg["layer_sizes"])
    if args.layer_sizes is not None:
        layer_sizes = tuple(int(x.strip()) for x in args.layer_sizes.split(",") if x.strip())

    if layer_sizes[0] != 784:
        raise ValueError(f"This validator expects first layer size 784, got {layer_sizes[0]}")

    sparsity_impact = tuple(0.0 for _ in range(max(len(layer_sizes), 1)))
    restrict = cfg["restrict"]
    if not isinstance(restrict, (tuple, list)):
        restrict = tuple(float(restrict) for _ in range(len(layer_sizes)))

    params = Params(
        dataset=cfg["dataset"],
        random_seed=args.seed,
        layer_sizes=layer_sizes,
        init_thresholds=cfg["init_thresholds"],
        num_epochs=1,
        learning_rate=cfg["learning_rate"],
        batch_size=args.num_inputs,
        load_file=False,
        shuffle_activations=False,
        restrict=tuple(restrict),
        firing_nb=10000,
        sync_rate=784,
        max_nonzero=784,
        shuffle_input=False,
        threshold_lr=0.0,
        sparsity_impact=sparsity_impact,
        w_reg=0.0,
        rerun="",
        top_weights=-1,
        history_size=0,
        use_bias=True,
    )

    x_np, y_np = load_mnist_inputs(args.num_inputs, batch_size=max(64, args.num_inputs), data_dir=args.data_dir)
    x = jnp.asarray(x_np, dtype=jnp.float32)

    weights = init_weights_from_async(layer_sizes, args.seed)
    rng_bias = np.random.default_rng(args.seed + 1)
    biases = [
        jnp.asarray(rng_bias.normal(size=(layer_sizes[i],)).astype(np.float32) * 0.1)
        for i in range(1, len(layer_sizes))
    ]

    torch_loss, torch_w_grads, torch_b_grads, torch_deltas = pytorch_grads(
        x_np, y_np, [np.asarray(w) for w in weights], [np.asarray(b) for b in biases]
    )
    print(f"firing_nb={params.firing_nb}, sync_rate={params.sync_rate}")
    print(f"tested_inputs={args.num_inputs}")
    print(f"torch_loss={torch_loss:.8f}")
    print(f"layer_sizes={layer_sizes}")

    prop_modes = [m.strip() for m in args.prop_modes.split(",") if m.strip()]
    for mode in prop_modes:
        custom_loss, custom_w_grads, custom_b_grads, custom_deltas = custom_async_like_grads(
            x, y_np, weights, biases, layer_sizes, params, prop_mode=mode
        )
        print(f"\n=== Propagation mode: {mode} ===")
        print(f"custom_loss={custom_loss:.8f}, torch_loss={torch_loss:.8f}")

        cosines = []
        print("Per-layer weight gradient similarity (custom vs PyTorch):")
        for i, (g_custom, g_torch) in enumerate(zip(custom_w_grads, torch_w_grads), start=1):
            g_custom_np = np.asarray(g_custom)
            cosine, rel_l2, rel_l2_scaled, scale, mae, max_abs = similarity_metrics(g_custom_np, g_torch)
            cosines.append(cosine)
            print(
                f"layer_{i}: cosine={cosine:.8f}, rel_l2={rel_l2:.8e}, "
                f"rel_l2_scaled={rel_l2_scaled:.8e}, scale={scale:.8e}, "
                f"mae={mae:.8e}, max_abs={max_abs:.8e}"
            )

        print("Per-layer bias gradient similarity (custom vs PyTorch):")
        for i, (g_custom, g_torch) in enumerate(zip(custom_b_grads, torch_b_grads), start=1):
            g_custom_np = np.asarray(g_custom)
            cosine, rel_l2, rel_l2_scaled, scale, mae, max_abs = similarity_metrics(g_custom_np, g_torch)
            cosines.append(cosine)
            print(
                f"bias_{i}: cosine={cosine:.8f}, rel_l2={rel_l2:.8e}, "
                f"rel_l2_scaled={rel_l2_scaled:.8e}, scale={scale:.8e}, "
                f"mae={mae:.8e}, max_abs={max_abs:.8e}"
            )

        mean_cos = float(np.mean(cosines)) if cosines else float("nan")
        min_cos = float(np.min(cosines)) if cosines else float("nan")
        print("Overall similarity:")
        print(f"mean_cosine={mean_cos:.8f}, min_cosine={min_cos:.8f}")

        if args.debug_deltas:
            print("Hidden-layer delta similarity (dL/dh, custom vs PyTorch):")
            hidden_cosines = []
            num_hidden = len(layer_sizes) - 2
            for hidden_idx in range(1, num_hidden + 1):
                d_custom = np.asarray(custom_deltas[hidden_idx])
                d_torch = np.asarray(torch_deltas[hidden_idx])
                cosine, rel_l2, rel_l2_scaled, scale, mae, max_abs = similarity_metrics(d_custom, d_torch)
                hidden_cosines.append(cosine)
                print(
                    f"hidden_{hidden_idx}: cosine={cosine:.8f}, rel_l2={rel_l2:.8e}, "
                    f"rel_l2_scaled={rel_l2_scaled:.8e}, scale={scale:.8e}, "
                    f"mae={mae:.8e}, max_abs={max_abs:.8e}"
                )
            if hidden_cosines:
                print(
                    f"delta_summary: mean_cosine={np.mean(hidden_cosines):.8f}, "
                    f"min_cosine={np.min(hidden_cosines):.8f}"
                )


if __name__ == "__main__":
    main()
