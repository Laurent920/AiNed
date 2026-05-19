import argparse
import math

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_helpers.mnist_helper import mnist_loader_manual
from forward_backward_pass.backpropagation import MLP_back_prop
from other_helpers.helpers import NeuronStates, Params, load_config_with_defaults
from forward_backward_pass.loss_functions import loss_bpp, loss_func


def one_hot(y, num_classes):
    return jax.nn.one_hot(jnp.asarray(y), num_classes=num_classes)


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


def parse_layer_defs(layer_sizes):
    defs = []
    for cfg_idx, spec in enumerate(layer_sizes[1:], start=1):
        if len(spec) > 1:
            out_ch = int(spec[0])
            kernel = tuple(int(v) for v in spec[1])
            padding = tuple(int(v) for v in spec[2])
            stride = tuple(int(v) for v in spec[3])
            pooling = str(spec[4]) if len(spec) >= 5 else ""
            if pooling.lower() == "none":
                pooling = ""
            pool_size = tuple(int(v) for v in spec[5]) if len(spec) >= 6 else (2, 2)
            pool_stride = tuple(int(v) for v in spec[6]) if len(spec) >= 7 else (2, 2)
            defs.append(
                {
                    "kind": "conv",
                    "cfg_idx": cfg_idx,
                    "out_channels": out_ch,
                    "kernel": kernel,
                    "padding": padding,
                    "stride": stride,
                    "pooling": pooling,
                    "pool_size": pool_size,
                    "pool_stride": pool_stride,
                }
            )
        else:
            defs.append({"kind": "fc", "cfg_idx": cfg_idx, "units": int(spec[0])})
    return defs


def init_weights_from_async_cnn(layer_sizes, seed):
    layer_defs = parse_layer_defs(layer_sizes)
    key = jax.random.key(seed)
    keys = jax.random.split(key, len(layer_defs))

    weights = []
    biases = []
    rng_bias = np.random.default_rng(seed + 1)
    cur_shape = tuple(int(v) for v in layer_sizes[0])  # (C, H, W)
    for i, layer_def in enumerate(layer_defs):
        if layer_def["kind"] == "conv":
            out_ch = layer_def["out_channels"]
            in_ch = cur_shape[0]
            kh, kw = layer_def["kernel"]
            ph, pw = layer_def["padding"]
            sh, sw = layer_def["stride"]

            fan_in = in_ch * kh * kw
            bound = math.sqrt(2.0 / fan_in)
            w = jax.random.uniform(
                keys[i],
                (out_ch, in_ch, kh, kw),
                dtype=jnp.float32,
                minval=-bound,
                maxval=bound,
            )
            weights.append(w)
            biases.append(None)  # conv layers have no bias in this checker

            h_out = (cur_shape[1] + 2 * ph - kh) // sh + 1
            w_out = (cur_shape[2] + 2 * pw - kw) // sw + 1
            if layer_def["pooling"] != "":
                pkh, pkw = layer_def["pool_size"]
                psh, psw = layer_def["pool_stride"]
                h_out = (h_out - pkh) // psh + 1
                w_out = (w_out - pkw) // psw + 1
            cur_shape = (out_ch, h_out, w_out)
        else:
            in_dim = int(np.prod(cur_shape)) if len(cur_shape) == 3 else int(cur_shape[0])
            out_dim = layer_def["units"]
            w = 1e-2 * jax.random.normal(keys[i], (in_dim, out_dim), dtype=jnp.float32)
            weights.append(w)
            b = jnp.asarray(rng_bias.normal(size=(out_dim,)).astype(np.float32) * 0.1)
            biases.append(b)
            cur_shape = (out_dim,)
    return weights, biases, layer_defs


def apply_pool_jax(x, pooling, pool_size, pool_stride):
    if pooling == "max":
        return jax.lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, 1, pool_size[0], pool_size[1]),
            window_strides=(1, 1, pool_stride[0], pool_stride[1]),
            padding="VALID",
        )
    if pooling == "avg":
        out = jax.lax.reduce_window(
            x,
            init_value=0.0,
            computation=jax.lax.add,
            window_dimensions=(1, 1, pool_size[0], pool_size[1]),
            window_strides=(1, 1, pool_stride[0], pool_stride[1]),
            padding="VALID",
        )
        return out / float(pool_size[0] * pool_size[1])
    return x


def pool_backward_exact(pre_pool, pooled_grad, pooling, pool_size, pool_stride):
    if pooling == "":
        return pooled_grad

    def pool_only(t):
        return apply_pool_jax(t, pooling, pool_size, pool_stride)

    _, pullback = jax.vjp(pool_only, pre_pool)
    return pullback(pooled_grad)[0]


def pool_backward_repeat(pre_pool, pooled_grad, pooling, pool_size, pool_stride):
    if pooling == "":
        return pooled_grad

    sh, sw = pool_size
    grad = jnp.repeat(jnp.repeat(pooled_grad, sh, axis=2), sw, axis=3)
    target_h = pre_pool.shape[2]
    target_w = pre_pool.shape[3]
    grad = grad[:, :, :target_h, :target_w]
    pad_h = max(0, target_h - grad.shape[2])
    pad_w = max(0, target_w - grad.shape[3])
    return jnp.pad(grad, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))


def conv_weight_grad_single(x, dy, stride, padding):
    pad_x, pad_y = padding
    lhs = x[:, None, :, :]
    rhs = dy[:, None, :, :]
    return jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=stride,
        padding=((pad_x, pad_x), (pad_y, pad_y)),
        dimension_numbers=("NCHW", "OIHW", "CNHW"),
    )


def conv_input_grad_batch(dy, w, stride, padding):
    pad_x, pad_y = padding
    w_flipped = jnp.flip(w, axis=(2, 3))
    rhs = w_flipped.transpose(1, 0, 2, 3)
    k_h, k_w = w.shape[2], w.shape[3]
    pad_h = k_h - 1 - pad_x
    pad_w = k_w - 1 - pad_y
    return jax.lax.conv_general_dilated(
        lhs=dy,
        rhs=rhs,
        window_strides=(1, 1),
        padding=((pad_h, pad_h), (pad_w, pad_w)),
        lhs_dilation=stride,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )


def forward_with_states(x, weights, biases, layer_defs, sync_rate, fc_mask_value_mode):
    activations = x
    records = []

    for wi, (w, b, layer_def) in enumerate(zip(weights, biases, layer_defs)):
        if layer_def["kind"] == "conv":
            ph, pw = layer_def["padding"]
            z = jax.lax.conv_general_dilated(
                lhs=activations,
                rhs=w,
                window_strides=layer_def["stride"],
                padding=((ph, ph), (pw, pw)),
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            a = jnp.maximum(z, 0.0)
            pooled = apply_pool_jax(
                a,
                layer_def["pooling"],
                layer_def["pool_size"],
                layer_def["pool_stride"],
            )
            records.append(
                {
                    "kind": "conv",
                    "cfg_idx": layer_def["cfg_idx"],
                    "input_residuals": activations,
                    "pre_pool_activations": a,
                    "layer_activity": (a > 0).astype(jnp.int32),
                    "output_shape": pooled.shape,
                    "padding": layer_def["padding"],
                    "stride": layer_def["stride"],
                    "pooling": layer_def["pooling"],
                    "pool_size": layer_def["pool_size"],
                    "pool_stride": layer_def["pool_stride"],
                }
            )
            activations = pooled
            continue

        fc_input = activations.reshape(activations.shape[0], -1) if activations.ndim > 2 else activations
        z = fc_input @ w + b
        is_last = wi == (len(weights) - 1)

        if is_last:
            out_state = NeuronStates(input_residuals=fc_input)
            records.append({"kind": "output", "cfg_idx": layer_def["cfg_idx"], "out_state": out_state})
            return z, records

        a = jnp.maximum(z, 0.0)
        bsz, in_dim = fc_input.shape
        output_value = in_dim if fc_mask_value_mode == "input_dim" else sync_rate
        state = NeuronStates(
            input_residuals=fc_input,
            input_vector=jnp.broadcast_to(jnp.arange(1, in_dim + 1), (bsz, in_dim)),
            output_vector=jnp.where(a > 0, output_value, 0),
            layer_activity=(a > 0).astype(jnp.int32),
            thresholds=jnp.zeros_like(a),
        )
        records.append(
            {
                "kind": "fc",
                "cfg_idx": layer_def["cfg_idx"],
                "state": state,
                "output_shape": a.shape,
            }
        )
        activations = a

    raise RuntimeError("Unexpected network definition")


def custom_async_cnn_grads(
    x,
    y,
    weights,
    biases,
    layer_defs,
    params,
    num_classes,
    fc_prop_mode="weight_res",
    conv_grad_reduction="mean",
    fc_mask_value_mode="input_dim",
    pool_backprop_mode="exact",
):
    logits, records = forward_with_states(
        x,
        weights,
        biases,
        layer_defs,
        sync_rate=params.sync_rate if isinstance(params.sync_rate, int) else 1,
        fc_mask_value_mode=fc_mask_value_mode,
    )

    targets = one_hot(y, num_classes)
    loss, dlogits = jax.value_and_grad(loss_func)(logits, targets)

    out_state = records[-1]["out_state"]
    out_grad, out_w_grad = jax.vmap(loss_bpp, in_axes=(None, 0, 0))(weights[-1], out_state, dlogits)

    w_grads = [None] * len(weights)
    b_grads = [None] * len(biases)
    w_grads[-1] = jnp.sum(out_w_grad, axis=0)
    # bias grad for output FC layer: sum over batch of dlogits
    b_grads[-1] = jnp.sum(dlogits, axis=0)
    next_grad = out_grad

    for wi in range(len(weights) - 2, -1, -1):
        rec = records[wi]
        w = weights[wi]

        if rec["kind"] == "fc":
            weight_grad, _th_grad, weight_res = MLP_back_prop(params, rec["state"], next_grad, rec["cfg_idx"])
            w_grads[wi] = weight_grad[0]
            # bias grad: next_grad masked by whether the neuron fired (output_vector > 0), summed over batch
            neuron_fired = (rec["state"].output_vector > 0).astype(next_grad.dtype)  # (B, out_dim)
            b_grads[wi] = jnp.sum(next_grad * neuron_fired, axis=0)

            if wi > 0:
                if fc_prop_mode == "no_mask":
                    send_grad = jnp.dot(next_grad, w.T)
                elif fc_prop_mode == "relu_only":
                    cur_relu_mask = (rec["state"].output_vector > 0).astype(next_grad.dtype)
                    send_grad = jnp.dot(next_grad * cur_relu_mask, w.T)
                else:
                    send_grad = jnp.dot(next_grad, w.T)
                    send_grad = send_grad * (~jnp.all(weight_res == 0, axis=2))

                prev_rec = records[wi - 1]
                if prev_rec["kind"] == "conv":
                    send_grad = send_grad.reshape(prev_rec["output_shape"])
                next_grad = send_grad
            continue

        if rec["kind"] == "conv":
            conv_next_grad = next_grad
            if rec["pooling"] != "":
                if pool_backprop_mode == "repeat":
                    conv_next_grad = pool_backward_repeat(
                        rec["pre_pool_activations"],
                        conv_next_grad,
                        rec["pooling"],
                        rec["pool_size"],
                        rec["pool_stride"],
                    )
                else:
                    conv_next_grad = pool_backward_exact(
                        rec["pre_pool_activations"],
                        conv_next_grad,
                        rec["pooling"],
                        rec["pool_size"],
                        rec["pool_stride"],
                    )

            activity_mask = jnp.where(rec["layer_activity"] > 0, 1.0, 0.0)
            conv_next_grad = conv_next_grad * activity_mask

            weight_grad_batch = jax.vmap(conv_weight_grad_single, in_axes=(0, 0, None, None))(
                rec["input_residuals"], conv_next_grad, rec["stride"], rec["padding"]
            )
            if conv_grad_reduction == "sum":
                w_grads[wi] = jnp.sum(weight_grad_batch, axis=0)
            else:
                w_grads[wi] = jnp.mean(weight_grad_batch, axis=0)
            # conv layers have no bias in this checker
            b_grads[wi] = None

            if wi > 0:
                next_grad = conv_input_grad_batch(conv_next_grad, w, rec["stride"], rec["padding"])
            continue

        raise RuntimeError(f"Unexpected record kind: {rec['kind']}")

    return float(loss), w_grads, b_grads


class TorchCNN(nn.Module):
    def __init__(self, layer_defs, weights):
        super().__init__()
        self.layer_defs = layer_defs
        self.layers = nn.ModuleList()
        for w, layer_def in zip(weights, layer_defs):
            if layer_def["kind"] == "conv":
                out_ch, in_ch, kh, kw = w.shape
                layer = nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=(kh, kw),
                    stride=layer_def["stride"],
                    padding=layer_def["padding"],
                    bias=False,
                )
            else:
                in_dim, out_dim = w.shape
                layer = nn.Linear(in_dim, out_dim, bias=True)
            self.layers.append(layer)

    def forward(self, x):
        for i, (layer, layer_def) in enumerate(zip(self.layers, self.layer_defs)):
            if layer_def["kind"] == "conv":
                x = layer(x)
                x = torch.relu(x)
                if layer_def["pooling"] == "max":
                    x = F.max_pool2d(x, kernel_size=layer_def["pool_size"], stride=layer_def["pool_stride"])
                elif layer_def["pooling"] == "avg":
                    x = F.avg_pool2d(x, kernel_size=layer_def["pool_size"], stride=layer_def["pool_stride"])
            else:
                if x.dim() > 2:
                    x = torch.flatten(x, start_dim=1)
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = torch.relu(x)
        return x


def pytorch_grads(x_np, y_np, weights, biases, layer_defs):
    model = TorchCNN(layer_defs, weights)
    with torch.no_grad():
        for layer, w, b, layer_def in zip(model.layers, weights, biases, layer_defs):
            if layer_def["kind"] == "conv":
                layer.weight.copy_(torch.tensor(np.asarray(w), dtype=torch.float32))
            else:
                layer.weight.copy_(torch.tensor(np.asarray(w.T), dtype=torch.float32))
                layer.bias.copy_(torch.tensor(np.asarray(b), dtype=torch.float32))

    x_t = torch.tensor(x_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)
    logits = model(x_t)
    loss = nn.CrossEntropyLoss()(logits, y_t)
    loss.backward()

    w_grads = []
    b_grads = []
    for layer, layer_def in zip(model.layers, layer_defs):
        g = layer.weight.grad.detach().cpu().numpy()
        if layer_def["kind"] == "fc":
            g = g.T
            b_grads.append(layer.bias.grad.detach().cpu().numpy())
        else:
            b_grads.append(None)
        w_grads.append(g)
    return float(loss.item()), w_grads, b_grads


def load_mnist_cnn_inputs(num_inputs, batch_size, data_dir, input_shape):
    downsample = input_shape[1] == 14 and input_shape[2] == 14
    (train_loader, _), _, _, _ = mnist_loader_manual(
        batch_size=batch_size,
        shuffle=False,
        preprocess=False,
        CNN_preprocess=False,
        downsample=downsample,
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
    expected_flat = int(np.prod(input_shape))
    if x.shape[1] != expected_flat:
        raise ValueError(f"Input shape mismatch: expected {expected_flat} features, got {x.shape[1]}")
    x = x.reshape(num_inputs, *input_shape)
    return x, y


def load_inputs(num_inputs, batch_size, data_dir, dataset, input_shape, num_classes, seed, input_source):
    if input_source == "synthetic":
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(num_inputs, *input_shape)).astype(np.float32)
        y = rng.integers(low=0, high=num_classes, size=(num_inputs,), dtype=np.int64)
        return x, y

    if dataset != "mnist":
        raise ValueError("Only MNIST is supported for non-synthetic input in this checker.")
    if input_shape[0] != 1:
        raise ValueError(f"MNIST expects input channels=1, got {input_shape[0]}")
    return load_mnist_cnn_inputs(num_inputs, batch_size, data_dir, input_shape)


def build_params(cfg, layer_sizes, seed, batch_size):
    restrict = cfg["restrict"]
    if isinstance(restrict, (int, float)):
        restrict = tuple(float(restrict) for _ in range(len(layer_sizes)))
    else:
        restrict = tuple(float(v) for v in restrict)
        if len(restrict) < len(layer_sizes):
            restrict = restrict + (restrict[-1],) * (len(layer_sizes) - len(restrict))
        if len(restrict) > len(layer_sizes):
            restrict = restrict[: len(layer_sizes)]

    sparsity_impact = tuple(0.0 for _ in range(len(layer_sizes)))
    return Params(
        dataset=cfg["dataset"],
        random_seed=seed,
        layer_sizes=layer_sizes,
        init_thresholds=float(cfg["init_thresholds"]),
        num_epochs=1,
        learning_rate=float(cfg["learning_rate"]),
        batch_size=batch_size,
        load_file=False,
        shuffle_activations=False,
        restrict=restrict,
        firing_nb=cfg["firing_nb"],
        sync_rate=cfg["sync_rate"],
        max_nonzero=0,
        shuffle_input=False,
        threshold_lr=0.0,
        sparsity_impact=sparsity_impact,
        w_reg=0.0,
        rerun="",
        top_weights=-1,
        history_size=0,
        use_bias=True,
    )


def layer_names(layer_defs):
    names = []
    conv_count = 0
    fc_count = 0
    for i, layer_def in enumerate(layer_defs):
        if layer_def["kind"] == "conv":
            conv_count += 1
            names.append(f"conv_{conv_count}")
        else:
            fc_count += 1
            if i == len(layer_defs) - 1:
                names.append("output")
            else:
                names.append(f"fc_{fc_count}")
    return names


def main():
    parser = argparse.ArgumentParser(description="Compare async_CNN-style gradients to PyTorch gradients")
    parser.add_argument("--config", type=str, default="configs/CNN_config.yaml")
    parser.add_argument("--num-inputs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--input-source", type=str, default="mnist", choices=["mnist", "synthetic"])
    parser.add_argument(
        "--fc-prop-mode",
        type=str,
        default="weight_res",
        choices=["weight_res", "no_mask", "relu_only"],
    )
    parser.add_argument(
        "--conv-grad-reduction",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="async_CNN uses mean for conv hidden layers after combine_batch_avg.",
    )
    parser.add_argument(
        "--fc-mask-value-mode",
        type=str,
        default="input_dim",
        choices=["input_dim", "sync_rate"],
        help="Value assigned to output_vector on active FC units for MLP_back_prop masking.",
    )
    parser.add_argument(
        "--pool-backprop-mode",
        type=str,
        default="exact",
        choices=["exact", "repeat"],
        help="Pooling backward mode for conv layers. 'repeat' matches current async_CNN approximation.",
    )
    args = parser.parse_args()

    cfg = load_config_with_defaults(args.config, is_cnn=True)
    # Keep parity with your MLP checker setup during gradient validation.
    cfg["firing_nb"] = 10000
    cfg["sync_rate"] = 784
    layer_sizes = tuple(cfg["layer_sizes"])
    if len(layer_sizes) < 3:
        raise ValueError(f"Expected at least input/hidden/output in layer_sizes, got {layer_sizes}")
    if len(layer_sizes[0]) != 3:
        raise ValueError(f"CNN checker expects input layer as [C,H,W], got {layer_sizes[0]}")
    if len(layer_sizes[-1]) != 1:
        raise ValueError(f"CNN checker expects output layer as [num_classes], got {layer_sizes[-1]}")

    num_classes = int(layer_sizes[-1][0])
    input_shape = tuple(int(v) for v in layer_sizes[0])
    layer_weights, layer_biases, layer_defs = init_weights_from_async_cnn(layer_sizes, args.seed)
    params = build_params(cfg, layer_sizes, args.seed, args.num_inputs)

    x_np, y_np = load_inputs(
        num_inputs=args.num_inputs,
        batch_size=max(64, args.num_inputs),
        data_dir=args.data_dir,
        dataset=cfg["dataset"],
        input_shape=input_shape,
        num_classes=num_classes,
        seed=args.seed,
        input_source=args.input_source,
    )
    x = jnp.asarray(x_np, dtype=jnp.float32)

    torch_loss, torch_w_grads, torch_b_grads = pytorch_grads(
        x_np, y_np, [np.asarray(w) for w in layer_weights],
        [np.asarray(b) if b is not None else None for b in layer_biases],
        layer_defs,
    )
    custom_loss, custom_w_grads, custom_b_grads = custom_async_cnn_grads(
        x,
        y_np,
        layer_weights,
        layer_biases,
        layer_defs,
        params,
        num_classes=num_classes,
        fc_prop_mode=args.fc_prop_mode,
        conv_grad_reduction=args.conv_grad_reduction,
        fc_mask_value_mode=args.fc_mask_value_mode,
        pool_backprop_mode=args.pool_backprop_mode,
    )

    print(f"tested_inputs={args.num_inputs}")
    print(f"input_source={args.input_source}")
    print(f"custom_loss={custom_loss:.8f}, torch_loss={torch_loss:.8f}")
    print(
        f"fc_prop_mode={args.fc_prop_mode}, conv_grad_reduction={args.conv_grad_reduction}, "
        f"fc_mask_value_mode={args.fc_mask_value_mode}, pool_backprop_mode={args.pool_backprop_mode}"
    )
    print(f"firing_nb={params.firing_nb}, sync_rate={params.sync_rate}")

    names = layer_names(layer_defs)
    cosines = []
    print("\nPer-layer weight gradient similarity (custom vs PyTorch):")
    for name, g_custom, g_torch in zip(names, custom_w_grads, torch_w_grads):
        g_custom_np = np.asarray(g_custom)
        cosine, rel_l2, rel_l2_scaled, scale, mae, max_abs = similarity_metrics(g_custom_np, g_torch)
        cosines.append(cosine)
        print(
            f"{name}: cosine={cosine:.8f}, rel_l2={rel_l2:.8e}, "
            f"rel_l2_scaled={rel_l2_scaled:.8e}, scale={scale:.8e}, "
            f"mae={mae:.8e}, max_abs={max_abs:.8e}"
        )

    print("\nPer-layer bias gradient similarity (custom vs PyTorch, FC layers only):")
    for name, g_custom, g_torch in zip(names, custom_b_grads, torch_b_grads):
        if g_custom is None:
            continue
        g_custom_np = np.asarray(g_custom)
        cosine, rel_l2, rel_l2_scaled, scale, mae, max_abs = similarity_metrics(g_custom_np, g_torch)
        cosines.append(cosine)
        print(
            f"bias_{name}: cosine={cosine:.8f}, rel_l2={rel_l2:.8e}, "
            f"rel_l2_scaled={rel_l2_scaled:.8e}, scale={scale:.8e}, "
            f"mae={mae:.8e}, max_abs={max_abs:.8e}"
        )

    mean_cos = float(np.mean(cosines)) if cosines else float("nan")
    min_cos = float(np.min(cosines)) if cosines else float("nan")
    print("\nOverall similarity:")
    print(f"mean_cosine={mean_cos:.8f}, min_cosine={min_cos:.8f}")


if __name__ == "__main__":
    main()
