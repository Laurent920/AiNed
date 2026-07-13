import os
import jax
import jax.numpy as jnp

from dataset_helpers.cnn_pytorch import get_weights_for_rank

# Opt-in kaiming_normal(mode='fan_out') init, matching the PyTorch twin. Default OFF (legacy init).
_KAIMING_FANOUT = os.environ.get("AINED_KAIMING_INIT", "") == "1"
# Opt-in AED variance-preserving init (WEIGHT_INIT_DERIVATION.md): for an accumulate-then-fire
# layer the fixed point is  sigma_w = sqrt( firing_nb / (gamma * fan_out) ),  gamma ~ 2*ln(fan_out/firing_nb).
_VARINIT = os.environ.get("AINED_VARINIT", "") == "1"
# FC-only variant: apply the variance-preserving formula to Linear layers (where it needs no `p`
# and legacy's hard-coded std=1e-2 makes logits vanish), but keep the legacy conv init, which
# measures flat in the async forward. The full _VARINIT conv branch over-scales (re-firing makes
# the true integration depth much larger than the inferred fire-once estimate).
_VARINIT_FC = os.environ.get("AINED_VARINIT_FC", "") == "1"


def _aed_sigma(fan_out, firing_nb):
    """Variance-preserving std for an accumulate-then-fire layer (WEIGHT_INIT_DERIVATION.md)."""
    k = firing_nb if isinstance(firing_nb, (int, float)) else firing_nb
    k = max(1, int(k))
    k = min(k, fan_out)                      # formula assumes sparse firing (k <= fan_out)
    gamma = max(1.0, 2.0 * float(jnp.log(max(fan_out / k, 1.0 + 1e-6))))
    return jnp.sqrt(k / (gamma * fan_out))


def init_params(key, layers, params, layer_idx, filename="", best=False, flat_layer_sizes=None):
    keys = jax.random.split(key, len(layers))
    load_file = params.load_file

    if layer_idx != 0:
        if load_file:
            folder = f"tensor_data/CNN/{params.dataset}/"
            f = "tensor_data"+filename+".npz"
            return get_weights_for_rank(folder+f, layer_idx)

        layer = layers[layer_idx]
        weights_shape = layer.weights_shape

        if layer.is_conv:
            out_ch, in_ch, kh, kw = weights_shape
            fan_in = in_ch * kh * kw
        else:
            fan_in = weights_shape[0]
            fan_out = weights_shape[1]

            fnb_l = params.firing_nb if isinstance(params.firing_nb, int) else params.firing_nb[layer_idx]
            if _VARINIT or _VARINIT_FC:
                std = _aed_sigma(fan_out, fnb_l)   # sqrt(firing_nb / (gamma * fan_out))
                return std * jax.random.normal(keys[layer_idx], weights_shape)

            if _KAIMING_FANOUT:
                std = jnp.sqrt(2.0 / fan_out)   # kaiming_normal(mode='fan_out'), matches PyTorch twin
                return std * jax.random.normal(keys[layer_idx], weights_shape)

            # bound = jnp.sqrt(6.0 / (fan_in+fan_out))
            # return jax.random.uniform(keys[layer_idx], weights_shape, jnp.float32, -bound, bound)
            # std = jnp.sqrt(2.0 / fan_in)
            std = 1e-2
            weights_linear_layer = std * jax.random.normal(keys[layer_idx], weights_shape)
            return weights_linear_layer

        fnb_l = params.firing_nb if isinstance(params.firing_nb, int) else params.firing_nb[layer_idx]
        if _VARINIT:
            # Conv accumulates over the ACTIVE receptive field, NOT the event spread (out_ch*k*k).
            # Effective integration depth  m = p*(in_ch*k_h*k_w) = E_active*k_h*k_w/(H_in*W_in),
            # where the active-input fraction p = E_active/(in_ch*H_in*W_in) is INFERRED from the
            # input sparsity (E_active, ~conserved for firing_nb=1) and the layer's input dims.
            _E = int(os.environ.get("AINED_ACTIVE_EVENTS", params.max_nonzero))
            # NOTE: params.flat_layer_sizes is still () at init time (set after init_weights),
            # so the caller passes the Network's built flat_layer_sizes.
            _fls = flat_layer_sizes if flat_layer_sizes else params.flat_layer_sizes
            _prev = _fls[layer_idx - 1]
            H_in, W_in = _prev[1], _prev[2]
            eff = max(1.0, _E * kh * kw / (H_in * W_in))
            std = _aed_sigma(eff, fnb_l)   # sqrt(firing_nb / (gamma * eff)),  gamma ~ 2*ln(eff/firing_nb)
            return std * jax.random.normal(keys[layer_idx], weights_shape)

        if _KAIMING_FANOUT:
            fan_out = out_ch * kh * kw       # kaiming_normal(mode='fan_out') for conv, matches twin
            std = jnp.sqrt(2.0 / fan_out)
            return std * jax.random.normal(keys[layer_idx], weights_shape)

        # Kaiming He Uniform initialization
        bound = jnp.sqrt(6.0 / fan_in)
        weights_conv_layer = jax.random.uniform(keys[layer_idx], weights_shape, jnp.float32, -bound, bound)

        # Kaiming He Normal initialization
        std = jnp.sqrt(2.0 / fan_in)
        weights_conv_layer = std * jax.random.normal(keys[layer_idx], weights_shape)

        bound = jnp.sqrt(2.0 / fan_in)
        weights_conv_layer = jax.random.uniform(keys[layer_idx], weights_shape, jnp.float32, -bound, bound)
        return weights_conv_layer
    else:
        return jnp.zeros((1, 1, 1, 1))
