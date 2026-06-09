import jax
import jax.numpy as jnp

from dataset_helpers.cnn_pytorch import get_weights_for_rank


def init_params(key, layers, params, layer_idx, filename="", best=False):
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

            # bound = jnp.sqrt(6.0 / (fan_in+fan_out))
            # return jax.random.uniform(keys[layer_idx], weights_shape, jnp.float32, -bound, bound)
            # std = jnp.sqrt(2.0 / fan_in)
            std = 1e-2
            weights_linear_layer = std * jax.random.normal(keys[layer_idx], weights_shape)
            return weights_linear_layer

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
