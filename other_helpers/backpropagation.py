import jax
import jax.numpy as jnp

from functools import partial


# region MLP back_prop
@partial(jax.jit, static_argnames=['params', 'layer_idx'])
def compute_full_bpp(params, all_neuron_states, next_grad, layer_idx):
    '''
    Computes the full bpp for a single element in the batch. 
    Shapes are given as reference for a network (784, 128, 64, 10) and for the first hidden layer with weights (784, 128).

    input_vector: contains for each input neuron the last iteration for which an event was received, input shape: (784,)
    output_vector: contains for each output neuron the last iteration for which it activated, output shape: (128,)
    layer_activity: contains for each neuron the number of times it activated, layer shape: (128,)

    next_res: contains the weight residuals of the next layer (64, 10)
    next_grad: contains the gradient from the next layer == gradient w.r.t. the output (O^t in mathematical derivation) (64,)

    (1) Compute weight_res:
        Computes the weight residual that contains 1 if the correponding weight has received an input and the correponding output has fired either directly or later after integrating the input
    (2) Recompute weight_res with next_res:
        Recompute the weight residuals of the current layer by taking into account the weight residuals of the next layer.
        Basically if one row (neuron) in the next layer's weights residuals is all zeros (=neuron never activated), then the corresponding column in the current layer should be set to zero. 
    (3) Apply restrict to the weight_res:
        Apply [1-(1-alpha)^n]/alpha, the result of the finite geometric series where n is the number of times a neuron activated in the layer and alpha is the restrict parameter
    (4) Compute the partial gradient w.r.t the weights by integrating the next layer's gradient:
        z_grad = weights_residuals * next_grad
    (5) Compute the full gradient w.r.t the weights by multiplying with the input residuals:
        weight_grad = input_residuals * z_grad

    Return:
        weight_grad, shape: (784, 128)
    '''
    input_vector = all_neuron_states.input_vector
    output_vector = all_neuron_states.output_vector
    layer_activity = all_neuron_states.layer_activity

    # (1) Shape: (784, 128)
    weight_res = (input_vector[:, None] <= output_vector[None, :])

    # (2) Shape: (784, 128)
    # if next_weight_res is not None:
    #     weight_res = weight_res * (~jnp.all(next_weight_res == 0, axis=1))[None, :]

    # (3) Shape: (784, 128)
    reset = params.restrict
    if not isinstance(reset, int):
        reset = reset[layer_idx]
    new_layer_activity = jnp.where(reset > 0, (1-jnp.power((1-reset), layer_activity+1))/reset, 1) # Shape (128,)
    mul_res = jnp.broadcast_to(new_layer_activity, weight_res.shape) 
    weight_res = weight_res * mul_res

    # (4) Shape: (784, 128)
    next_grad_expanded = jnp.expand_dims(next_grad, axis=0)  # Shape: (1, 128)
    # jax.debug.print("shapes {} {}", weight_res.shape, next_grad_expanded.shape)
    z_grad = weight_res * next_grad_expanded

    # (5) Shape: (784, 128)
    x = all_neuron_states.input_residuals # Shape (784,)
    x_reshaped = x[..., jnp.newaxis]      # Shape becomes (784, 1)

    # Debug prints
    # jax.debug.print("new_layer_activity has NaN: {} shape: {}", jnp.any(jnp.isnan(new_layer_activity)), output_vector.shape)
    # jax.debug.print("new_layer_activity stats: min={}, max={}, mean={} shape: {}", 
    #                 jnp.min(new_layer_activity), jnp.max(new_layer_activity), jnp.mean(new_layer_activity), output_vector.shape)
    # jax.debug.print("layer_activity has NaN: {} shape: {}", jnp.any(jnp.isnan(layer_activity)), output_vector.shape)
    # jax.debug.print("layer_activity stats: min={}, max={}, mean={} shape: {}", 
    #                 jnp.min(layer_activity), jnp.max(layer_activity), jnp.mean(layer_activity), output_vector.shape)
    # jax.debug.print("x_reshaped has NaN: {} shape: {}", jnp.any(jnp.isnan(x_reshaped)), output_vector.shape)
    # jax.debug.print("z_grad has NaN: {} shape: {}", jnp.any(jnp.isnan(z_grad)), output_vector.shape)
    # jax.debug.print("x_reshaped stats: min={}, max={}, mean={} shape: {}", 
    #                 jnp.min(x_reshaped), jnp.max(x_reshaped), jnp.mean(x_reshaped), output_vector.shape)
    # jax.debug.print("z_grad stats: min={}, max={}, mean={} shape: {}", 
    #                 jnp.min(z_grad), jnp.max(z_grad), jnp.mean(z_grad), output_vector.shape)
    weight_grad = x_reshaped * z_grad # (784, 128)

    return weight_grad, weight_res

@partial(jax.jit, static_argnames=['params', 'layer_idx'])
def MLP_back_prop(params, all_neuron_states, next_grad, layer_idx):
    weight_grad, weight_res  = jax.vmap(compute_full_bpp, in_axes=(None, 0, 0, None))(params, all_neuron_states, next_grad, layer_idx) # Shape: (B, 784, 128)
    mean_weight_grad = jnp.mean(weight_grad, axis=0) # (784, 128)
    mean_weight_grad = jnp.expand_dims(mean_weight_grad, axis=0)  # Shape: (1, 784, 128)

    layer_activity = jnp.where(all_neuron_states.layer_activity > 0, 1, 0)
    th_grad = -jnp.mean(next_grad * layer_activity, axis=0)  # Shape: (128)
    thresholds = all_neuron_states.thresholds[0] # The whole batch has the same thresholds
    th_grad = th_grad * thresholds * (thresholds - 1)
    # jax.debug.print("{} {} {}", layer_activity.shape, thresholds, th_grad.shape)

    return mean_weight_grad, th_grad, weight_res

# region RNN back_prop
@partial(jax.jit, static_argnames=['params', 'layer_idx'])
def compute_full_RNN_bpp(params, all_neuron_states, next_grad, layer_idx):
    '''
    Computes the full bpp for a single element in the batch. 
    Shapes are given as reference for a network (784, 128, 64, 10) and for the first hidden layer with weights (784, 128).

    input_vector: contains for each input neuron the last iteration for which an event was received, input shape: (784,)

    next_res: contains the weight residuals of the next layer (64, 10)
    next_grad: contains the gradient from the next layer == gradient w.r.t. the output (O^t in mathematical derivation) (64,)

    (4) Compute the partial gradient w.r.t the weights by integrating the next layer's gradient:
        z_grad = weights_residuals * next_grad
    (5) Compute the full gradient w.r.t the weights by multiplying with the input residuals:
        weight_grad = input_residuals * z_grad

    Return:
        weight_grad, shape: (784, 128)
    '''
    weight_res = jnp.where(all_neuron_states.output_residuals == 0, 0, 1) # Shape: (128)

    # (4) Shape: (784, 128)
    next_grad_expanded = jnp.expand_dims(next_grad, axis=0)  # Shape: (1, 128)
    # jax.debug.print("shapes {} {}", weight_res.shape, next_grad_expanded.shape)
    z_grad = weight_res * next_grad_expanded

    # (5) Shape: (784, 128)
    x = all_neuron_states.output_residuals # Shape (128,)
    x_reshaped = x[..., jnp.newaxis]      # Shape becomes (128, 1)

    # Debug prints
    # jax.debug.print("x_reshaped stats: min={}, max={}, mean={} shape: {}", 
    #                 jnp.min(x_reshaped), jnp.max(x_reshaped), jnp.mean(x_reshaped), output_vector.shape)
    # jax.debug.print("z_grad stats: min={}, max={}, mean={} shape: {}", 
    #                 jnp.min(z_grad), jnp.max(z_grad), jnp.mean(z_grad), output_vector.shape)
    weight_grad = x_reshaped * z_grad # (128, 128)

    return weight_grad, weight_res

@partial(jax.jit, static_argnames=['params', 'layer_idx'])
def RNN_back_prop(params, all_neuron_states, next_grad, layer_idx):
    weight_grad, weight_res  = jax.vmap(compute_full_RNN_bpp, in_axes=(None, 0, 0, None))(params, all_neuron_states, next_grad, layer_idx) # Shape: (B, 784, 128)

    mean_weight_grad = jnp.mean(weight_grad, axis=0) # (784, 128)
    # mean_weight_grad = jnp.expand_dims(mean_weight_grad, axis=0)  # Shape: (1, 784, 128)

    # jax.debug.print("mean_weight_grad stats: min={}, max={}, mean={} shape: {}", 
    #                 jnp.min(mean_weight_grad), jnp.max(mean_weight_grad), jnp.mean(mean_weight_grad), mean_weight_grad.shape)
    
    # jax.debug.print("mean_weight_grad stats: mean shape {} total shape: {}", mean_weight_grad.shape, weight_grad.shape)
    return mean_weight_grad, weight_res