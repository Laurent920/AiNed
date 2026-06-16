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
    # Use a fired/not-fired mask instead of the timing-based (input_time <= output_time) mask.
    # The timing mask biased gradient toward late-firing (monopolizing) neurons because a higher
    # last-fire-time includes more inputs, amplifying their gradient over epochs until they dominate
    # and all other neurons die. The activity mask gives equal weight to all fired neurons.
    weight_res = jnp.broadcast_to((output_vector > 0)[None, :], (len(input_vector), len(output_vector)))

    # (2) Shape: (784, 128)
    # if next_weight_res is not None:
    #     weight_res = weight_res * (~jnp.all(next_weight_res == 0, axis=1))[None, :]

    # (3) Shape: (784, 128)
    reset = params.restrict
    if not isinstance(reset, int) and not isinstance(reset, float):
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
    mean_weight_grad = jnp.sum(weight_grad, axis=0) # (784, 128)
    mean_weight_grad = jnp.expand_dims(mean_weight_grad, axis=0)  # Shape: (1, 784, 128)
    layer_activity = jnp.where(all_neuron_states.layer_activity > 0, 1, 0)
    th_grad = -jnp.mean(next_grad * layer_activity, axis=0)  # Shape: (128)
    if params.init_thresholds != 0:
        thresholds = all_neuron_states.thresholds[0]
        th_grad = th_grad * thresholds * (1 - thresholds)
    # jax.debug.print("{} {} {}", layer_activity.shape, thresholds, th_grad.shape)
    neuron_fired = (all_neuron_states.output_vector > 0).astype(next_grad.dtype)  # (B, out_dim)
    # bias is added once per input event in the forward pass, so scale by total events per sample
    n_events = jnp.sum(all_neuron_states.input_activity, axis=1, keepdims=True)  # (B, 1)
    bias_grad = jnp.sum(next_grad * neuron_fired * n_events/params.max_nonzero, axis=0)

    return mean_weight_grad, th_grad, weight_res, bias_grad

# region RNN back_prop
@partial(jax.jit, static_argnames=['params', 'layer_idx'])
def compute_full_RNN_bpp(params, all_neuron_states, next_grad, layer_idx):
    '''
    Computes the correct RNN backprop for a single element in the batch.

    W_Ih gradient:
        dLoss/dW_Ih = dLoss/dy^T * W_hy * sum_i [ ReLU'(z^i) * sum_k [ prod_{j=k}^{i-1} A_j * x_k ] ]
        where A_j = W_hh if z^j > 0, else 1
        
        Accumulated in forward pass via:
        - rnn_running_sum (n_input,): running sum per input neuron, updated on each input event
            - on input event k with value x_k: running_sum = running_sum * A_current + x_k
              (A_current = W_hh if z > 0, else 1; but at input time we use the *previous* A since z is
               recomputed after the input, so we multiply by A from the previous firing state)
            - on firing (z^i > 0): total_sum += running_sum
        - rnn_total_sum (n_input,): total accumulated residual for W_Ih

    W_hh gradient (compact matrix trace used in async_RNN forward bookkeeping):
        U_i = source(R_{i-1}) + U_{i-1} @ A_{i-1}                              (shape: n_hidden x n_hidden)
        source(R) has identical columns: source[:, n] = R for all n.
        S   = sum_i [ U_i * ReLU'(z_i)[None, :] ]                              (shape: n_hidden x n_hidden)
        A_j = I + diag(ReLU'(z_j)) @ (W_hh - I)
        R_k = ReLU(z_k)

        Forward pass stores:
        - U_i in `rnn_running_product`
        - S in `rnn_total_product_sum`

    The final gradient is then:
        weight_grad_Ih[input_k, output_j] = next_grad[j] * rnn_total_sum[k]   (shape: n_input x n_hidden)
        weight_grad_hh[m, n] = next_grad[n] * rnn_total_product_sum[m, n]
    '''
    # --- W_Ih gradient using rnn_total_sum ---
    # rnn_total_sum shape: (n_input, n_hidden) - accumulated per (input_neuron, hidden_neuron) pair
    # OR shape: (n_input,) if shared across hidden neurons (depends on your accumulation strategy)
    # Here we assume rnn_total_sum has shape (n_input, n_hidden)

    rnn_total_sum = all_neuron_states.rnn_total_sum  # Shape: (n_input, n_hidden)

    # next_grad shape: (n_hidden,)
    # weight_grad_Ih[i, j] = rnn_total_sum[i, j] * next_grad[j]
    weight_grad_Ih = rnn_total_sum * next_grad[None, :]  # Shape: (n_input, n_hidden)

    # Weight residual for W_Ih: nonzero wherever rnn_total_sum is nonzero
    weight_res_Ih = jnp.where(rnn_total_sum != 0, 1, 0)  # Shape: (n_input, n_hidden)

    # --- W_hh gradient ---
    # Check if exact RTRL traces are available (H, H, H)
    exact_hh_total = all_neuron_states.exact_hh_total
    if exact_hh_total is not None:
        # Exact RTRL: grad_hh[m,n] = sum_j next_grad[j] * exact_hh_total[m,n,j]
        weight_grad_hh = jnp.einsum("j,mnj->mn", next_grad, exact_hh_total)
    else:
        # Compact trace approximation
        rnn_total_product_sum = all_neuron_states.rnn_total_product_sum
        if rnn_total_product_sum.ndim == 1:
            weight_grad_hh = jnp.outer(rnn_total_product_sum, next_grad)
        else:
            weight_grad_hh = rnn_total_product_sum * next_grad[None, :]

    # --- bias gradient ---
    exact_bias_total = all_neuron_states.exact_bias_total
    if exact_bias_total is not None:
        # Exact RTRL: grad_bias[n] = sum_j next_grad[j] * exact_bias_total[n,j]
        grad_bias = jnp.einsum("j,nj->n", next_grad, exact_bias_total)
    else:
        # Diagonal approximation
        bias_total_sum = all_neuron_states.bias_total_sum  # Shape: (n_hidden,)
        grad_bias = bias_total_sum * next_grad  # Shape: (n_hidden,)

    return weight_grad_Ih, weight_res_Ih, weight_grad_hh, grad_bias

@partial(jax.jit, static_argnames=['params', 'layer_idx'])
def RNN_back_prop(params, all_neuron_states, next_grad, layer_idx):
    weight_grad_Ih, weight_res_Ih, weight_grad_hh, grad_bias = jax.vmap(
        compute_full_RNN_bpp, in_axes=(None, 0, 0, None)
    )(params, all_neuron_states, next_grad, layer_idx)
    # weight_grad_Ih: (B, n_input, n_hidden)
    # weight_grad_hh: (B, n_hidden, n_hidden)
    # grad_bias: (B, n_hidden)

    # Keep parity with loss scaling used in the MLP/CNN paths:
    # loss_grad already carries batch normalization from loss_func.
    # Sum here avoids an additional unintended 1/B scaling.
    mean_weight_grad_Ih = jnp.sum(weight_grad_Ih, axis=0)  # (n_input, n_hidden)
    mean_weight_grad_hh = jnp.sum(weight_grad_hh, axis=0)  # (n_hidden, n_hidden)
    mean_bias_grad = jnp.sum(grad_bias, axis=0)  # (n_hidden,)

    # jax.debug.print("mean_weight_grad_Ih stats: min={}, max={}, mean={} shape: {}",
    #                 jnp.min(mean_weight_grad_Ih), jnp.max(mean_weight_grad_Ih),
    #                 jnp.mean(mean_weight_grad_Ih), mean_weight_grad_Ih.shape)
    # jax.debug.print("mean_weight_grad_hh stats: min={}, max={}, mean={} shape: {}",
    #                 jnp.min(mean_weight_grad_hh), jnp.max(mean_weight_grad_hh),
    #                 jnp.mean(mean_weight_grad_hh), mean_weight_grad_hh.shape)

    return mean_weight_grad_Ih, mean_weight_grad_hh, weight_res_Ih, mean_bias_grad
