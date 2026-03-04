import jax
import jax.numpy as jnp

@jax.jit
def loss_func(logits, labels):
    # Numerically stable cross-entropy: matches PyTorch's CrossEntropyLoss exactly.
    # Uses log-sum-exp via jax.nn.log_softmax, no epsilon clamping.
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # (B, C)
    losses = -jnp.sum(labels * log_probs, axis=-1)    # (B,)
    return jnp.mean(losses)

@jax.jit
def loss_bpp(weights, all_neuron_states, loss_grad):
    '''
    For each batch element:
    Compute the gradient of output layer and the gradient w.r.t the weights of the output layer
    Shapes are given for an output layer of shape (128, 10) 

    (1) Compute the gradient w.r.t the output of the layer:
        out_grad = weights @ loss_grad
    (2) Compute the gradient w.r.t the weights of the layer:
        weight_grad = loss_grad * input_residuals

    Return:
        out_grad, shape: (128,)
        weight_grad, shape: (128, 10)
    '''
    out_grad = jnp.dot(weights, loss_grad) # Shape: (128,)
    # out_grad *= all_neuron_states.input_activity
    # jax.debug.print("{}, mean outgrad {}", all_neuron_states.input_activity, jnp.mean(out_grad))
    # jax.debug.print("out grad shape: {}, layer activity shape: {}", out_grad.shape, all_neuron_states.input_activity.shape)

    loss_grad_expanded = jnp.expand_dims(loss_grad, axis=1)  # Shape: (10, 1)
    all_input_residuals = all_neuron_states.input_residuals # Shape: (128,)

    weight_grad = loss_grad_expanded * all_input_residuals  # Shape: (10, 128)
    
    return out_grad, weight_grad.T