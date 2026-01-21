import jax
import jax.numpy as jnp

@jax.jit
def softmax_cross_entropy_with_logits(logits, labels):
    # Compute the softmax in a numerically stable way
    logits_max = jnp.max(logits, axis=0, keepdims=True)
    exps = jnp.exp(logits - logits_max)
    softmax = exps / (jnp.sum(exps, axis=0, keepdims=True) + 1e-8)
    # Compute the cross-entropy loss
    cross_entropy = -jnp.sum(labels * jnp.log(softmax + 1e-8), axis=0)
    return cross_entropy

@jax.jit
def mean_loss(logits, labels):
    batched_softmax_cross_entropy = jax.vmap(softmax_cross_entropy_with_logits, in_axes=(0, 0))
    losses = batched_softmax_cross_entropy(logits, labels)
    # jax.debug.print("Losses per batch element: {}", jnp.mean(losses))
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