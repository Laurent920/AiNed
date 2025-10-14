import jax
import jax.numpy as jnp
from jax import grad, vjp, custom_vjp
from mpi4py import MPI
import mpi4jax

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Layer configuration
INPUT_DIM = 784
HIDDEN_DIM = 128
OUTPUT_DIM = 10

# Tags for MPI communication
TAG_FORWARD_INPUT_TO_HIDDEN = 0
TAG_FORWARD_HIDDEN_TO_OUTPUT = 1
TAG_BACKWARD_OUTPUT_TO_HIDDEN = 2
TAG_BACKWARD_HIDDEN_TO_INPUT = 3


# ============ INPUT LAYER (Rank 0) ============
@custom_vjp
def input_layer_forward(x):
    """Forward pass: send input neuron-by-neuron to hidden layer"""
    if rank == 0:
        # Send each neuron as [index, value]
        for i in range(INPUT_DIM):
            neuron_packet = jnp.array([float(i), x[i]])
            mpi4jax.send(neuron_packet, dest=1, tag=TAG_FORWARD_INPUT_TO_HIDDEN, comm=comm)
    return x

def input_layer_forward_fwd(x):
    """Forward pass with saved values for backward"""
    out = input_layer_forward(x)
    return out, (x,)  # Save input for gradient accumulation

def input_layer_forward_bwd(residuals, g):
    """Backward pass: receive gradient neuron-by-neuron from hidden layer"""
    if rank == 0:
        x, = residuals
        # Initialize gradient accumulator
        grad_input = jnp.zeros_like(x)
        
        # Receive gradient for each input neuron
        for i in range(INPUT_DIM):
            grad_packet = mpi4jax.recv(
                jnp.zeros(2),  # [index, gradient_value]
                source=1,
                tag=TAG_BACKWARD_HIDDEN_TO_INPUT,
                comm=comm
            )
            neuron_idx = int(grad_packet[0])
            grad_value = grad_packet[1]
            grad_input = grad_input.at[neuron_idx].set(grad_value)
        
        return (grad_input,)
    else:
        return (g,)

input_layer_forward.defvjp(input_layer_forward_fwd, input_layer_forward_bwd)


# ============ HIDDEN LAYER (Rank 1) ============
@custom_vjp
def hidden_layer_forward(W, placeholder_input):
    """Forward pass: receive input neurons, compute partial outputs, send neuron-by-neuron"""
    if rank == 1:
        # Initialize output accumulator
        hidden_out = jnp.zeros(HIDDEN_DIM)
        
        # Receive and process each input neuron
        for _ in range(INPUT_DIM):
            neuron_packet = mpi4jax.recv(
                jnp.zeros(2),  # [index, value]
                source=0,
                tag=TAG_FORWARD_INPUT_TO_HIDDEN,
                comm=comm
            )
            input_idx = int(neuron_packet[0])
            input_value = neuron_packet[1]
            
            # Compute contribution: W[:, input_idx] * input_value
            hidden_out = hidden_out + W[:, input_idx] * input_value
        
        # Send each hidden neuron to output layer
        for h_idx in range(HIDDEN_DIM):
            hidden_packet = jnp.array([float(h_idx), hidden_out[h_idx]])
            mpi4jax.send(hidden_packet, dest=2, tag=TAG_FORWARD_HIDDEN_TO_OUTPUT, comm=comm)
        
        return hidden_out
    return jnp.zeros(HIDDEN_DIM)

def hidden_layer_forward_fwd(W, placeholder_input):
    if rank == 1:
        hidden_out = jnp.zeros(HIDDEN_DIM)
        input_values = jnp.zeros(INPUT_DIM)
        
        # Receive and process each input neuron
        for _ in range(INPUT_DIM):
            neuron_packet = mpi4jax.recv(
                jnp.zeros(2),
                source=0,
                tag=TAG_FORWARD_INPUT_TO_HIDDEN,
                comm=comm
            )
            input_idx = int(neuron_packet[0])
            input_value = neuron_packet[1]
            
            # Save input values for backward pass
            input_values = input_values.at[input_idx].set(input_value)
            
            # Compute contribution
            hidden_out = hidden_out + W[:, input_idx] * input_value
        
        # Send each hidden neuron to output layer
        for h_idx in range(HIDDEN_DIM):
            hidden_packet = jnp.array([float(h_idx), hidden_out[h_idx]])
            mpi4jax.send(hidden_packet, dest=2, tag=TAG_FORWARD_HIDDEN_TO_OUTPUT, comm=comm)
        
        return hidden_out, (W, input_values)  # Save W and input
    return jnp.zeros(HIDDEN_DIM), (W, placeholder_input)

def hidden_layer_forward_bwd(residuals, g):
    if rank == 1:
        W, input_values = residuals
        
        # Initialize gradient accumulators
        grad_W = jnp.zeros_like(W)
        grad_input = jnp.zeros(INPUT_DIM)
        
        # Receive gradient for each hidden neuron from output layer
        grad_hidden = jnp.zeros(HIDDEN_DIM)
        for _ in range(HIDDEN_DIM):
            grad_packet = mpi4jax.recv(
                jnp.zeros(2),  # [index, gradient_value]
                source=2,
                tag=TAG_BACKWARD_OUTPUT_TO_HIDDEN,
                comm=comm
            )
            hidden_idx = int(grad_packet[0])
            grad_value = grad_packet[1]
            grad_hidden = grad_hidden.at[hidden_idx].set(grad_value)
        
        # Compute gradient w.r.t. W
        # grad_W[i,j] = grad_hidden[i] * input_values[j]
        for i in range(HIDDEN_DIM):
            for j in range(INPUT_DIM):
                grad_W = grad_W.at[i, j].set(grad_hidden[i] * input_values[j])
        
        # Compute gradient w.r.t. input
        # grad_input[j] = sum_i(W[i,j] * grad_hidden[i])
        for j in range(INPUT_DIM):
            grad_input = grad_input.at[j].set(jnp.sum(W[:, j] * grad_hidden))
        
        # Send gradient for each input neuron back to input layer
        for j in range(INPUT_DIM):
            grad_packet = jnp.array([float(j), grad_input[j]])
            mpi4jax.send(grad_packet, dest=0, tag=TAG_BACKWARD_HIDDEN_TO_INPUT, comm=comm)
        
        return (grad_W, jnp.zeros_like(input_values))
    return (jnp.zeros((HIDDEN_DIM, INPUT_DIM)), jnp.zeros(INPUT_DIM))

hidden_layer_forward.defvjp(hidden_layer_forward_fwd, hidden_layer_forward_bwd)


# ============ OUTPUT LAYER (Rank 2) ============
@custom_vjp
def output_layer_forward(labels, placeholder_hidden):
    """Forward pass: receive hidden neurons one-by-one, accumulate, compute loss"""
    if rank == 2:
        # Initialize accumulator for hidden output
        hidden_out = jnp.zeros(HIDDEN_DIM)
        
        # Receive each hidden neuron
        for _ in range(HIDDEN_DIM):
            hidden_packet = mpi4jax.recv(
                jnp.zeros(2),  # [index, value]
                source=1,
                tag=TAG_FORWARD_HIDDEN_TO_OUTPUT,
                comm=comm
            )
            hidden_idx = int(hidden_packet[0])
            hidden_value = hidden_packet[1]
            hidden_out = hidden_out.at[hidden_idx].set(hidden_value)
        
        # Compute softmax cross-entropy loss
        logits = hidden_out[:OUTPUT_DIM]
        exp_logits = jnp.exp(logits - jnp.max(logits))
        probs = exp_logits / jnp.sum(exp_logits)
        loss = -jnp.log(probs[labels] + 1e-10)
        
        return loss
    return jnp.array(0.0)

def output_layer_forward_fwd(labels, placeholder_hidden):
    if rank == 2:
        # Initialize accumulator for hidden output
        hidden_out = jnp.zeros(HIDDEN_DIM)
        
        # Receive each hidden neuron
        for _ in range(HIDDEN_DIM):
            hidden_packet = mpi4jax.recv(
                jnp.zeros(2),
                source=1,
                tag=TAG_FORWARD_HIDDEN_TO_OUTPUT,
                comm=comm
            )
            hidden_idx = int(hidden_packet[0])
            hidden_value = hidden_packet[1]
            hidden_out = hidden_out.at[hidden_idx].set(hidden_value)
        
        # Compute softmax cross-entropy loss
        logits = hidden_out[:OUTPUT_DIM]
        exp_logits = jnp.exp(logits - jnp.max(logits))
        probs = exp_logits / jnp.sum(exp_logits)
        loss = -jnp.log(probs[labels] + 1e-10)
        
        return loss, (labels, probs, hidden_out)
    return jnp.array(0.0), (labels, jnp.zeros(OUTPUT_DIM), placeholder_hidden)

def output_layer_forward_bwd(residuals, g):
    if rank == 2:
        labels, probs, hidden_out = residuals
        
        # Gradient of cross-entropy w.r.t. logits
        grad_logits = probs.copy()
        grad_logits = grad_logits.at[labels].add(-1.0)
        grad_logits = grad_logits * g  # Scale by upstream gradient
        
        # Pad gradient to match hidden dimension
        grad_hidden = jnp.zeros(HIDDEN_DIM)
        grad_hidden = grad_hidden.at[:OUTPUT_DIM].set(grad_logits)
        
        # Send gradient for each hidden neuron back to hidden layer
        for h_idx in range(HIDDEN_DIM):
            grad_packet = jnp.array([float(h_idx), grad_hidden[h_idx]])
            mpi4jax.send(grad_packet, dest=1, tag=TAG_BACKWARD_OUTPUT_TO_HIDDEN, comm=comm)
        
        return (jnp.array(0.0), jnp.zeros_like(hidden_out))
    return (jnp.array(0.0), jnp.zeros(HIDDEN_DIM))

output_layer_forward.defvjp(output_layer_forward_fwd, output_layer_forward_bwd)


# ============ MAIN TRAINING FUNCTION ============
def distributed_forward_backward(W, x, y):
    """
    Complete forward and backward pass across all ranks
    Returns loss and gradients
    """
    if rank == 0:
        # Input layer: send neurons one by one
        x_out = input_layer_forward(x)
        return jnp.array(0.0)  # Loss computed on rank 2
    
    elif rank == 1:
        # Hidden layer: receive, compute, send neurons one by one
        placeholder_input = jnp.zeros(INPUT_DIM)
        hidden_out = hidden_layer_forward(W, placeholder_input)
        return jnp.array(0.0)
    
    elif rank == 2:
        # Output layer: accumulate neurons, compute loss
        placeholder_hidden = jnp.zeros(HIDDEN_DIM)
        loss = output_layer_forward(y, placeholder_hidden)
        return loss
    
    return jnp.array(0.0)


# Example usage
if __name__ == "__main__":
    # Initialize parameters on each rank
    key = jax.random.PRNGKey(rank)
    
    if rank == 0:
        x = jax.random.normal(key, (INPUT_DIM,))
        y = 3  # Example label
        
        # Compute gradient - this will trigger backward pass
        grad_fn = grad(lambda x_param: distributed_forward_backward(None, x_param, y))
        grad_x = grad_fn(x)
        
        print(f"Rank {rank}: Sent {INPUT_DIM} neurons, received {INPUT_DIM} gradients")
        print(f"Rank {rank}: grad_x shape: {grad_x.shape}, mean: {jnp.mean(jnp.abs(grad_x)):.6f}")
        
    elif rank == 1:
        W = jax.random.normal(key, (HIDDEN_DIM, INPUT_DIM)) * 0.01
        
        # Use JAX's grad to compute gradients automatically
        grad_fn = grad(lambda w: distributed_forward_backward(w, None, None))
        grad_W = grad_fn(W)
        
        print(f"Rank {rank}: Processed {INPUT_DIM} input neurons -> {HIDDEN_DIM} hidden neurons")
        print(f"Rank {rank}: grad_W shape: {grad_W.shape}, mean: {jnp.mean(jnp.abs(grad_W)):.6f}")
        
    elif rank == 2:
        y = 3  # Label
        loss = distributed_forward_backward(None, None, y)
        print(f"Rank {rank}: Accumulated {HIDDEN_DIM} neurons, computed loss: {loss:.6f}")