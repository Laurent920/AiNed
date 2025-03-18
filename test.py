from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np
import mpi4jax
from mpi4jax import send, recv, bcast

import tree_math
import dataclasses

from mnist_helper import torch_loader, one_hot


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@dataclasses.dataclass
class Neuron_states:
    values: jnp.ndarray
    threshold: jnp.float32
    
    
# Define layer computation with threshold and surrogate gradient
def layer_computation(input_data, weights, neuron_states):
    print(f"Rank {rank}: Shapes in activation: {jnp.shape(input_data)}, {jnp.shape(weights)}")
    
    activations = jnp.dot(input_data, weights) + neuron_states.values 
    jax.debug.print("activation values: {a}", a=activations)
    
    activated_output = jnp.where(activations > neuron_states.threshold, activations, 0.0)
    surrogate_grad = jax.nn.sigmoid(activations) * (1 - jax.nn.sigmoid(activations))
    print(f"Rank {rank}: Shapes in activation: {jnp.shape(activated_output)}")

    # Update the neuron states (old values - activated values) #TODO change the logic if output is different than the activation value
    new_neuron_states = Neuron_states(values=neuron_states.values-activated_output, threshold=neuron_states.threshold)
    return activated_output, surrogate_grad, new_neuron_states

# Define the loss function
def loss_fn(output, target):
    return jnp.mean((output - target) ** 2)


# Train the network
def train(key, layer_sizes, thresholds, num_epochs, learning_rate, batch_size):
    # Initialize parameters (input data for rank 0 and weights for other ranks)
    key, subkey = jax.random.split(key) 
    if rank == 0:
        input_data = init_params(subkey)
    else:
        weights, neuron_states = init_params(subkey)
        target_output = jnp.array([1.0, 0.0], dtype=jnp.float32)  # Example target
        
        token = jax.lax.create_token()
        
    # Simulate a layer running on a separate MPI rank
    if rank == 0:
        # Forward pass (Send input data to Layer 1)
        token = send(input_data, dest=1, comm=comm)
        print(jnp.shape(input_data))
    elif rank == 1:
        # Layer 1: Receive input, compute, and send output
        activations, token = recv(jnp.zeros((layer_sizes[rank-1], )), source=0, comm=comm, token=token)  
        output, surrogate_grad , neuron_states= layer_computation(activations, weights, neuron_states)
        token = send(output, dest=2, comm=comm)  # Send output to Layer 2
        print(f"Output layer {rank}: {(jnp.shape(output))} \n{output}")
        
    elif rank == 2:
        # Layer 2: Receive input and compute final output
        activations, token = recv(jnp.zeros((layer_sizes[rank-1], )), source=1, comm=comm, token=token)  
        output, surrogate_grad, neuron_states = layer_computation(activations, weights, neuron_states)
        print(f"Final Output layer {rank}: {(jnp.shape(output))}, {output}")
        

    # Synchronize all ranks before starting the backward pass
    token = mpi4jax.barrier(comm=comm, token=token)

    if rank == 2:
      # Compute loss and gradients 
        loss, grad_wrt_output = jax.value_and_grad(loss_fn)(output, target_output)
        grad_output = grad_wrt_output * surrogate_grad  # Apply surrogate gradient
        print(f"loss: {loss}, output grad: {(grad_output)}")

        # Compute gradients with respect to inputs and weights
        def layer_loss(input_data, weights):
            output, _, _ = layer_computation(input_data, weights, neuron_states)
            return loss_fn(output, grad_output)

        # Compute gradients using jax.grad
        # grad_weights = jax.grad(layer_loss, 1)(activations, weights)
        # print(f"auto grad: {grad_weights}")

        grad_weights = jnp.outer(activations, grad_output)
        print(f"manual grad: {grad_weights}")
        
        # Update weights (gradient descent)
        weights -= learning_rate * grad_weights
        
        # Backward pass: Send gradient to Layer 1
        token = send(grad_output, dest=1, comm=comm, token=token)
        
    # Backward pass for Layer 1
    if rank == 1:
        # Receive gradients from Layer 2
        grad_output, token = recv(jnp.zeros((layer_sizes[rank], )), source=2, comm=comm, token=token)

        # Compute gradients with respect to inputs and weights
        def layer_loss(input_data, weights):
            output, _, _ = layer_computation(input_data, weights, neuron_states)
            return loss_fn(output, grad_output)

        # Compute gradients using jax.grad
        grad_input, grad_weights = jax.grad(layer_loss, argnums=(0, 1))(activations, weights)

        # Update weights (gradient descent)
        weights -= learning_rate * grad_weights

# Initialize network parameters
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = jax.random.split(key)
  return scale * jax.random.normal(w_key, (n, m))#, scale * jax.random.normal(b_key, (n,))

def init_params(key):# Initialize weights for each layer
    keys = jax.random.split(key, len(layer_sizes))
    
    if rank == 0:
        
        input_data = jax.random.normal(keys[0], layer_sizes[0])
        return input_data
    else:
        weights = random_layer_params(layer_sizes[rank], layer_sizes[rank-1], keys[rank])
        print(jnp.shape(weights))
        
        neuron_states = Neuron_states(values=jnp.zeros(layer_sizes[rank]), threshold=thresholds[rank-1])
        return weights, neuron_states

if __name__ == "__main__":
    key = jax.random.key(42)
    # Network structure and parameters
    layer_sizes = [28, 10, 2] 
    thresholds = [0.1, 0.1]   
    num_epochs = 1
    learning_rate = 0.01       
    batch_size = 10
    
    train(key, layer_sizes, thresholds, num_epochs, learning_rate, batch_size)