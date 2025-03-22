from mpi4py import MPI
import jax
import jax.numpy as jnp
from jax import custom_jvp
from jax.tree_util import Partial

import numpy as np
import mpi4jax
from mpi4jax import send, recv, bcast

import tree_math
import dataclasses
from typing import Generic, Any, Union, TypeVar, Tuple

# from mnist_helper import torch_loader, one_hot
from iris_species_helper import torch_loader
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@dataclasses.dataclass
@tree_math.struct
class Neuron_states:
    values: jnp.ndarray
    threshold: jnp.float32

@custom_jvp # If threshold == 0 then this behaves as a ReLu activation function 
def activation_func(neuron_states, activations):
    return jnp.where(activations > neuron_states.threshold, activations, 0.0)

@activation_func.defjvp
def activation_func_jvp(primals, tangents, k=1.0):
    # Surrogate gradient, redefine the function for the backward pass
    neuron_states, activations, = primals
    neuron_states_dot, activations_dot, = tangents
    ans = activation_func(neuron_states, activations)
    ans_dot = jnp.where(activations > neuron_states.threshold, activations, 0.0)
    return ans, ans_dot
    
    
def layer_computation(input_data, weights, neuron_states):
    print(f"Rank {rank}: Shapes in activation: {jnp.shape(input_data)}, {jnp.shape(weights)}")
    
    activations = jnp.dot(input_data, weights) + neuron_states.values 
    jax.debug.print("activation values: {a}", a=activations)
    if rank == size-1:
        return Neuron_states(values=activations, threshold=neuron_states.threshold)
    
    activated_output = activation_func(neuron_states, activations)
    print(f"Rank {rank}: Shapes in activation: {jnp.shape(activated_output)}")

    # Update the neuron states (old values - activated values) #TODO change the logic if output is different than the activation value
    new_neuron_states = Neuron_states(values=activations-activated_output, threshold=neuron_states.threshold)
    return activated_output, new_neuron_states

# Define the loss function
def loss_fn(output, target):
    return -jnp.mean((output - target) ** 2)

def predict(x, weights, neuron_states, token):
    # Simulate a layer running on a separate MPI rank
    if rank == 0:
        # Forward pass (Send input data to Layer 1)
        token = send(x, dest=1, comm=comm)
        print(f"Input layer: {jnp.shape(x)}\n {x}")
        return token, 0, 0
    elif rank == 1:
        # Layer 1: Receive input, compute, and send output
        activations, token = recv(jnp.zeros((layer_sizes[rank-1], )), source=rank-1, comm=comm, token=token)  
        output, new_neuron_states= layer_computation(activations, weights, neuron_states)
        token = send(output, dest=rank+1, comm=comm)  # Send output to Layer 2
        print(f"Output layer {rank}: {(jnp.shape(output))} \n{output}")
        return token, activations, new_neuron_states
    elif rank == size-1:
        # Layer 2: Receive input and compute final output
        activations, token = recv(jnp.zeros((layer_sizes[rank-1], )), source=rank-1, comm=comm, token=token)  
        new_neuron_states = layer_computation(activations, weights, neuron_states)
        output = new_neuron_states.values
        print(f"Output layer {rank}: {(jnp.shape(output))} \n{output}")
        return token, activations, new_neuron_states  
        
        
# Train the network
def train(key, layer_sizes, thresholds, num_epochs, learning_rate, batch_size):    
    # Initialize parameters (input data for rank 0 and weights for other ranks)
    key, subkey = jax.random.split(key) 
    
    training_generator = torch_loader(batch_size)
    if rank != 0:
        weights, neuron_states = init_params(subkey)
        # y = jnp.array([1.0, 0.0], dtype=jnp.float32)  # Example target
    else:
        # Preprocess the data 

        weights = jnp.zeros(layer_sizes[0])

    empty_neuron_states = Neuron_states(values=jnp.zeros((layer_sizes[rank])), threshold=thresholds[(rank-1)%len(thresholds)])    
    token = jax.lax.create_token()    
    for epoch in range(num_epochs):
        for batch_x, y in training_generator:
            for x in batch_x:
                neuron_states = empty_neuron_states
                token, activations, new_neuron_states = predict(x, weights, neuron_states, token)              

            # Synchronize all ranks before starting the backward pass
            token = mpi4jax.barrier(comm=comm, token=token)

            if rank == size-1:
                # Compute loss and gradients 
                output = new_neuron_states.values - jax.scipy.special.logsumexp(new_neuron_states.values)
                print(f"Final Output layer {rank}: {(jnp.shape(output))}, {output}")
                
                loss, grad_output = jax.value_and_grad(loss_fn)(output, y)
                print(f"loss: {loss}, output grad: {(grad_output)}")

                # Compute gradients with respect to inputs and weights
                def layer_loss(input_data, weights):
                    output = layer_computation(input_data, weights, neuron_states)
                    return loss_fn(output.values, grad_output)

                # Compute gradients using jax.grad
                grad_weights = jax.grad(layer_loss, 1)(activations, weights)
                print(f"auto grad: {grad_weights}")
                
                # grad_weights = jnp.outer(activations, grad_output)
                # print(f"manual grad: {grad_weights}")
                
                # Update weights (gradient descent)
                weights -= learning_rate * grad_weights
                
                # Backward pass: Send gradient to Layer 1
                token = send(grad_output, dest=rank-1, comm=comm, token=token)
            elif rank == 1:
                # Receive gradients from Layer 2
                grad_output, token = recv(jnp.zeros((layer_sizes[rank], )), source=rank+1, comm=comm, token=token)

                # Compute gradients with respect to inputs and weights
                def layer_loss(input_data, weights):
                    output, _ = layer_computation(input_data, weights, neuron_states)
                    return loss_fn(output, grad_output)

                # Compute gradients using jax.grad
                grad_weights = jax.grad(layer_loss, 1)(activations, weights)
                print(f"auto grad: {grad_weights}")

                # grad_weights = jnp.outer(activations, grad_output)
                # print(f"manual grad: {grad_weights}")
                
                # Update weights (gradient descent)
                weights -= learning_rate * grad_weights
            neuron_states = new_neuron_states
            break
    # TODO Compute the accuracy of the training
    
# Initialize network parameters
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = jax.random.split(key)
    return scale * jax.random.normal(w_key, (n, m))#, scale * jax.random.normal(b_key, (n,))

def init_params(key):# Initialize weights for each layer
    keys = jax.random.split(key, len(layer_sizes))
    
    if rank != 0:
        weights = random_layer_params(layer_sizes[rank], layer_sizes[rank-1], keys[rank])
        print(jnp.shape(weights))
        
        neuron_states = Neuron_states(values=jnp.zeros(layer_sizes[rank]), threshold=thresholds[rank-1])
        return weights, neuron_states


def test_surrogate_grad():
    neuron_states = Neuron_states(values=jnp.zeros((3,)), threshold=0.25)
    activations = jnp.array([0.1, 0.2, 0.3])
    print(f"input activations: {activations}, threshold: {neuron_states.threshold}")
    output = activation_func(neuron_states, activations)
    
    output, grad_output = jax.vmap(jax.value_and_grad(Partial(activation_func, neuron_states)))(activations)
    print(f"output: {output}, output grad: {grad_output}")
    
    # test layer grad
    # weights = np.ones((3,3))
    # if rank != size-1:
    #     def layer_loss(activations, weights):
    #         output, _ = layer_computation(activations, weights, neuron_states)
    #         return loss_fn(output, grad_output)
    #     jax.grad(layer_loss, 1)(activations, weights)
        
    #     activation, new_neuron_states = layer_computation(activations, weights, neuron_states)
    #     print(activation, new_neuron_states)
        
    #     value, grad = jax.vmap(jax.grad(Partial(layer_computation, weights=weights, neuron_states=neuron_states)))(activations)
    #     print(value, grad)
    
if __name__ == "__main__":
    key = jax.random.key(42)
    # Network structure and parameters
    # layer_sizes = [28*28, 128, 10] 
    layer_sizes = [4, 5, 3] 
    thresholds = [0, 0]   
    num_epochs = 1
    learning_rate = 0.01       
    batch_size = 1
    
    # test_surrogate_grad()
    train(key, layer_sizes, thresholds, num_epochs, learning_rate, batch_size)