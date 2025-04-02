import os

USE_CPU_ONLY = True
flags = os.environ.get("XLA_FLAGS", "")
if USE_CPU_ONLY:
    flags += " --xla_force_host_platform_device_count=8"
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = flags


import jax
devices = jax.devices()
print(devices)
import jax.numpy as jnp
from jax import custom_jvp, device_get, device_put
from jax.tree_util import Partial
from jax import jacfwd, jacrev
from jax import device_put, device_get


import numpy as np

import tree_math
import dataclasses
from typing import Generic, Any, Union, TypeVar, Tuple

# from mnist_helper import torch_loader, one_hot
from iris_species_helper import torch_loader


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

def update_state(nn_params, layer_name, new_state):
    """Update the state of a specific layer in nn_params."""
    updated_params = nn_params.copy()  # Shallow copy of the outer dict
    updated_params[layer_name] = {
        **nn_params[layer_name],  # Preserve existing weights/bias
        "state": dataclasses.replace(nn_params[layer_name]["state"], **new_state)
    }
    return updated_params

def reset_all_states(nn_params, reset_value=0.0):
    """Reset all neuron states while preserving structure and devices."""
    def process_node(node):
        # Only process nodes that are Neuron_states
        if isinstance(node, Neuron_states):
            new_values = jnp.zeros_like(node.values) + reset_value
            new_state = dataclasses.replace(
                node,
                values=new_values
            )
            return new_state
        return node
    
    return jax.tree.map(
        process_node,
        nn_params,
        is_leaf=lambda x: isinstance(x, Neuron_states)  # Treat Neuron_states as leaves
    )

def update_layer_with_gradients(nn_params, layer_name, grads, learning_rate):
    """Update weights/bias while preserving states as leaves."""
    # Extract the target layer
    layer_params = nn_params[layer_name]
    
    # Gradient updates 
    updated_layer = {
        **layer_params,
        "weights": layer_params["weights"] - learning_rate * grads["weights"],
        "bias": layer_params["bias"] - learning_rate * grads["bias"],
        # State remains untouched (automatically preserved as a leaf)
    }
    
    # Return new params with updated layer
    return {**nn_params, layer_name: updated_layer}


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

def predict(batch_x, nn_params):
    for x in batch_x:
        print()
        
    # Synchronize all ranks before starting the backward pass
    return 
        
        
# Train the network
def train(key, layer_sizes, thresholds, num_epochs, learning_rate, batch_size):    
    # Initialize parameters (input data for rank 0 and weights for other ranks)
    key, subkey = jax.random.split(key) 
    
    training_generator, train, test = torch_loader(batch_size)
    nn_params = init_params(subkey, layer_sizes, thresholds)

    for epoch in range(num_epochs):
        for batch_x, y in training_generator:
            new_nn_params = predict(batch_x, nn_params)

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
    accuracy((weights, neuron_states, token), train)
    
def accuracy(params, data):
    x, y = data
    weights, neuron_states, token = params
    predicted_class = jnp.argmax(predict(x, weights, neuron_states, token))
    return jnp.mean(predicted_class == y)
    
# Initialize network parameters
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = jax.random.split(key)
    return scale * jax.random.normal(w_key, (n, m))#, scale * jax.random.normal(b_key, (n,))

def init_params(key, layer_sizes, thresholds):# Initialize weights for each layer
    keys = jax.random.split(key, len(layer_sizes))
    
    nn_params = {}
    for i, layer_size in enumerate(layer_sizes):
        if i == 0:
            continue
        nn_params[f"layer_{i}"] = device_put(random_layer_params(layer_sizes[i], layer_sizes[i-1], keys[i-1]), devices[i])
        print(jnp.shape(nn_params[f"layer_{i}"]["weights"]), jnp.shape(nn_params[f"layer_{i}"]["bias"]), nn_params[f"layer_{i}"]["weights"].devices())
        
        state = Neuron_states(values=jnp.zeros(layer_sizes[i]), threshold=thresholds[i-1])
        nn_params[f"layer_{i}"]["state"] = device_put(state, devices[i])
        print(nn_params[f"layer_{1}"]["state"].threshold.devices())
    return nn_params


def test_surrogate_grad():
    neuron_states = Neuron_states(values=jnp.zeros((3,)), threshold=0.25)
    activations = jnp.array([0.1, 0.2, 0.3])
    print(f"input activations: {activations}, threshold: {neuron_states.threshold}")
    output = activation_func(neuron_states, activations)
    
    output, grad_output = jax.vmap(jax.value_and_grad(Partial(activation_func, neuron_states)))(activations)
    print(f"output: {output}, output grad: {grad_output}")
    
    # test layer grad
    weights = np.ones((3,3))
    if True: #rank != size-1:
        # def layer_loss(activations, weights):
        #     output, _ = layer_computation(activations, weights, neuron_states)
        #     return loss_fn(output, grad_output)
        # jax.grad(layer_loss, 1)(activations, weights)
        
        # activation, new_neuron_states = layer_computation(activations, weights, neuron_states)
        # print(activation, new_neuron_states)
        
        value, grad = jax.vmap(jax.value_and_grad(Partial(layer_computation, weights=weights, neuron_states=neuron_states)))(activations)
        print(value, grad)
    
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