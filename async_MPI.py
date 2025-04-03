from socketserver import ThreadingMixIn
from mnist_helper import one_hot, one_hot_encode
from mpi4py import MPI
import jax
import jax.numpy as jnp
from jax import custom_jvp
from jax.tree_util import Partial
from jax import jacfwd, jacrev
import time


import numpy as np
import mpi4jax
from mpi4jax import send, recv, bcast

import tree_math
import dataclasses
from typing import Generic, Any, Union, TypeVar, Tuple

from mnist_helper import torch_loader_manual
# from iris_species_helper import torch_loader

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@dataclasses.dataclass
@tree_math.struct
class Neuron_states:
    values: jnp.ndarray
    threshold: jnp.float32
    
@dataclasses.dataclass
@tree_math.struct
class Batch_Neuron_states:
    Neuron_states: dict

@dataclasses.dataclass
@tree_math.struct
class Params:
    layer_sizes: list[int]
    thresholds: float
    num_epochs: int
    learning_rate: float
    batch_size: int
    

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
    
@jax.jit
def layer_computation(neuron_idx, input_data, weights, neuron_states):    
    activations = jnp.dot(input_data, weights[neuron_idx]) + neuron_states.values
    
    # Debug print (JAX-compatible)
    # jax.debug.print("Rank {rank}: Input values: {input}, Neuron idx: {i}, weights: {w}", rank=rank, input=input_data, i=neuron_idx, w=weights[neuron_idx])
    # jax.debug.print("Rank {rank}: Activation values: {a}", rank=rank, a=activations)
    
    # Handle final layer condition using `jax.lax.cond`
    def last_layer_case(_):
        return jnp.zeros_like(activations), Neuron_states(values=activations, threshold=neuron_states.threshold)
    
    def hidden_layer_case(_):
        activated_output = activation_func(neuron_states, activations)
        new_neuron_states = Neuron_states(values=activations - activated_output, threshold=neuron_states.threshold)
        return activated_output, new_neuron_states
    
    # if rank == size-1:
    #     return last_layer_case(None)
    # else:
    #     return hidden_layer_case(None)
    
    return jax.lax.cond(rank == size-1, last_layer_case, hidden_layer_case, None)

# Define the loss function
def loss_fn(output, target):
    return -jnp.mean((output - target) ** 2)


def predict(weights, empty_neuron_states, token, batch_data=None):
    @jax.jit
    def input_layer(token):
        # Forward pass (Send input data to Layer 1)
        def send_input(input_neuron_idx, token):
            return send(jnp.array([input_neuron_idx, x[input_neuron_idx]]), dest=1, tag=0, comm=comm, token=token)
        
        token = jax.lax.fori_loop(0, x.shape[0], send_input, token) 
        
        # Send -1 to signal all the data has been sent  
        token = send(jnp.array([-1.0, 0.0]), dest=1, tag=0, comm=comm)

        return token, jnp.zeros(()), empty_neuron_states
    
    # @jax.jit
    # def other_layers(token):
    #     def cond(state):
    #         neuron_idx = state            
    #         return neuron_idx != -1
        
    #     def forward_pass(state):
    #         def hidden_layers():
    #             output, new_neuron_states= layer_computation(int(neuron_idx), activations, weights, neuron_states)
    #             neuron_states = new_neuron_states
    #             for idx, out_val in enumerate(output):
    #                 token = send(jnp.array([idx, out_val]), dest=rank+1, tag=0, comm=comm, token=token) 
    #             return token, activations, neuron_states
            
    #         def output_layer():
    #             output, new_neuron_states = layer_computation(int(neuron_idx), activations, weights, neuron_states)
    #             neuron_states = new_neuron_states
    #             output = new_neuron_states.values
    #             return token, activations, neuron_states
            
    #         (neuron_idx, activations), token = recv(jnp.zeros((2,)), source=rank-1, tag=0, comm=comm, token=token)
            
    #         jax.lax.cond(rank==size-1, output_layer, hidden_layers, )
    #         return  
        
    #     initial_state = ()
    #     final_state = jax.lax.while_loop(cond, forward_pass, initial_state)

    #     return
    
    # token, activations, neuron_states = jax.lax.cond(rank==0, input_layer, other_layers, token)
    
    all_outputs = jnp.zeros((batch_size, layer_sizes[-1]))
    for batch_nb in range(batch_size):
        neuron_states = empty_neuron_states  
        if rank == 0:
            # Forward pass (Send input data to Layer 1)
            x = batch_data[batch_nb]

            # Python Loop
            for input_neuron_idx, data in enumerate(batch_data[batch_nb]):
                # print(data.dtype)
                if data <= 0:
                    continue
                token = send(jnp.array([input_neuron_idx, data]), dest=1, tag=0, comm=comm, token=token)
            token = send(jnp.array([-1.0, 0.0]), dest=1, tag=0, comm=comm, token=token)
            activations = jnp.zeros(())
            # print(f"Input layer: {jnp.shape(x)} {x}")

            # JAX Loop
            # token, activations, neuron_states = input_layer(token)
        else:
            # Simulate a layer running on a separate MPI rank
            while True:
                # Receive input from previous layer
                (neuron_idx, activations), token = recv(jnp.zeros((2,)), source=rank-1, tag=0, comm=comm, token=token)
                # print(f"Received data in layer {rank} with neuron_idx: {neuron_idx}, {neuron_idx.dtype}, activations: {activations}")
                
                # Break if all the inputs have been processed (idx=-1)
                if neuron_idx == -1:
                    # print(f"Rank {rank} exiting, received -1")
                    if rank == 1:
                        token = send(jnp.array([-1.0, 0.0]), dest=rank+1, tag=0, comm=comm)  # Send -1 to next layer                        
                    break
                output, new_neuron_states= layer_computation(int(neuron_idx), activations, weights, neuron_states)
                neuron_states = new_neuron_states


                if rank == 1:
                    # Hidden layers: Receive input, compute, and send output
                    for idx, out_val in enumerate(output):
                        if out_val <= 0:
                            continue
                        token = send(jnp.array([idx, out_val]), dest=rank+1, tag=0, comm=comm)  # Send output to next layer 
                elif rank == size-1:
                    # Layer 2: Receive input and compute final output
                    all_outputs = all_outputs.at[batch_nb].set(new_neuron_states.values)
        
    # Synchronize all ranks before starting the backward pass
    token = mpi4jax.barrier(comm=comm, token=token)
    
    return token, all_outputs, neuron_states
        
        
# Train the network
def train(params: Params, token, weights, batch_iterator):     
    """
    tag 0:  forward and backward computation data, format: (previous_layer_neuron_index, neuron_value)
            end of input is encoded with the index -1
    tag 10: data labels
    """   
    empty_neuron_states = Neuron_states(values=jnp.zeros((layer_sizes[rank])), threshold=thresholds[(rank-1)%len(thresholds)])    
    for epoch in range(num_epochs):
        for _ in range(total_batches):
            neuron_states = empty_neuron_states
            if rank == 0:
                batch_x, batch_y = next(batch_iterator)
                print(f"Batch_y: {batch_y}")
                
                token, activations, new_neuron_states = predict(weights, neuron_states, token, batch_data=batch_x)
                token = send(batch_y, dest=size-1, tag=10,comm=comm)
            else:
                token, activations, new_neuron_states = predict(weights, neuron_states, token)

            if rank == size-1:
                y, token = recv(jnp.zeros((batch_size, )), source=0, tag=10, comm=comm, token=token)  
                y = jnp.array(one_hot_encode(y, num_classes=layer_sizes[-1]))
                print(f"received batch_y: {y}")
                
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
                token = send(grad_output, dest=rank-1, tag=0, comm=comm, token=token)
            elif rank == 1:
                # Receive gradients from Layer 2
                grad_output, token = recv(jnp.zeros((layer_sizes[rank-1],)), source=rank+1, tag=0, comm=comm, token=token)
                print(jnp.shape(grad_output))
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
    
def accuracy(params, data):
    x, y = data
    
    weights, neuron_states, token = params
    predicted_class = jnp.argmax(predict(x, weights, neuron_states, token))
    return jnp.mean(predicted_class == y)
    
# Initialize network parameters
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = jax.random.split(key)
    return scale * jax.random.normal(w_key, (n, m))#, scale * jax.random.normal(b_key, (n,))
    # return jnp.full((n, m), 0.1)

def init_params(key, load_file=False):# Initialize weights for each layer
    keys = jax.random.split(key, len(layer_sizes))
    
    if rank != 0:
        neuron_states = Neuron_states(values=jnp.zeros(layer_sizes[rank]), threshold=thresholds[rank-1])

        if load_file:
            w_data = np.load("tensor_data.npz")
            for i, k in enumerate(w_data.files):
                if i == rank-1:
                    weights = jnp.array(w_data[k])
                    print(weights.shape)
                    return weights, neuron_states                
                
        weights = random_layer_params(layer_sizes[rank], layer_sizes[rank-1], keys[rank])        
        return weights, neuron_states


def test_surrogate_grad():
    neuron_states = Neuron_states(values=jnp.zeros((3,)), threshold=0.25)
    activations = jnp.array([0.1, 0.2, 0.3])
    print(f"input activations: {activations}, threshold: {neuron_states.threshold}")
    output = activation_func(neuron_states, activations)
    
    output, grad_output = jax.vmap(jax.value_and_grad(Partial(activation_func, neuron_states)))(activations)
    print(f"output: {output}, output grad: {grad_output}")
    
    # test layer grad
    weights = np.ones((3,3))
    if rank != size-1:
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
    layer_sizes = [28*28, 128, 10]
    load_file = True
    # layer_sizes = [4, 5, 3] 
    # load_file = False
    thresholds = [0, 0]  
    num_epochs = 1
    learning_rate = 0.01
    batch_size = 128

    params = Params(
        layer_sizes=layer_sizes, 
        thresholds=thresholds, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate, 
        batch_size=batch_size
    )

    if rank == size-1:
        correct_predictions = 0
        total_samples = 0 
        all_epoch_accuracies = []
    # test_surrogate_grad()

    # Initialize parameters (input data for rank 0 and weights for other ranks)
    key, subkey = jax.random.split(key) 
    if rank != 0:
        weights, neuron_states = init_params(subkey, load_file=load_file)
        total_batches = 0
        batch_iterator = None
    if rank == 0:
        # Preprocess the data 
        training_generator, train_set, test_set, total_batches = torch_loader_manual(batch_size)
        batch_iterator = iter(training_generator)

        weights = jnp.zeros(layer_sizes[0])
        
    # Broadcast total_batches to all other ranks
    total_batches, token = bcast(total_batches, root=0, comm=comm)
    
    # train(params, token, weights, batch_iterator)
    # init_x = jnp.array(1.0)  # Initial value
    # num_steps = 5  # Number of iterations
    # final_x, results = jax.lax.scan(predict, init_x, xs=None, length=num_steps)
    
    token = mpi4jax.barrier(comm=comm, token=token)
    start_time = time.time()

    empty_neuron_states = Neuron_states(values=jnp.zeros((layer_sizes[rank])), threshold=thresholds[(rank-1)%len(thresholds)])    
    for epoch in range(num_epochs):
        if rank == size-1:
            epoch_correct = 0
            epoch_total = 0
        
        for i in range(total_batches):
            neuron_states = empty_neuron_states
            if rank == 0:
                batch_x, batch_y = next(batch_iterator)
                batch_y = jnp.array(batch_y, dtype=jnp.float32)
                # print(f"Batch_y: {batch_y}, {batch_y.dtype}")
                
                token, activations, new_neuron_states = predict(weights, neuron_states, token, batch_data=jnp.array(batch_x))
                token = send(batch_y, dest=size-1, tag=10,comm=comm, token=token)
            else:
                token, outputs, new_neuron_states = predict(weights, neuron_states, token)
                
                if rank == size-1:
                    y, token = recv(jnp.zeros((batch_size,)), source=0, tag=10, comm=comm, token=token)
                    # Get predictions (indices of max values)
                    predictions = jnp.argmax(outputs, axis=-1)
                    
                    # Calculate accuracy for this batch
                    batch_correct = jnp.sum(predictions == y)
                    epoch_correct += batch_correct
                    epoch_total += batch_size
                    
                    jax.debug.print("Batch {}: Predictions: {}, True: {}, Correct: {}/{}, network output: {}",
                                i, predictions, y, batch_correct, len(y), new_neuron_states.values)
                        
        # Calculate epoch accuracy
        if rank == size-1:
            epoch_accuracy = epoch_correct / epoch_total
            all_epoch_accuracies.append(epoch_accuracy)
            correct_predictions += epoch_correct
            total_samples += epoch_total
            
            jax.debug.print("Epoch {} Accuracy: {:.2f}%", epoch, epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")

    # Final accuracy
    if rank == size-1:
        total_accuracy = correct_predictions / total_samples
        jax.debug.print("Final Accuracy: {:.2f}%", total_accuracy * 100)
        
    jax.block_until_ready(token)

    # Synchronize all MPI processes again
    token = mpi4jax.barrier(comm=comm, token=token)
    end_time = time.time()

    if rank == 0:
        print(f"Execution Time: {end_time - start_time:.6f} seconds")