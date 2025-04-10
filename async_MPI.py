from socketserver import ThreadingMixIn
from mnist_helper import one_hot, one_hot_encode
from mpi4py import MPI
import jax
import jax.numpy as jnp
from jax import custom_jvp, jit
from jax.tree_util import Partial
from jax import jacfwd, jacrev
import time

import json
import os
import sys
import numpy as np
import mpi4jax
from mpi4jax import send, recv, bcast

import tree_math
import dataclasses
from typing import Generic, Any, Union, TypeVar, Tuple

from mnist_helper import torch_loader_manual
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
    residuals: jnp.ndarray
    
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
    

# region INFERENCE
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
    
@jit
def layer_computation(neuron_idx, layer_input, weights, neuron_states):    
    activations = jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values
    #TODO being able to compute multiple incoming index neurons
    
    # jax.debug.print("Rank {rank}: Input values: {input}, Neuron idx: {i}, weights: {w}", rank=rank, input=layer_input, i=neuron_idx, w=weights[neuron_idx])
    # jax.debug.print("Rank {}: Activation values: {}", rank, activations)
    
    # Handle final layer condition using `jax.lax.cond`
    def last_layer_case(_):
        return jnp.zeros_like(activations), Neuron_states(values=activations, threshold=neuron_states.threshold)
    
    def hidden_layer_case(_):
        activated_output = activation_func(neuron_states, activations)
        new_neuron_states = Neuron_states(values=activations - activated_output, threshold=neuron_states.threshold)
        return activated_output, new_neuron_states
    
    return jax.lax.cond(rank == size-1, last_layer_case, hidden_layer_case, None)


def predict(weights, empty_neuron_states, token, batch_data: jnp.ndarray):
    #region JAX loop
    def input_layer(args):
        token, neuron_states, x = args
        # Forward pass (Send input data to Layer 1)
        def send_input(i, carry):
            token, count = carry
            data = x[i]

            def send_data(t):
                return send(jnp.array([i, data]), dest=1, tag=0, comm=comm, token=t), count + 1

            def skip_data(t):
                return t, count

            token, count = jax.lax.cond(
                data != 0,
                send_data,
                skip_data,
                operand=token
            )
            return token, count

        # Initial carry: (token, iteration=0)
        token, iteration = jax.lax.fori_loop(0, x.shape[0], send_input, (token, 0))

        # Send end signal
        token = send(jnp.array([-1.0, 0.0]), dest=1, tag=0, comm=comm, token=token)

        return token, jnp.zeros(()), neuron_states, iteration+1
    
    def other_layers(args):
        token, neuron_states, _ = args
        def cond(state): # end of input has been reached -> break the while loop
            _, _, _, neuron_idx, _= state            
            return neuron_idx != -1
        
        def forward_pass(state):
            token, layer_input, neuron_states, neuron_idx, iteration = state
            
            def hidden_layers(input): # Send activation to the next layers
                token, activated_output = input

                def send_activation(i, token):
                    out_val = activated_output[i]
                    return jax.lax.cond(
                        out_val != 0,
                        lambda t: send(jnp.array([i, out_val]), dest=rank+1, tag=0, comm=comm, token=t),
                        lambda t: t,
                        operand=token
                    )

                token = jax.lax.fori_loop(0, activated_output.shape[0], send_activation, token)
                return token
            
            # def output_layer(input):
            #     token, activated_output = input
            #     neuron_states = new_neuron_states
            #     output = new_neuron_states.values
            #     return token
            
            # Receive neuron values from previous layers and compute the activations
            (neuron_idx, layer_input), token = recv(jnp.zeros((2,)), source=rank-1, tag=0, comm=comm, token=token)
            activated_output, new_neuron_states= layer_computation(neuron_idx.astype(int), layer_input, weights, neuron_states)

            # jax.debug.print("Rank {} received data {} at neuron idx: {}", rank, layer_input, neuron_idx)
            neuron_states = new_neuron_states
            token = jax.lax.cond(rank==size-1, lambda input: input[0], hidden_layers, (token, activated_output))
            return token, layer_input, neuron_states, neuron_idx, iteration+1
        
        neuron_idx = 0
        layer_input = jnp.zeros(())
        initial_state = (token, layer_input, neuron_states, neuron_idx, 0)
        
        # Loop until the rank receives a -1 neuron_idx
        token, layer_input, neuron_states, neuron_idx, iteration = jax.lax.while_loop(cond, forward_pass, initial_state)
        # jax.debug.print("rank {} exited the while loop with neuron_idx: {} and neuron state {}", rank, neuron_idx, neuron_states)
        
        # Send -1 to the next rank when all incoming data has been processed
        token = jax.lax.cond(
            rank != size - 1,
            lambda t: send(jnp.array([-1.0, 0.0]), dest=rank + 1, tag=0, comm=comm, token=t),
            lambda t: t,
            operand=token
        )
        return token, layer_input, neuron_states, iteration
    
    def loop_over_batches(token, x):
        neuron_states = empty_neuron_states  
        token, layer_input, new_neuron_states, iterations = jax.lax.cond(rank==0, input_layer, other_layers, (token, neuron_states, x))
        
        return token, (new_neuron_states.values, iterations)
    
    # Loop over batches, accumulate output values and return them
    token, (all_outputs, all_iterations) = jax.lax.scan(loop_over_batches, token, batch_data)

    # Synchronize all ranks before starting the backward pass
    token = mpi4jax.barrier(comm=comm, token=token)

    # jax.block_until_ready(all_outputs)
    # jax.debug.print("rank {} finished computing and waiting at the barrier after scanning over {} elements", rank, all_outputs.shape)
    return token, all_outputs, all_iterations


# def predict(weights, empty_neuron_states, token, batch_data: jnp.ndarray):


#region Python loop      
def python_predict(weights, empty_neuron_states, token, batch_data: jnp.ndarray):
    # region Python Loop
    all_outputs = jnp.zeros((batch_size, layer_sizes[-1]))
    for batch_nb in range(batch_size):
        neuron_states = empty_neuron_states  
        
        if rank == 0:
            # Forward pass (Send input data to Layer 1)
            x = batch_data[batch_nb]

            for input_neuron_idx, data in enumerate(batch_data[batch_nb]):
                # print(data.dtype)
                if data <= 0:
                    continue
                token = send(jnp.array([input_neuron_idx, data]), dest=1, tag=0, comm=comm, token=token)
            token = send(jnp.array([-1.0, 0.0]), dest=1, tag=0, comm=comm, token=token)
            layer_input = jnp.zeros(())
            # print(f"Input layer: {jnp.shape(x)} {x}")
        else:
            # Simulate a layer running on a separate MPI rank
            while True:
                # Receive input from previous layer
                (neuron_idx, layer_input), token = recv(jnp.zeros((2,)), source=rank-1, tag=0, comm=comm, token=token)
                # print(f"Received data in layer {rank} with neuron_idx: {neuron_idx}, {neuron_idx.dtype}, layer_input: {layer_input}")
                
                # Break if all the inputs have been processed (idx=-1)
                if neuron_idx == -1:
                    if rank == 1:
                        token = send(jnp.array([-1.0, 0.0]), dest=rank+1, tag=0, comm=comm)  # Send -1 to next layer  
                    elif rank == size-1:
                        # Last layer: store the output neurons values
                        all_outputs = all_outputs.at[batch_nb].set(new_neuron_states.values)                      
                    break
                output, new_neuron_states= layer_computation(int(neuron_idx), layer_input, weights, neuron_states)
                neuron_states = new_neuron_states

                if rank == 1:
                    # Hidden layers: Receive input, compute, and send output
                    for idx, out_val in enumerate(output):
                        if out_val <= 0:
                            continue
                        token = send(jnp.array([idx, out_val]), dest=rank+1, tag=0, comm=comm)  # Send output to next layer 
        
    # Synchronize all ranks before starting the backward pass
    token = mpi4jax.barrier(comm=comm, token=token)
    
    return token, all_outputs, all_outputs[-1]

# Define the loss function
def loss_fn(output, target):
    return -jnp.mean((output - target) ** 2)
        
# region TRAINING
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
                
                token, layer_input, new_neuron_states = predict(weights, neuron_states, token, batch_data=batch_x)
                token = send(batch_y, dest=size-1, tag=10,comm=comm)
            else:
                token, layer_input, new_neuron_states = predict(weights, neuron_states, token)

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
                grad_weights = jax.grad(layer_loss, 1)(layer_input, weights)
                print(f"auto grad: {grad_weights}")
                
                # grad_weights = jnp.outer(layer_input, grad_output)
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
                grad_weights = jax.grad(layer_loss, 1)(layer_input, weights)
                print(f"auto grad: {grad_weights}")

                # grad_weights = jnp.outer(layer_input, grad_output)
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

def init_params(key, load_file=False, best=False):
    # Initialize weights for each layer
    keys = jax.random.split(key, len(layer_sizes))
    
    if rank != 0:
        neuron_states = Neuron_states(values=jnp.zeros(layer_sizes[rank]), threshold=thresholds[rank-1], residuals=np.zeros((layer_sizes[rank-1], layer_sizes[rank])))

        if load_file:
            # 
            filename = f"tensor_data_{'_'.join(map(str, layer_sizes))}.npz"
            if best:
                filename = "best_" + filename
            filepath = os.path.join("tensor_data", filename)
            w_data = np.load(filepath)
            for i, k in enumerate(w_data.files):
                if i == rank-1:
                    weights = jnp.array(w_data[k])
                    print(weights.shape)
                    return weights, neuron_states                
        
        # Random initializatoin of the weights       
        weights = random_layer_params(layer_sizes[rank], layer_sizes[rank-1], keys[rank])        
        return weights, neuron_states


def test_surrogate_grad():
    neuron_states = Neuron_states(values=jnp.zeros((3,)), threshold=0.25)
    layer_input = jnp.array([0.1, 0.2, 0.3])
    print(f"input layer_input: {layer_input}, threshold: {neuron_states.threshold}")
    output = activation_func(neuron_states, layer_input)
    
    output, grad_output = jax.vmap(jax.value_and_grad(Partial(activation_func, neuron_states)))(layer_input)
    print(f"output: {output}, output grad: {grad_output}")
    
    # test layer grad
    weights = np.ones((3,3))
    if rank != size-1:
        # def layer_loss(layer_input, weights):
        #     output, _ = layer_computation(layer_input, weights, neuron_states)
        #     return loss_fn(output, grad_output)
        # jax.grad(layer_loss, 1)(layer_input, weights)
        
        # activation, new_neuron_states = layer_computation(layer_input, weights, neuron_states)
        # print(activation, new_neuron_states)
        
        value, grad = jax.vmap(jax.value_and_grad(Partial(layer_computation, weights=weights, neuron_states=neuron_states)))(layer_input)
        print(value, grad)
    
def pad_batch(batch_x, batch_y, batch_size):
    # Pad the x data with 0 and the y data with nan for the last batch
    current_size = batch_y.shape[0]
    if current_size < batch_size:
        pad_amount = batch_size - current_size
        pad_y = jnp.full((pad_amount,), jnp.nan)
        pad_x = jnp.zeros((pad_amount, batch_x.shape[1]))  

        batch_y = jnp.concatenate([batch_y, pad_y], axis=0)
        batch_x = jnp.concatenate([batch_x, pad_x], axis=0)
    
    return batch_x, batch_y

    
if __name__ == "__main__":
    key = jax.random.key(42)
    # Network structure and parameters
    layer_sizes = [28*28, 128, 64, 10]
    layer_sizes = [28*28, 128, 10]
    best = False
    load_file = True
    # layer_sizes = [4, 5, 3] 
    # load_file = False
    thresholds = [0, 0 ,0]  
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
    
    if len(layer_sizes) != size:
        print(f"Error: layer_sizes ({len(layer_sizes)}) must match number of MPI ranks ({size})")
        sys.exit(1)

    if rank == size-1:
        correct_predictions = 0
        total_samples = 0 
        all_epoch_accuracies = []
    # test_surrogate_grad()

    # Initialize parameters (input data for rank 0 and weights for other ranks)
    key, subkey = jax.random.split(key) 
    if rank != 0:
        weights, neuron_states = init_params(subkey, load_file=load_file, best=best)
        total_batches = 0
        batch_iterator = None
    if rank == 0:
        # Preprocess the data 
        training_generator, train_set, test_set, total_batches = torch_loader_manual(batch_size, shuffle=False)
        # training_generator, train_set, test_set, total_batches = torch_loader(batch_size, shuffle=False)

        batch_iterator = iter(training_generator)

        weights = jnp.zeros(layer_sizes[0])
        
    # Broadcast total_batches to all other ranks
    total_batches, token = bcast(total_batches, root=0, comm=comm)
    
    # train(params, token, weights, batch_iterator)
    # init_x = jnp.array(1.0)  # Initial value
    # num_steps = 5  # Number of iterations
    # final_x, results = jax.lax.scan(predict, init_x, xs=None, length=num_steps)
    print(f"number of batches: {total_batches}")
    token = mpi4jax.barrier(comm=comm, token=token)
    start_time = time.time()

    empty_neuron_states = Neuron_states(values=jnp.zeros((layer_sizes[rank])), threshold=thresholds[(rank-1)%len(thresholds)], residuals=np.zeros((layer_sizes[rank-1], layer_sizes[rank])))    
    for epoch in range(num_epochs):
        if rank == size-1:
            epoch_correct = 0
            epoch_total = 0
            
        all_iterations = []
        
        for i in range(total_batches):
            neuron_states = empty_neuron_states
            if rank == 0:
                batch_x, batch_y = next(batch_iterator)
                batch_y = jnp.array(batch_y, dtype=jnp.float32)
                # print(f"Batch_x: {batch_x.shape}, {batch_y.dtype}")
                
                batch_x, batch_y = pad_batch(batch_x, batch_y, batch_size)
                
                token, outputs, iterations = jit(predict)(weights, neuron_states, token, batch_data=jnp.array(batch_x))
                
                all_iterations.append(iterations)

                token = send(batch_y, dest=size-1, tag=10,comm=comm, token=token)
                
                token = send(jnp.mean(jnp.array(all_iterations).flatten()), dest=size-1, tag=20,comm=comm, token=token)
            else:
                token, outputs, iterations = jit(predict)(weights, neuron_states, token, jnp.zeros((batch_size, layer_sizes[0])))
                all_iterations.append(iterations)

                if rank == size-1:

                    y, token = recv(jnp.zeros((batch_size,)), source=0, tag=10, comm=comm, token=token)                    
                    # Get predictions (indices of max values)
                    predictions = jnp.argmax(outputs, axis=-1)
                    
                    # Calculate accuracy for this batch
                    valid_mask = ~jnp.isnan(y)
                    valid_y = y[valid_mask]
                    valid_predictions = predictions[valid_mask]

                    batch_correct = jnp.sum(valid_predictions == valid_y)
                    epoch_correct += batch_correct
                    epoch_total += valid_y.shape[0]

                    
                    jax.debug.print("Batch {}: Predictions: {}, True: {}, Iterations avg: {}, Correct: {}/{}, last network output: {}",
                                i, valid_predictions, valid_y, jnp.mean(iterations), batch_correct, valid_y.shape[0], outputs[-1])
                else:
                    token = send(jnp.mean(jnp.array(all_iterations).flatten()), dest=size-1, tag=20,comm=comm, token=token)
        all_iterations = jnp.array(all_iterations).flatten()
        
        jax.debug.print("Rank {} finished all batches with an average iteration of {} out of {} data points", rank, jnp.mean(all_iterations), all_iterations.shape[0])
        
        # Calculate epoch accuracy
        if rank == size-1:
            epoch_accuracy = epoch_correct / epoch_total
            all_epoch_accuracies.append(epoch_accuracy)
            correct_predictions += epoch_correct
            total_samples += epoch_total
            
            jax.debug.print("Epoch {} Accuracy: {:.2f}%", epoch, epoch_accuracy * 100)
            jax.debug.print("----------------------------\n")
            
            all_iteration_mean = []
            for i in range(size-1):
                mean, token = recv(jnp.zeros(()), source=i, tag=20, comm=comm, token=token)                    
                all_iteration_mean.append(mean)
            all_iteration_mean.append(jnp.mean(all_iterations))

    jax.block_until_ready(token)

    # Synchronize all MPI processes again
    token = mpi4jax.barrier(comm=comm, token=token)
    end_time = time.time()

    if rank == size-1:            
        # Final accuracy
        total_accuracy = correct_predictions / total_samples
        jax.debug.print("Final Accuracy: {:.2f}%", total_accuracy * 100)
        
        # Execution time
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.6f} seconds")
        
        # Set up file path
        filename = "_".join(map(str, layer_sizes)) + ".json"
        if best:
            filename = "best_" + filename
        result_dir = os.path.join("network_results", "mnist")
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, filename)

        # Store the results
        result_data = {
            "time": float(execution_time),
            "processes": size,
            "accuracy": float(total_accuracy),
            "iterations mean": [float(mean) for mean in all_iteration_mean[1:]],
            "layer_sizes": layer_sizes,#[layer for layer in layer_sizes],
            "batch_size": batch_size,
            "thresholds": thresholds
        }


        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=4)

        print(f"Results saved to {result_path}")