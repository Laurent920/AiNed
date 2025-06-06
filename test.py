import enum
from mpi4py import MPI
import jax
import jax.numpy as jnp
import mpi4jax
import dataclasses
import tree_math
from jax import custom_jvp
import pickle
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@dataclasses.dataclass
@tree_math.struct
class Neuron_states:
    values: jnp.ndarray
    threshold: jnp.float32
    input_residuals: jnp.ndarray
    weight_residuals: dict[str, jnp.ndarray]

#region Batched inference
# def clone_neuron_state(template: Neuron_states, batch_size: int) -> Neuron_states:
#     def tile(x):
#         return jnp.broadcast_to(x, (batch_size,) + x.shape)

#     return Neuron_states(
#         values=tile(template.values),
#         threshold=jnp.broadcast_to(template.threshold, (batch_size,)),
#         input_residuals=tile(template.input_residuals),
#         weight_residuals={
#             k: tile(v) for k, v in template.weight_residuals.items()
#         },
#         last_sent_iteration=tile(template.last_sent_iteration)
#     )

# @partial(jax.jit, static_argnames=['params'])
# def predict_batched(params, weights, empty_neuron_states, token, batch_data: jnp.ndarray):
#     """
#         sending: (batch_size, 2): list of neuron_index and values
#     """
#     empty_neuron_states = clone_neuron_state(empty_neuron_states, batch_size)
    
#     def input_layer(args):
#         token, neuron_states, x = args
        
#         x = jax.vmap(lambda x: preprocess_to_sparse_data_padded(x, params.max_nonzero))(x)
#         # TODO preprocess the input before calling predict
        
#         # Forward pass (Send input data to Layer 1)
#         nb_neurons = x.shape[1]
#         def cond_send_input(carry):
#             i, _ = carry
#             out_val = x[:, i]
#             return jnp.logical_and(i < nb_neurons, jnp.any(out_val != -2))

#         def send_input(carry):
#             i, token = carry
#             out_val = x[:, i]
#             token = send(out_val, dest=rank + 1, tag=0, comm=comm, token=token)
#             return i + 1, token

#         iteration, token = jax.lax.while_loop(cond_send_input, send_input, (0, token))

#         # Send end signal
#         token = send(jnp.full((batch_size, 2), -1.0), dest=1, tag=0, comm=comm, token=token)

#         return token, jnp.zeros((batch_size)), neuron_states, iteration
    
#     def other_layers(args):
#         token, neuron_states, _ = args
#         def cond(state): # end of input has been reached -> break the while loop
#             _, _, _, neuron_idx, _= state   
#             # jax.debug.print("Rank {} neuron idx in while cond {}, shape: {}", rank, neuron_idx, neuron_idx.shape)         
#             return jnp.all(neuron_idx != -1)
        
#         def forward_pass(state):
#             token, layer_input, neuron_states, neuron_idx, iteration = state
            
#             def hidden_layers(input): # Send activation to the next layers
#                 token, activated_output = input
#                 nb_neurons = activated_output.shape[1]
#                 activated_output = jax.vmap(lambda x: preprocess_to_sparse_data_padded(x, nb_neurons))(activated_output)

#                 # jax.debug.print("Rank {}, nb neurons: {}", rank, nb_neurons)
#                 def cond_send_activation(carry):
#                     i, _ = carry
#                     out_val = activated_output[:, i]
#                     return jnp.logical_and(i < nb_neurons, jnp.any(out_val != -2))

#                 def send_activation(carry):
#                     i, token = carry
#                     out_val = activated_output[:, i]
#                     token = send(out_val, dest=rank + 1, tag=0, comm=comm, token=token)
#                     return i + 1, token

#                 _, token = jax.lax.while_loop(cond_send_activation, send_activation, (0, token))
#                 return token
            
#             # Receive neuron values from previous layers and compute the activations
#             input_data, token = recv(jnp.zeros((batch_size, 2), dtype=jnp.float32), source=rank-1, tag=0, comm=comm, token=token)
#             # if rank == 2:
#             #     jax.debug.print("{}",input_data)
#             neuron_idx, layer_input = input_data[:, 0].astype(jnp.int32), input_data[:, 1]
#             # jax.debug.print("Rank {} neuron states shape: {} dtype: {}", rank, neuron_states.values.shape, neuron_states.values.dtype)
#             # jax.debug.print("Rank {} received data type {} at neuron idx type: {}", rank, layer_input.dtype, neuron_idx.dtype)
            
#             activated_output, new_neuron_states = jax.vmap(
#                     layer_computation,
#                     in_axes=(0, 0, None, 0)  # neuron_idx[batch], layer_input[batch], weights[shared], neuron_states[batch]
#                     )(neuron_idx, layer_input, weights, neuron_states, params, iteration)
            
#             # activated_output, new_neuron_states = jnp.zeros((layer_sizes[rank])), neuron_states
#             # jax.debug.print("Rank {} activated outputs {}", rank, activated_output.shape)
#             # jax.debug.print("Rank {} received data {} at neuron idx: {}", rank, layer_input, neuron_idx)
#             neuron_states = new_neuron_states
#             token = jax.lax.cond(rank==size-1, lambda input: input[0], hidden_layers, (token, activated_output))
#             return token, layer_input, neuron_states, neuron_idx, iteration+1
        
#         neuron_idx = jnp.zeros((batch_size), dtype=jnp.int32)
#         layer_input = jnp.zeros((batch_size))
#         initial_state = (token, layer_input, neuron_states, neuron_idx, 0)
        
#         # Loop until the rank receives a -1 neuron_idx
#         token, layer_input, neuron_states, neuron_idx, iteration = jax.lax.while_loop(cond, forward_pass, initial_state)
#         # jax.debug.print("rank {} exited the while loop with neuron_idx: {} and neuron state {}", rank, neuron_idx, neuron_states)
        
#         # Send -1 to the next rank when all incoming data has been processed
#         token = jax.lax.cond(
#             rank != size - 1,
#             lambda t: send(jnp.full((batch_size, 2), -1.0), dest=rank + 1, tag=0, comm=comm, token=t),
#             lambda t: t,
#             operand=token
#         )
#         return token, layer_input, neuron_states, iteration-1
       
#     token, all_outputs, all_neuron_states, all_iterations = jax.lax.cond(rank==0, input_layer, other_layers, (token, empty_neuron_states, batch_data))
#     all_outputs = all_neuron_states.values
    
#     # Synchronize all ranks before starting the backward pass
#     token = mpi4jax.barrier(comm=comm, token=token)

#     # jax.block_until_ready(all_outputs)
#     # jax.debug.print("rank {} finished computing and waiting at the barrier after scanning over {} elements", rank, all_outputs.shape)
#     return token, all_outputs, all_iterations, all_neuron_states

#region Python loop      
# def python_predict(weights, empty_neuron_states, token, batch_data: jnp.ndarray):
#     # region Python Loop
#     all_outputs = jnp.zeros((batch_size, layer_sizes[-1]))
#     for batch_nb in range(batch_size):
#         neuron_states = empty_neuron_states  
        
#         if rank == 0:
#             # Forward pass (Send input data to Layer 1)
#             x = batch_data[batch_nb]

#             for input_neuron_idx, data in enumerate(batch_data[batch_nb]):
#                 # print(data.dtype)
#                 if data <= 0:
#                     continue
#                 token = send(jnp.array([input_neuron_idx, data]), dest=1, tag=0, comm=comm, token=token)
#             token = send(jnp.array([-1.0, 0.0]), dest=1, tag=0, comm=comm, token=token)
#             layer_input = jnp.zeros(())
#             # print(f"Input layer: {jnp.shape(x)} {x}")
#         else:
#             # Simulate a layer running on a separate MPI rank
#             while True:
#                 # Receive input from previous layer
#                 (neuron_idx, layer_input), token = recv(jnp.zeros((2,)), source=rank-1, tag=0, comm=comm, token=token)
#                 # print(f"Received data in layer {rank} with neuron_idx: {neuron_idx}, {neuron_idx.dtype}, layer_input: {layer_input}")
                
#                 # Break if all the inputs have been processed (idx=-1)
#                 if neuron_idx == -1:
#                     if rank == 1:
#                         token = send(jnp.array([-1.0, 0.0]), dest=rank+1, tag=0, comm=comm)  # Send -1 to next layer  
#                     elif rank == size-1:
#                         # Last layer: store the output neurons values
#                         all_outputs = all_outputs.at[batch_nb].set(new_neuron_states.values)                      
#                     break
#                 output, new_neuron_states= layer_computation(int(neuron_idx), layer_input, weights, neuron_states)
#                 neuron_states = new_neuron_states

#                 if rank == 1:
#                     # Hidden layers: Receive input, compute, and send output
#                     for idx, out_val in enumerate(output):
#                         if out_val <= 0:
#                             continue
#                         token = send(jnp.array([idx, out_val]), dest=rank+1, tag=0, comm=comm)  # Send output to next layer 
        
#     # Synchronize all ranks before starting the backward pass
#     token = mpi4jax.barrier(comm=comm, token=token)
    
#     return token, all_outputs, all_outputs[-1]
#region test surrogate grad
# def test_surrogate_grad():
#     neuron_states = Neuron_states(values=jnp.zeros((3,)), threshold=0.25)
#     layer_input = jnp.array([0.1, 0.2, 0.3])
#     print(f"input layer_input: {layer_input}, threshold: {neuron_states.threshold}")
#     output = activation_func(neuron_states, layer_input)
    
#     output, grad_output = jax.vmap(jax.value_and_grad(Partial(activation_func, neuron_states)))(layer_input)
#     print(f"output: {output}, output grad: {grad_output}")
    
#     # test layer grad
#     weights = np.ones((3,3))
#     if rank != size-1:
#         # def layer_loss(layer_input, weights):
#         #     output, _ = layer_computation(layer_input, weights, neuron_states)
#         #     return loss_fn(output, grad_output)
#         # jax.grad(layer_loss, 1)(layer_input, weights)
        
#         # activation, new_neuron_states = layer_computation(layer_input, weights, neuron_states)
#         # print(activation, new_neuron_states)
        
#         value, grad = jax.vmap(jax.value_and_grad(Partial(layer_computation, weights=weights, neuron_states=neuron_states)))(layer_input)
#         print(value, grad)
    
#region test
@jax.jit
def send_data():
    data0 = jnp.array([1.6, 4.7, 0., 0.])
    data1 = jnp.array([1., 1.])
    data2 = jnp.arange(1, 11)
    
    token = mpi4jax.send(jnp.zeros(()), dest=size-1, tag=99, comm=comm)

    # token = mpi4jax.send(data0, dest=size-1, tag=0, token=token, comm=comm)
    # token = mpi4jax.send(data1, dest=size-1, tag=10, token=token, comm=comm)
    for data in data2:
        token = mpi4jax.send(data, dest=size-1, tag=2, token=token, comm=comm)
    
    token = mpi4jax.send(jnp.zeros((1,)), dest=size-1, tag=3, token=token, comm=comm)
    return token  # Return token for JAX tracking

@jax.jit
def receive_data():
    recv_buf2 = jnp.array(0)  # Ensure correct dtype
    token = None

    def cond(state):
        _, tag, _ = state
        return tag != 3  # Continue loop until tag == 3

    def body(state):
        recv_buf, _, token = state
        status = MPI.Status()
        recv_buf, token = mpi4jax.recv(recv_buf, source=0, token=token, comm=comm, status=status)
        new_tag = status.Get_tag()
        jax.debug.print("received data: {r}, tag:{t}", r=recv_buf, t=new_tag)
        return recv_buf, new_tag, token

    # Initial state: (recv_buf2, tag, token)
    _, token = mpi4jax.recv(jnp.zeros(()), source=0, tag=99, comm=comm)  # Dummy receive for token initialization

    initial_state = (recv_buf2, jnp.array(-1), token)

    final_state = jax.lax.while_loop(cond, body, initial_state)

    return final_state[0]  # Return received data


@jax.custom_vjp
def test_custom_vjp(input, weight):
    return jnp.zeros(jnp.dot(input, weight))

# Forward pass: returns output and residuals needed for backward
def test_custom_vjp_fwd(input, weight):
    if rank == 0:
        input_prev = input
        output = jnp.dot(input, weight)
        token = mpi4jax.send(output, dest=rank+1, tag=1, comm=comm)
    else:
        input_prev, token = mpi4jax.recv(jnp.zeros(()), source=rank-1, tag=1, comm=comm)
        output = jnp.dot(input_prev, weight)
    
    return output, (input_prev, weight)  # these are saved for use in the backward

# Backward pass: receives residuals and the cotangent (gradient from next layer)
def test_custom_vjp_bwd(residuals, g):
    input, weight = residuals
    print(f"Rank {rank} backward received gradient: {g}")

    grad_input = g * weight  # d(output)/d(input) = weight
    grad_weight = g * input  # d(output)/d(weight) = input
    
    print(f"Rank {rank} grad_input: {grad_input}, grad_weight: {grad_weight}")

    return grad_input, grad_weight

# Register the custom VJP
test_custom_vjp.defvjp(test_custom_vjp_fwd, test_custom_vjp_bwd)

@jax.custom_vjp
def loss_fn(input, weight, target):
    output, residuals = test_custom_vjp_fwd(input, weight)
    # Loss function: Mean Squared Error (MSE)
    return jnp.mean((output - target) ** 2)

# Define the forward pass for the custom JVP
def loss_fn_fwd(input, weight, target):
    output, residuals = test_custom_vjp_fwd(input, weight)
    jax.debug.print("Rank {}, output: {}", rank, output)
    l = jnp.mean((output - target) ** 2)
    jax.debug.print("loss: {}", l)
    return l, (output, residuals)

# Define the backward pass for the custom JVP
def loss_fn_bwd(residuals, g):
    (output, (input, weight)) = residuals
    jax.debug.print("{}",output)
    N = output.size  # Number of elements in output
    # Gradient of the loss with respect to the output
    grad_output = g * (2 / N) * (output - target)
    return grad_output*weight, grad_output*input, None  # Gradient for 'output' and None for 'target' (no gradient wrt target)

# Register the custom JVP with the forward and backward functions
loss_fn.defvjp(loss_fn_fwd, loss_fn_bwd)


# def loss_fn(input, weight, target):
#     output, residuals = test_custom_vjp_fwd(input, weight)
#     jax.debug.print("Rank {}, output: {}", rank, output)
#     l = jnp.mean((output - target) ** 2)
#     jax.debug.print("loss: {}", l)
#     return l

def other_layers(input, weight):
    # Call the forward function to get residuals
    output, residuals = test_custom_vjp_fwd(input, weight)
    jax.debug.print("Rank {}, output: {}", rank, output)

    # Receive gradient from next layer
    next_grad, token = mpi4jax.recv(jnp.zeros(()), source=rank + 1, tag=0, comm=comm)
    jax.debug.print("Rank {}, received gradient: {}", rank, next_grad)

    # Backward pass using saved residuals and received gradient
    grad_input, grad_weight = test_custom_vjp_bwd(residuals, next_grad)
    
    # Send gradient to the previous layer (if not input layer)
    if rank > 0:
        token = mpi4jax.send(grad_input, dest=rank - 1, tag=0, comm=comm, token=token)

    return grad_input, grad_weight

@jax.jit
def process_single_batch(activity, values):
    def body(i, carry):
        activates, values = carry

        def update_if_active_fn(carry):
            activates, values = carry

            # Extract row i
            vals = values[i]  # shape: (128,)

            # Case 1: neuron_val == 0 and activates[j] == 1 → set to 1
            condition = (vals == 0) & (activates[:, 0] == 1)
            new_vals = jnp.where(condition, 1, vals)
            values = values.at[i].set(new_vals)

            # Case 2: neuron_val == 1 and activates[j] == 0 → set activates[j] = 1
            update_activates = (vals == 1) & (activates[:, 0] == 0)
            new_activates = jnp.where(update_activates, 1, activates[:, 0])
            activates = new_activates[:, None]

            return activates, values

        return jax.lax.cond(
            jnp.squeeze(activity[i]),
            update_if_active_fn,
            lambda carry: carry,
            operand=(activates, values)
        )

    # Initial state
    activates = jnp.zeros((values.shape[1], 1), dtype=jnp.int32)

    # Reverse loop with fori_loop
    n = activity.shape[0]
    activates, values = jax.lax.fori_loop(
        0, n,
        lambda idx, carry: body(n - 1 - idx, carry),  # reversed order
        (activates, values)
    )
    return values

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

if __name__ == "__main__":
    # input = 1.0
    # target = 1.0
    # weight = 0.5
    # lr = 0.001

    # for i in range(1):
    #     if rank == size-1:
    #         # Differentiate w.r.t. weight
    #         output, gradient = jax.value_and_grad(loss_fn, argnums=(0, 1))(input, weight, target)
    #         jax.debug.print("Rank {}, output: {}", rank, output)
    #         print("Gradient wrt input and weight:", gradient)
    #         weight -= gradient[1] * lr
    #         token = mpi4jax.send(jnp.array(gradient[0]), dest=rank-1, tag=0, comm=comm)
    #     else:
    #         gradient = other_layers(input, weight)    
    #         print("Gradient wrt input and weight:", gradient)
    #         weight -= gradient[1] * lr

    # ______________________________________________________________________________________________________________________________________    
    # if rank == 0:
    #     send_data()  # JAX-compiled send function

    # if rank == size-1:
    #     data0, data1 = receive_data()  # JAX-compiled receive function
    #     print(f"data0: {data0}")
    #     print(f"data1: {data1}")
    
    # ______________________________________________________________________________________________________________________________________    
    # def weight_res_complete(activity, values):
    #     activates = jnp.zeros((values.shape[1], 1))
    #     for i in reversed(range(len(activity))):
    #         jax.debug.print("in loop {}", i)
    #         if not activity[i]: # update if active input
    #             continue
    #         jax.debug.print("val {}, activates {}", values, activates)
    #         values.at(i).set(jnp.where(values[i] > 0, 1, 0))
    #         activates = jnp.where(activates, 1, 0)
    #     return jnp.array(values)

    # weight_res = {
    #     "activity": jnp.array([[True, True, True, False], [False, True, True, True]]),  # Example
    #     "values": jnp.array([[
    #         [0, 0, 0, 0],  # neuron 0
    #         [0, 0, 1, 0],  # neuron 1
    #         [1, 0, 0, 1],  # neuron 2
    #         [0, 0, 0, 0],  # neuron 3
    #     ],
    #         [
    #         [0, 0, 0, 0],  # neuron 0
    #         [0, 0, 1, 0],  # neuron 1
    #         [0, 0, 0, 1],  # neuron 2
    #         [0, 1, 0, 1],  # neuron 3
    #     ]
    #                          ])
    # }
    # weight_res = jax.vmap(process_single_batch, in_axes=(0, 0))(weight_res["activity"], weight_res["values"])

    # print(weight_res)

    # weight_res = {
    #     "activity": jnp.array([True, True, True, False]),  # Example
    #     "values": jnp.array([
    #         [0, 0, 0, 0],  # neuron 0
    #         [0, 0, 1, 0],  # neuron 1
    #         [1, 0, 0, 1],  # neuron 2
    #         [0, 0, 0, 0],  # neuron 3
    #     ])}
    # print(weight_res_complete(weight_res["activity"], weight_res["values"]))
    # ______________________________________________________________________________________________________________________________________    
    def preprocess_to_sparse_data_padded(x, max_nonzero):
        # Pre-allocate max possible
        processed_data = jnp.full((max_nonzero, 2), -1)
        
        def body_fn(i, carry):
            processed_data, j = carry
            val = x[i]
            processed_data, j = jax.lax.cond(
                val != 0,
                lambda _: (processed_data.at[j].set(jnp.array([i, val])), j + 1),
                lambda _: (processed_data, j),
                operand=None
            )
            return processed_data, j

        init_val = (processed_data, 0)
        processed_data, _ = jax.lax.fori_loop(0, x.shape[0], body_fn, init_val)
        return processed_data
    
    x = jnp.zeros((3, 10))
    rows = jnp.array([0, 0, 0, 1, 2])
    columns = jnp.array([0, 3, 7, 8, 0])
    values = jnp.array([10, 10, 10, 10, 10])
    x = x.at[rows, columns].set(values)
    
    max_nonzero = jnp.max(jnp.array([jnp.count_nonzero(row) for row in x]))
    # print(max_nonzero)
    # print(x)
    # p_data = jax.vmap(lambda x: preprocess_to_sparse_data_padded(x, max_nonzero))(x)
    # print((p_data))
    # first_values = p_data[:, 0]
    # a = first_values[:, 0]
    # b = first_values[:, 1]
    # print(a, b)
    
    # ______________________________________________________________________________________________________________________________________    
    def clone_neuron_state(template: Neuron_states, batch_size: int) -> Neuron_states:
        def tile(x):
            return jnp.broadcast_to(x, (batch_size,) + x.shape)

        return Neuron_states(
            values=tile(template.values),
            threshold=jnp.broadcast_to(template.threshold, (batch_size,)),
            input_residuals=tile(template.input_residuals),
            weight_residuals={
                k: tile(v) for k, v in template.weight_residuals.items()
            }
        )
        
    layer_sizes = [28*28, 128, 10]
    thresholds = [0, 0 ,0]  
    empty_neuron_states = Neuron_states(values=jnp.zeros((layer_sizes[rank])), 
                                        threshold=thresholds[(rank-1)%len(thresholds)], 
                                        input_residuals=jnp.zeros((layer_sizes[rank-1], 1)),
                                        weight_residuals={"activity": jnp.zeros((layer_sizes[rank-1], 1), dtype=bool), 
                                                          "values": jnp.zeros((layer_sizes[rank-1], layer_sizes[rank]))})
    
    # print(empty_neuron_states)
    # print("___________________________________")
    # print(clone_neuron_state(empty_neuron_states, 2))
    
    # ______________________________________________________________________________________________________________________________________    
    def layer_computation_batched(neuron_idx, layer_input, weights, neuron_states):    
        # activations = jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values
        
        activations = jax.lax.cond(neuron_idx==-1,
                               lambda _: (neuron_states.values).astype(jnp.float32),
                               lambda _: jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values,
                               None
                               )
        
        new_input_residuals = neuron_states.input_residuals.at[neuron_idx].add(layer_input)
        
        def last_layer_case(_):
            return jnp.zeros_like(activations), Neuron_states(values=activations, threshold=neuron_states.threshold, input_residuals=new_input_residuals, weight_residuals=neuron_states.weight_residuals)
        
        def hidden_layer_case(_):
            activated_output = activation_func(neuron_states, activations)
            
            # jax.debug.print("Rank {} weight_residuals shape: {}, neuron_idx: {}, input: {}", rank, neuron_states.weight_residuals["values"].shape, neuron_idx, layer_input)
            new_activities = neuron_states.weight_residuals["activity"].at[neuron_idx].set(True)            
            new_values = neuron_states.weight_residuals["values"].at[neuron_idx].add(jnp.where(activated_output > 0, 1, 0))
            new_weight_residuals = {"activity": new_activities, 
                                    "values": new_values}
            
            new_neuron_states = Neuron_states(values=activations - activated_output, threshold=neuron_states.threshold, input_residuals=new_input_residuals, weight_residuals=new_weight_residuals)
            return activated_output, new_neuron_states
        
        return jax.lax.cond(rank == size-1, last_layer_case, hidden_layer_case, None)
    
    # ______________________________________________________________________________________________________________________________________    

    # Simulate neuron states
    class NeuronStates:
        def __init__(self):
            self.weight_residuals = {
                "activity": jnp.array([False, True, False, True, False]),  # 0 = inactive, 1 = active
                "values": jnp.array([0, 0, 0, 0, 0])     # starting residuals 
            }

    # Initialize
    # neuron_states = NeuronStates()

    # # Suppose this is the new output of the neurons
    # activated_output = jnp.array([1.0, 2.0, 0.0, 0.0, 3.0])

    # # Indices we want to update (could be all neurons, or a subset)
    # neuron_idx = jnp.array([0, 1, 2, 3, 4])  # update all neurons

    # # Your code
    # non_activated_neurons = jnp.where(neuron_states.weight_residuals["activity"], 0, 1)
    # active_gradient = jax.numpy.logical_and(non_activated_neurons, activated_output)
    # new_values = neuron_states.weight_residuals["values"].at[neuron_idx].add(active_gradient)

    # new_layer_activities = jnp.logical_or(neuron_states.weight_residuals["activity"], activated_output)

    # # Output
    # print("Initial activity:", neuron_states.weight_residuals["activity"])
    # print("Activated output:", activated_output)
    # print("Non-activated neurons (inverted activity):", non_activated_neurons)
    # print("Active gradient (logical AND):", active_gradient)
    # print("New values after update:", new_values)
    # print("New layer activity after update:", new_layer_activities)
    
    # ______________________________________________________________________________________________________________________________________
    # Comparing multi-process gradients
    
    def load_pickle_objects(path):
        data = []
        with open(path, "rb") as f:
            while True:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break
        return data

    # Load data
    data2 = load_pickle_objects("logs/rank_2.pkl")
    data4 = load_pickle_objects("logs/rank_4.pkl")
    data5 = load_pickle_objects("logs/rank_5.pkl")

    # Sanity check: Ensure equal lengths
    assert len(data2) == len(data4) == len(data5), "Pickle logs have different lengths."

    # Compare entry by entry
    for i, (entry2, entry4, entry5) in enumerate(zip(data2, data4, data5)):
        avg_loss = (entry4["loss"] + entry5["loss"]) / 2
        avg_w_grad = (jnp.array(entry4["w_grad"]) + jnp.array(entry5["w_grad"])) / 2
        avg_out_grad = (jnp.array(entry4["out_grad"]) + jnp.array(entry5["out_grad"])) / 2

        diff_loss = entry2["loss"] - avg_loss
        diff_w_grad = jnp.linalg.norm(jnp.array(entry2["w_grad"]) - avg_w_grad)
        diff_out_grad = jnp.linalg.norm(jnp.array(entry2["out_grad"]) - avg_out_grad)

        print(f"[i={i}] Δloss: {diff_loss:.6f}, Δw_grad norm: {diff_w_grad:.6f}, Δout_grad norm: {diff_out_grad:.6f}")
