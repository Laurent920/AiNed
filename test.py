import enum
from mpi4py import MPI
import jax
import jax.numpy as jnp
import mpi4jax
import dataclasses
import tree_math
from jax import custom_jvp

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
    print(max_nonzero)
    print(x)
    p_data = jax.vmap(lambda x: preprocess_to_sparse_data_padded(x, max_nonzero))(x)
    print((p_data))
    first_values = p_data[:, 0]
    a = first_values[:, 0]
    b = first_values[:, 1]
    print(a, b)
    
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
        activations = jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values
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
    
    