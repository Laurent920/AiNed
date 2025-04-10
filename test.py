from mpi4py import MPI
import jax
import jax.numpy as jnp
import mpi4jax

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
    return jnp.dot(input, weight)

# Forward pass: returns output and residuals needed for backward
def test_custom_vjp_fwd(input, weight):
    if rank == 0:
        input_prev = input
        output = test_custom_vjp(input, weight)
        token = mpi4jax.send(output, dest=rank+1, tag=1, comm=comm)
    else:
        input_prev, token = mpi4jax.recv(jnp.zeros(()), source=rank-1, tag=1, comm=comm)
        output = test_custom_vjp(input_prev, weight)
    
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

def loss_fn(input, weight, target):
    output = test_custom_vjp(input, weight)
    jax.debug.print("Rank {}, output: {}", rank, output)
    l = jnp.mean((output - target) ** 2)
    jax.debug.print("loss: {}", l)
    return l

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

if __name__ == "__main__":
    input = 1.0
    target = 1.0
    weight = 0.5
    lr = 0.001

    for i in range(10):
        if rank == size-1:
            # Differentiate w.r.t. weight
            gradient = jax.grad(loss_fn, argnums=(0, 1))(input, weight, target)
            print("Gradient wrt input and weight:", gradient)
            weight -= gradient[1] * lr
            token = mpi4jax.send(jnp.array(gradient[0]), dest=rank-1, tag=0, comm=comm)
        else:
            gradient = other_layers(input, weight)    
            print("Gradient wrt input and weight:", gradient)
            weight -= gradient[1] * lr


    # if rank == 0:
    #     send_data()  # JAX-compiled send function

    # if rank == size-1:
    #     data0, data1 = receive_data()  # JAX-compiled receive function
    #     print(f"data0: {data0}")
    #     print(f"data1: {data1}")
