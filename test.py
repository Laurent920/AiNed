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


if __name__ == "__main__":
    if rank == 0:
        send_data()  # JAX-compiled send function

    if rank == size-1:
        data0, data1 = receive_data()  # JAX-compiled receive function
        print(f"data0: {data0}")
        print(f"data1: {data1}")
