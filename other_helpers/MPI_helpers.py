from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp

from mpi4py import MPI
from mpi4jax import send, recv

@dataclass(frozen=True)  # Makes it immutable and hashable
class MPIConfig:
    rank: int
    split_rank: int
    last_rank: int
    process_per_layer: int
    batch_part: int
    comm: object
    
    def __hash__(self):
        # Custom hash that excludes non-hashable fields like comm
        return hash((self.rank, self.split_rank, self.process_per_layer))
    
    def __eq__(self, other):
        if not isinstance(other, MPIConfig):
            return False
        return (self.rank == other.rank and 
                self.split_rank == other.split_rank and 
                self.process_per_layer == other.process_per_layer)
    
@partial(jax.jit, static_argnames=['mpi_config',])
def combine_batch_avg(data, mpi_config):
    '''
    Concatenate all the data from one split_rank onto one rank to reconstruct the batch and 
    resharing the averaged result to the corresponding split_ranks
    '''
    rank = mpi_config.rank
    split_rank = mpi_config.split_rank
    process_per_layer = mpi_config.process_per_layer
    comm = mpi_config.comm

    data = jnp.array(data)        
    # jax.debug.print("rank {} data shape {}", rank, data.shape)

    leader_rank = split_rank * process_per_layer
    if rank == leader_rank:
        avg = data
        for i in range(0, process_per_layer-1): # Receive the data from all the corresponding ranks in one split rank
            received_data = recv(data, source=rank+i+1, tag=20, comm=comm)
            avg = jnp.concatenate([avg, received_data], axis=0)            
        # print(f"Rank {rank} before combining batches, avg shape: {avg.shape}")
        avg = jnp.mean(avg, axis=0)

        for i in range(process_per_layer-1): # Resharing the average data to all the corresponding ranks
            send(avg, dest=rank+i+1, tag=20, comm=comm)
    else:
        send(data, dest=leader_rank, tag=20, comm=comm)
        avg = recv(jnp.zeros((data.shape[1:])), source=leader_rank, tag=20, comm=comm)      
    return avg

# @partial(jax.jit, static_argnames=['mpi_config',])
def gather_batch(data, mpi_config, average=True):
    '''
    Gather all the data from one split_rank onto one rank and resharing the average result to the corresonding split_ranks
    '''
    rank = mpi_config.rank
    split_rank = mpi_config.split_rank
    process_per_layer = mpi_config.process_per_layer
    comm = mpi_config.comm

    data = jnp.array(data)
    leader_rank = split_rank * process_per_layer
    if rank == leader_rank:
        avg = data
        for i in range(process_per_layer-1): # Receive the data from all the corresponding ranks in one split rank
            received_data = recv(data, source=rank+i+1, tag=20, comm=comm)
            avg += received_data
        if average:
            avg = avg / process_per_layer
        
        for i in range(process_per_layer-1): # Resharing the average data to all the corresponding ranks
            send(avg, dest=rank+i+1, tag=20, comm=comm)
    else:
        send(data, dest=leader_rank, tag=20, comm=comm)
        avg = recv(data, source=leader_rank, tag=20, comm=comm)
    return avg

def pad_batch(batch_x, batch_y, batch_size):
    # Pad the x data with 0 and the y data with nan for the last batch
    current_size = batch_y.shape[0]
    if current_size < batch_size:
        pad_amount = batch_size - current_size
        pad_y = jnp.full((pad_amount,), -1.0, dtype=jnp.float32)
        pad_x = jnp.zeros((pad_amount,) + batch_x.shape[1:], dtype=batch_x.dtype)  
        # jax.debug.print("rank {}, has batch size: {} and pad batch size: {}", rank, current_size, pad_x.shape)
        batch_y = jnp.concatenate([batch_y, pad_y], axis=0)
        batch_x = jnp.concatenate([batch_x, pad_x], axis=0)
    
    return batch_x, batch_y

def split_batch(params, batch_iterator, mpi_config, tuple_size):
    # tuple_size =2 for MLP and =4 for CNN 
    rank = mpi_config.rank
    process_per_layer = mpi_config.process_per_layer
    comm = mpi_config.comm
    batch_part = mpi_config.batch_part

    if rank == 0:
        all_batch_x, all_batch_y = next(batch_iterator)
        # print(all_batch_x)
        all_batch_y = jnp.array(all_batch_y, dtype=jnp.float32)
        all_batch_x = jnp.array(all_batch_x, dtype=jnp.float32)
        # print('shape before pad batch: {}', all_batch_x.shape)
        all_batch_x, all_batch_y = pad_batch(all_batch_x, all_batch_y, batch_part* process_per_layer)
        
        for process in range(process_per_layer):
            if process == 0:
                batch_x = all_batch_x[:batch_part]
                batch_y = all_batch_y[:batch_part]
            else:
                batch_x_to_send = all_batch_x[batch_part*(process):batch_part*(process+1)]
                batch_y_to_send = all_batch_y[batch_part*(process):batch_part*(process+1)]
                # print(f"rank {rank}, Batch_x: {batch_x_to_send.shape}, Batch_y: {batch_y_to_send.shape}")
                
                send(batch_x_to_send, dest=process, tag=4, comm=comm)
                send(batch_y_to_send, dest=process, tag=4, comm=comm)
    else:
        batch_x = recv(jnp.zeros((batch_part, params.max_nonzero, tuple_size)), source=0, tag=4, comm=comm)  
        batch_y = recv(jnp.zeros((batch_part,)), source=0, tag=4, comm=comm) 
    
    return batch_x, batch_y

def l2_weight_regularization(mpi_config, weights):
    rank = mpi_config.rank
    split_rank = mpi_config.split_rank
    last_rank = mpi_config.last_rank
    process_per_layer = mpi_config.process_per_layer
    comm = mpi_config.comm
    leader_rank = split_rank * process_per_layer
    
    weights_sum = jnp.sum(weights**2)

    if split_rank != last_rank and rank == leader_rank:
        if split_rank != 0:
            send(weights_sum, dest=last_rank * process_per_layer, tag=7,comm=comm)
    elif split_rank == last_rank and rank == leader_rank:
        for i in range(1, last_rank):
            # Storing mean iterations
            sum = recv(weights_sum, source=i * process_per_layer, tag=7, comm=comm)
            weights_sum += sum
            
    return weights_sum