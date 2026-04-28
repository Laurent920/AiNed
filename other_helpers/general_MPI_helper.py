from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from mpi4py import MPI
from mpi4jax import send, recv
import mpi4jax

try:
    from other_helpers.helpers import NeuronStates
except ModuleNotFoundError:
    from helpers import NeuronStates

#region combine_batch_avg
@partial(jax.jit, static_argnames=['mpi_config',])
def combine_batch_avg(data, mpi_config):
    '''
    Concatenate all the data from one split_rank onto one rank to reconstruct the batch and 
    resharing the averaged result to the corresponding split_ranks
    '''
    rank = mpi_config.rank
    comm = mpi_config.comm
    batch_distrib = mpi_config.batch_distribution

    data = jnp.array(data)   
    if len(batch_distrib) == 1:
        return data.squeeze(axis=0)
    # jax.debug.print("rank {} data shape {}", rank, data.shape)
    if mpi_config.is_batch_leader:
        avg = data
        for r, _ in batch_distrib: # Receive the data from all the corresponding ranks in one split rank
            if r == rank: continue
            received_data = recv(data, source=r, tag=20, comm=comm)
            avg = jnp.concatenate([avg, received_data], axis=0)            
        # print(f"Rank {rank} before combining batches, avg shape: {avg.shape}")
        avg = jnp.mean(avg, axis=0)

        for r, _ in batch_distrib: # Resharing the average data to all the corresponding ranks
            if r == rank: continue
            send(avg, dest=r, tag=20, comm=comm)
    else:
        send(data, dest=mpi_config.get_batch_leader, tag=20, comm=comm)
        avg = recv(jnp.zeros((data.shape[1:])), source=mpi_config.get_batch_leader, tag=20, comm=comm)
    # jax.debug.print(f"Rank {rank} finished combining batch avg shape: {avg.shape}")
    return avg

#region gather_batch
# @partial(jax.jit, static_argnames=['mpi_config',])
def gather_batch(data, mpi_config, average=True):
    '''
    Gather all the data from one split_rank onto one rank and resharing the average result to the corresponding split_ranks
    '''
    rank = mpi_config.rank
    comm = mpi_config.comm
    batch_distrib = mpi_config.batch_distribution

    data = jnp.array(data)
    if len(batch_distrib) == 1:
        return data
    
    if mpi_config.is_batch_leader:
        avg = data
        for r, _ in batch_distrib: # Receive the data from all the corresponding ranks in one split rank
            if r == rank: continue
            received_data = recv(data, source=r, tag=20, comm=comm)
            avg += received_data
        if average:
            avg = avg / len(batch_distrib)
        
        for r, _ in batch_distrib: # Resharing the average data to all the corresponding ranks
            if r == rank: continue
            send(avg, dest=r, tag=20, comm=comm)
    else:
        send(data, dest=mpi_config.get_batch_leader, tag=20, comm=comm)
        avg = recv(data, source=mpi_config.get_batch_leader, tag=20, comm=comm)
    return avg

def pad_batch(batch_x, batch_y, batch_size, label_pad_value=-1.0):
    """
    Pad the last batch to a fixed size.
    Works for both scalar labels (classification) and vector labels (regression).
    """
    current_size = batch_y.shape[0]
    if current_size < batch_size:
        pad_amount = batch_size - current_size
        label_tail_shape = batch_y.shape[1:] if batch_y.ndim > 1 else ()
        pad_y = jnp.full((pad_amount,) + label_tail_shape, label_pad_value, dtype=batch_y.dtype)
        pad_x = jnp.zeros((pad_amount,) + batch_x.shape[1:], dtype=batch_x.dtype)
        # jax.debug.print("rank {}, has batch size: {} and pad batch size: {}", rank, current_size, pad_x.shape)
        batch_y = jnp.concatenate([batch_y, pad_y], axis=0)
        batch_x = jnp.concatenate([batch_x, pad_x], axis=0)
    
    return batch_x, batch_y

#region split_batch
def split_batch(params, batch_iterator, mpi_config, tuple_size, label_shape=None, label_pad_value=-1.0):
    # tuple_size =2 for MLP and =4 for CNN 
    rank = mpi_config.rank
    comm = mpi_config.comm
    batch_part = mpi_config.batch_part
    batch_size = batch_part.total_size
    batch_distrib = mpi_config.batch_distribution

    # jax.debug.print("rank {} batch distrib {}", rank, batch_distrib)

    if rank == 0:
        all_batch_x, all_batch_y = next(batch_iterator)
        # print(rank, all_batch_x.shape)
        all_batch_y = jnp.array(all_batch_y, dtype=jnp.float32)
        all_batch_x = jnp.array(all_batch_x, dtype=jnp.float32)
        # print('shape before pad batch: {}', all_batch_x[0])
        all_batch_x, all_batch_y = pad_batch(
            all_batch_x, all_batch_y, batch_size, label_pad_value=label_pad_value
        )
        
        for process, b_partition in batch_distrib:
            # print("rank in split batch:", rank, process, b_partition)
            if process == 0:
                batch_x = all_batch_x[:b_partition.end_idx+1]
                batch_y = all_batch_y[:b_partition.end_idx+1]
            else:
                batch_x_to_send = all_batch_x[b_partition.start_idx:b_partition.end_idx+1]
                batch_y_to_send = all_batch_y[b_partition.start_idx:b_partition.end_idx+1]
                # print(f"rank {rank}, Batch_x: {batch_x_to_send.shape}, Batch_y: {batch_y_to_send.shape}")
                
                send(batch_x_to_send, dest=process, tag=4, comm=comm)
                send(batch_y_to_send, dest=process, tag=4, comm=comm)
    else:
        # print(f"rank {rank} waiting for shape {(batch_part.get_size, params.max_nonzero, tuple_size)}")
        batch_x = recv(jnp.zeros((batch_part.get_size, params.max_nonzero, tuple_size)), source=0, tag=4, comm=comm)  
        if label_shape is None:
            label_shape = ()
        elif isinstance(label_shape, int):
            label_shape = (label_shape,)
        batch_y = recv(jnp.zeros((batch_part.get_size,) + tuple(label_shape), dtype=jnp.float32), source=0, tag=4, comm=comm) 
    # jax.debug.print("rank {} batch y {}", rank, batch_y)
    # print(f'rank {rank} finished splitting batch')
    return batch_x, batch_y

def gather_model_partition(mpi_config, data):
    """
    Gather all the model partitions of the current layer and reconstruct the full layer data and share the full data to all 
    processes of the current layer.
    E.g. last layer with 2 processes and 10 neurons: Gather partitions [0-4] and [5-9] to reconstruct the full layer [0-9]
    """

    batch_size = mpi_config.batch_part.get_size
    full_layer_data = jnp.zeros((batch_size, mpi_config.current_layer[0][1].total_size))
    leader = mpi_config.get_current_group_leader
    rank = mpi_config.rank
    # print(f"rank {rank} has full layer data shape {full_layer_data.shape} and data shape {data.shape}")
    if rank == leader: 
        for i, (process, partition) in enumerate(mpi_config.current_layer):
            start, end = partition.start_idx, partition.end_idx+1
            
            if process != rank: 
                rcv_data = recv(jnp.zeros((batch_size, mpi_config.current_layer[i][1].get_size)), source=process, tag=21, comm=mpi_config.comm)
            else:
                rcv_data = data 
            # print(f"rank {rank} received shape {rcv_data.shape}, second dim {mpi_config.current_layer[i][1].get_size}")
            full_layer_data = full_layer_data.at[:, start:end].set(rcv_data)
        for i, (process, partition) in enumerate(mpi_config.current_layer):
            if process == rank: continue
            send(full_layer_data, dest=process, tag=21, comm=mpi_config.comm)
    else:
        send(data, dest=leader, tag=21, comm=mpi_config.comm)
        full_layer_data = recv(full_layer_data, source=leader, tag=21, comm=mpi_config.comm)
    return full_layer_data

def concatenate_model_partition(mpi_config, data, dim):
    """
    Concatenate all the data from the model partitions of the current layer and share the resulting data to all 
    processes of the current layer within the same batch partition.
    E.g. hidden layer with 2 processes and weights of shape (128, 5): Reconstruct the full weights as (128, 10)
    """
    full_data_shape = data.shape
    if dim == 1:
        full_data_shape = mpi_config.current_layer[0][1].total_size
    elif dim == 2:
        full_data_shape = (data.shape[0], mpi_config.current_layer[0][1].total_size)
    full_layer_data = jnp.zeros(full_data_shape)
    
    leader = mpi_config.get_current_group_leader
    rank = mpi_config.rank
    # print(f"rank {rank} has full layer data shape {full_layer_data.shape} and data shape {data.shape}")
    if rank == leader: 
        for i, (process, partition) in enumerate(mpi_config.current_layer):
            start, end = partition.start_idx, partition.end_idx+1
            
            if process != rank: 
                if dim == 2:
                    rcv_data = recv(jnp.zeros((data.shape[0], mpi_config.current_layer[i][1].get_size)), source=process, tag=21, comm=mpi_config.comm)
                elif dim == 1:
                    rcv_data = recv(jnp.zeros((mpi_config.current_layer[i][1].get_size,)), source=process, tag=21, comm=mpi_config.comm)
            else:
                rcv_data = data 
            # print(f"rank {rank} received shape {rcv_data.shape}, second dim {mpi_config.current_layer[i][1].get_size}")
            if dim == 2:
                full_layer_data = full_layer_data.at[:, start:end].set(rcv_data)
            elif dim == 1:
                full_layer_data = full_layer_data.at[start:end].set(rcv_data)

        for i, (process, partition) in enumerate(mpi_config.current_layer):
            if process == rank: continue
            send(full_layer_data, dest=process, tag=21, comm=mpi_config.comm)
    else:
        send(data, dest=leader, tag=21, comm=mpi_config.comm)
        full_layer_data = recv(full_layer_data, source=leader, tag=21, comm=mpi_config.comm)
    return full_layer_data

def leader_share_to_whole_layer(mpi_config, data):
    leader = mpi_config.get_current_group_leader
    rank = mpi_config.rank

    if rank == leader:
        for i, (process, partition) in enumerate(mpi_config.current_layer):
            if process != rank: 
                send(data, dest=process, tag=21, comm=mpi_config.comm)
    else:
        data = recv(data, source=leader, tag=21, comm=mpi_config.comm)
    return data

def share_iteration_to_whole_layer(mpi_config, data):
    full_layer_data = jnp.zeros(data.shape)
    leader = mpi_config.get_current_group_leader
    rank = mpi_config.rank
    # print(f"rank {rank} has data {data} and data shape {data.shape}")
    if rank == leader: 
        for i, (process, partition) in enumerate(mpi_config.current_layer):
            start, end = partition.start_idx, partition.end_idx+1
            
            if process != rank: 
                rcv_data = recv(full_layer_data, source=process, tag=21, comm=mpi_config.comm)
            else:
                rcv_data = data 
            # print(f"rank {rank} received shape {rcv_data.shape}, second dim {mpi_config.current_layer[i][1].get_size}")
            full_layer_data += rcv_data
        for i, (process, partition) in enumerate(mpi_config.current_layer):
            if process == rank: continue
            send(full_layer_data, dest=process, tag=21, comm=mpi_config.comm)
    else:
        send(jnp.array(data), dest=leader, tag=21, comm=mpi_config.comm)
        full_layer_data = recv(full_layer_data, source=leader, tag=21, comm=mpi_config.comm)
    return full_layer_data

def gather_w_it_th(mpi_config, params, weights, mean_iterations, thresholds):
    """ 
    Gather all the weights, iteration values and thresholds at the last layer's leader rank to store them
    """
    rank = mpi_config.rank
    layer_idx = mpi_config.layer_idx
    last_layer = mpi_config.last_layer_idx
    comm = mpi_config.comm
    weights_dict = {}
    all_iteration_mean = []
    thresholds_dict = {}    

    if layer_idx == 0:
        mean_iterations = share_iteration_to_whole_layer(mpi_config, mean_iterations)
    weights = concatenate_model_partition(mpi_config, weights, dim=len(weights.shape))
    thresholds = concatenate_model_partition(mpi_config, thresholds, dim=len(thresholds.shape))

    # print(rank, thresholds.shape, mean_iterations)
    if not mpi_config.is_last_layer and mpi_config.is_batch_leader:
        dest = mpi_config.get_last_layer_batch_leader

        # print(f"rank {rank}, iterations: {mean_iterations}")
        send(jnp.array(mean_iterations), dest=dest, tag=5,comm=comm)
        if layer_idx != 0:
            send(weights, dest=dest, tag=5,comm=comm)
            send(thresholds, dest=dest, tag=5,comm=comm)

    elif mpi_config.is_last_layer and mpi_config.is_batch_leader:
        for i, leader_rank in enumerate(mpi_config.all_leader_ranks):
            if rank == leader_rank: continue
            # Storing mean iterations
            it_mean = recv(mean_iterations, source=leader_rank, tag=5, comm=comm)
            all_iteration_mean.append(it_mean)
            if leader_rank==0: 
                continue

            # Storing the weights 
            w = recv(jnp.zeros((params.layer_sizes[i-1], params.layer_sizes[i])), source=leader_rank, tag=5, comm=comm)   
            weights_dict[f"layer_{i}"] = w.tolist()
            
            # Storing the thresholds
            thr = recv(jnp.zeros(params.layer_sizes[i]), source=leader_rank, tag=5, comm=comm)
            thresholds_dict[f"thresholds_{i}"]= thr.tolist()
            
        all_iteration_mean.append(mean_iterations)  # Append the mean iterations of the last layer
        weights_dict[f"layer_{last_layer}"] = weights.tolist()

        print("all iteration mean: rank", rank, all_iteration_mean)

    return weights_dict, all_iteration_mean, thresholds_dict

#region Partition
@dataclass(frozen=True)
class Partition:
    start_idx: int
    end_idx: int
    total_size: int

    @property
    def get_size(self):
        return self.end_idx - self.start_idx+1

    def contain(self, data):
            index = jnp.atleast_1d(data)[0] # jax.debug.print("data shape {} data: {}, first element of data {}", data.shape, data, data.at[0].get())
            return jnp.logical_and(index >= self.start_idx, index <= self.end_idx)
            
    def __hash__(self):
        # Custom hash using the immutable fields
        return hash((self.start_idx, self.end_idx, self.total_size))
    
    def print(self, name: str = "Partition", indent: int = 0):
        prefix = " " * indent
        print(f"{prefix}{name}:")
        print(f"{prefix}  Range: [{self.start_idx}:{self.end_idx})")
        print(f"{prefix}  Size: {self.get_size}/{self.total_size}")
        print(f"{prefix}  Coverage: {(self.get_size/self.total_size)*100:.1f}%")

#region MPIConfig
@dataclass(frozen=True)  # Makes it immutable and hashable
class MPIConfig:
    rank: int
    size: int
    comm: object

    layer_idx: int # = split_rank
    last_layer_idx: int

    batch_part: Partition
    model_part: Partition # process_per_layer: int

    previous_layer: tuple[tuple[int, Partition]]    # rank, model_part
    current_layer: tuple[tuple[int, Partition]]     # rank, model_part
    next_layer: tuple[tuple[int, Partition]]        # rank, model_part
    nb_previous: int

    all_leader_ranks: tuple[tuple[int, int]]        # layer, rank

    batch_distribution: tuple[tuple[int, Partition]]# rank, batch_part
    batch_first_and_last_rank: tuple[int, int]

    res_connect_prev: tuple[tuple[int, Partition, Partition]]
    res_connect_next: tuple[tuple[int, Partition, Partition]]

    def __hash__(self):
        # Hash only hashable fields (exclude comm)
        return hash((
            self.rank,
            self.size,
            self.layer_idx,
            self.last_layer_idx,
            self.batch_part,
            self.model_part,
            self.previous_layer,
            self.current_layer,
            self.next_layer,
            self.nb_previous,
            self.all_leader_ranks,
            self.batch_distribution,
            self.batch_first_and_last_rank,
            self.res_connect_prev,
            self.res_connect_next
        ))

    @property
    def get_previous_layer_ranks(self):
        return tuple([rank for rank, _ in self.previous_layer])
    
    @property
    def get_current_layer_ranks(self):
        return tuple([rank for rank, _ in self.current_layer])
    
    @property
    def get_next_layer_ranks(self):
        return tuple([rank for rank, _ in self.next_layer])
    
    @property
    def get_previous_group_leader(self):
        return min(self.get_previous_layer_ranks)

    @property
    def get_current_group_leader(self):
        return min(self.get_current_layer_ranks)

    @property
    def get_next_group_leader(self):
        return min(self.get_next_layer_ranks)

    @property
    def is_last_layer(self):
        return self.last_layer_idx == self.layer_idx
    
    @property
    def is_batch_leader(self):
        return self.rank in self.all_leader_ranks
    
    @property
    def is_last_layer_leader(self):
        return self.is_last_layer and self.is_batch_leader
    
    @property
    def get_batch_leader(self):
        return self.all_leader_ranks[self.layer_idx]

    @property
    def get_last_layer_batch_leader(self):
        return max(self.all_leader_ranks)

    @property
    def get_process_per_batch(self):
        return len(self.batch_distribution)
    
    @property
    def get_process_per_layer(self):
        return len(self.current_layer)

    def MPI_partition(self, weights, empty_neuron_states):
        start = self.model_part.start_idx
        end = self.model_part.end_idx+1
        svec = empty_neuron_states.sync_rate_vector
        part_empty_neuron_states = NeuronStates(
            values=empty_neuron_states.values[start:end],
            bias=empty_neuron_states.bias[start:end],
            thresholds=empty_neuron_states.thresholds[start:end],
            input_residuals=empty_neuron_states.input_residuals,
            input_order=empty_neuron_states.input_order,
            input_activity=empty_neuron_states.input_activity,
            layer_activity=empty_neuron_states.layer_activity[start:end],
            output_activity=empty_neuron_states.output_activity[:, start:end],
            last_sent_iteration=empty_neuron_states.last_sent_iteration[start:end],
            input_vector=empty_neuron_states.input_vector,
            output_vector=empty_neuron_states.output_vector[start:end],
            sync_rate_vector=svec[start:end] if svec is not None else None,
            values_history=empty_neuron_states.values_history[:, start:end],
            history_index=jnp.array(0, dtype=jnp.int32),
        )
        # print(f"weights shape: {weights.shape}, resulting size s-e: {start}-{end}")
        try:
            part_weights = weights[:,start:end]
        except:
            part_weights = weights[start:end]
        return part_weights, part_empty_neuron_states

    def print(self):
        """Print a compact one-line summary of the configuration."""
        print(f"Rank {self.rank}: Layer {self.layer_idx} \n| "
              f"Batch [{self.batch_part.start_idx}:{self.batch_part.end_idx}] \n| "
              f"Model [{self.model_part.start_idx}:{self.model_part.end_idx}] \n| "
              f"Current layer {list(self.current_layer)} \n| "
              f"Previous layer {list(self.previous_layer)} \n| "
              f"Next layer {list(self.next_layer)} \n| " 
              f"all_leader_ranks {list(self.all_leader_ranks)} \n| "
              f"Batch distribution {list(self.batch_distribution)} \n| "
              f"batch_first_and_last_rank {list(self.batch_first_and_last_rank)}")
        
# Register MPIConfig as a PyTree with all static data
def _mpiconfig_flatten(config):
    """Flatten: return (children, aux_data)
    Since everything is static, we have no children."""
    return ((), config)

def _mpiconfig_unflatten(aux_data, children):
    """Unflatten: reconstruct from aux_data and children
    Since we have no children, just return the config."""
    return aux_data

# Register the PyTree
jax.tree_util.register_pytree_node(
    MPIConfig,
    _mpiconfig_flatten,
    _mpiconfig_unflatten
)

#region MPI Build
class MPIProcessDistribution:
    """Builder for creating flexible MPI topologies for 
    - Model parallelism (splitting each layer across processes)
    - Data parallelism (splitting batches across processes)
    - ResNet Connections
    """
    def __init__(self, mpi_size: int, batch_size: int, layer_sizes: tuple[int, ...]):
            assert mpi_size >= len(layer_sizes), f"The number of processes ({mpi_size}) needs to at least match the number of layers({len(layer_sizes)})"

            self.nb_layers = len(layer_sizes)
            self.mpi_size = mpi_size
            self.layer_sizes = layer_sizes
            self.batch_size = batch_size
            self.layer_assignments = {}  # rank -> (layer_idx, bach_part, model_part)

    def data_split_uniform(self):
        """
        Data parallelism: requires the number of processes to be a multiple of the number of layers
        mapping scheme: 6 processes, 3 layers
            0 -> 1 -> 2
            3 -> 4 -> 5

        If the batch size is not perfectly divisible by the number of processes per layer, 
        the remaining samples are distributed one by one to the first few processes in the layer until all samples are assigned.
        """
        assert self.mpi_size % self.nb_layers == 0, f"Pure data parallelism requires the number of processes ({self.mpi_size}) to be a multiple of the number of layers ({self.nb_layers})"

        process_per_layer = self.mpi_size // self.nb_layers
        batch_remain = self.batch_size % process_per_layer
        batch_part_size = self.batch_size // process_per_layer

        layer_idx = 0
        batch_part_start = 0
        batch_part_end = batch_part_start + batch_part_size -1
        for i in range(self.mpi_size):
            self.layer_assignments[i] = (layer_idx, 
                                         (batch_part_start, batch_part_end), 
                                         (0, self.layer_sizes[layer_idx]-1))
            if layer_idx == self.nb_layers - 1: # Set up the batch partitions for the next network
                batch_part_start = batch_part_end+1
                batch_part_end = batch_part_start + batch_part_size -1
                if batch_remain != 0:
                    batch_part_end += 1
                    batch_remain -= 1

            layer_idx = (layer_idx+1)%self.nb_layers
        return self
    
    def model_split_custom(self, processes_per_layer: tuple[int, ...]):
        """
        Model parallelism with a custom number of processes per layer.
        All processes in the same layer share the same batch partition.
        Rank layout: layers are assigned in order, e.g. processes_per_layer=(1,2,1) → ranks 0|1,2|3

        Example: processes_per_layer=(1, 2, 1), batch_size=36
            rank 0 → layer 0, neurons [0:784]
            rank 1 → layer 1, neurons [0:128]
            rank 2 → layer 1, neurons [129:256]
            rank 3 → layer 2, neurons [0:10]
        """
        assert len(processes_per_layer) == self.nb_layers, \
            f"processes_per_layer length ({len(processes_per_layer)}) must match number of layers ({self.nb_layers})"
        assert sum(processes_per_layer) == self.mpi_size, \
            f"sum of processes_per_layer ({sum(processes_per_layer)}) must equal total MPI size ({self.mpi_size})"
        assert all(p >= 1 for p in processes_per_layer), "each layer needs at least 1 process"

        process_rank = 0
        for layer_idx, nb_processes in enumerate(processes_per_layer):
            layer_neurons = self.layer_sizes[layer_idx]     
            layer_part_size = layer_neurons // nb_processes
            layer_remain = layer_neurons % nb_processes

            model_part_start = 0
            model_part_end = model_part_start + layer_part_size -1
            for _ in range(nb_processes):
                self.layer_assignments[process_rank] =  (layer_idx, 
                                                        (0, self.batch_size-1), 
                                                        (model_part_start, model_part_end))
                model_part_start = model_part_end + 1
                model_part_end = model_part_start + layer_part_size -1
                if layer_remain != 0:   # Distribute the remain of the layer's neurons one by one to the first few processes in the layer until all neurons are assigned
                    model_part_end += 1
                    layer_remain -= 1

                process_rank += 1
        print(self.layer_assignments)
        return self

    def model_split_uniform(self):
        """
        Model parallelism: Splits the processes as uniformely as possible by sharing to the deeper layers first 
        (excluding the last layer because it only needs to integrate the values, less computation required)
        Mapping scheme: 
        |6 processes, 3 layers
            0&1 -> 2&3 -> 4&5
        |7 processes, 3 layers
            0&1 -> 2&3&4 -> 5&6
        |8 processes, 3 layers
            0&1&2 -> 3&4&5 -> 6&7
        """
        process_per_layer = self.mpi_size // self.nb_layers
        model_remain = self.mpi_size % self.nb_layers 
        process_distribution = [process_per_layer] * self.nb_layers
        for i in reversed(range(self.nb_layers-1)): # Distributing the additional processes to the deeper layers first (excluding the last layer)
            if model_remain == 0:
                break
            process_distribution[i] += 1
            model_remain -= 1
        

        process_rank = 0
        for layer_idx, nb_processes in enumerate(process_distribution):
            layer_neurons = self.layer_sizes[layer_idx]     
            layer_part_size = layer_neurons // nb_processes
            layer_remain = layer_neurons % nb_processes

            model_part_start = 0
            model_part_end = model_part_start + layer_part_size -1
            for _ in range(nb_processes):
                self.layer_assignments[process_rank] =  (layer_idx, 
                                                        (0, self.batch_size-1), 
                                                        (model_part_start, model_part_end))
                model_part_start = model_part_end+1
                model_part_end = model_part_start + layer_part_size -1
                if layer_remain != 0:   # Distribute the remain of the layer's neurons one by one to the first few processes in the layer until all neurons are assigned
                    model_part_end += 1
                    layer_remain -= 1

                process_rank += 1
        print(self.layer_assignments)
        return self

    def build(self, rank, comm):
        """
            Builds the partitions according to the specifications in self.layer_assignments
            - Batch partition refers to the splitting of a batch across processes (See data_split_uniform)
            - Model partition refers to the splitting of a layer across processes (See model_split_uniform)
        """
        # Creating batch and model partitions of current rank
        layer_idx, batch_parts, model_parts = self.layer_assignments[rank]
        print(rank, self.layer_assignments)
        batch_part = Partition(start_idx=batch_parts[0],
                               end_idx=batch_parts[1],
                               total_size=self.batch_size)
        model_part = Partition(start_idx=model_parts[0],
                               end_idx=model_parts[1],
                               total_size=self.layer_sizes[layer_idx])
        

        prev, curr, next = [], [], []
        batch_distrib = []
        all_leaders = []
        for _ in range(self.nb_layers):
            all_leaders.append([])

        b_first_rank, b_last_rank = self.mpi_size, 0
        for r in self.layer_assignments.keys():
            l_idx, b_parts, m_parts = self.layer_assignments[r] # Layer index, batch parts, model parts
            all_leaders[l_idx].append(r)
            
            # Check if the rank belongs to the same batch partition
            batch_cond = b_parts[0] == batch_part.start_idx and b_parts[1] == batch_part.end_idx
            if batch_cond: 
                if m_parts[0] == 0: # Only consider the first model partition of the layer (leader ranks)
                    b_last_rank = max(b_last_rank, r)   # Last rank of the batch partition where to send the labels
                    b_first_rank = min(b_first_rank, r) # First rank of the batch partition from which we receive the labels
                
                # print(rank, r, b_parts, batch_part)
                m_part = Partition(start_idx=m_parts[0],
                                    end_idx=m_parts[1],
                                    total_size=self.layer_sizes[l_idx])
                
                info = (r, m_part)
                if l_idx == layer_idx-1:    # Previous layer model parts where we receive the data from
                    prev.append(info)
                elif l_idx == layer_idx:    # Current layer model parts 
                    if rank == 0:
                        print(f"r {r}, l_idx {l_idx}, layer_index {layer_idx}")
                    curr.append(info)
                elif l_idx == layer_idx+1:  # Next layer model parts where we send the data to
                    next.append(info)
            # if l_idx == layer_idx:
            if l_idx == layer_idx and (not batch_cond or r == rank):
                b_part = Partition(start_idx=b_parts[0],
                               end_idx=b_parts[1],
                               total_size=self.batch_size)
                batch_distrib.append((r, b_part))              

        all_leaders_rank = tuple([min(layer_ranks) for layer_ranks in all_leaders])
        batch_first_and_last_rank = (b_first_rank, b_last_rank)
        return MPIConfig(
            rank=rank,
            size=self.mpi_size,
            comm=comm,
            layer_idx=layer_idx,
            last_layer_idx=self.nb_layers-1,
            batch_part=batch_part,
            model_part=model_part,
            previous_layer=tuple(prev),
            current_layer=tuple(curr),
            next_layer=tuple(next),
            nb_previous=len(prev),
            all_leader_ranks=all_leaders_rank,
            batch_distribution=tuple(batch_distrib),
            batch_first_and_last_rank=batch_first_and_last_rank,
            res_connect_next=(),
            res_connect_prev=()
        )

#region Send and Rcv
def forward_send(mpi_config: MPIConfig, data, it=0):    
    for process, partition in mpi_config.next_layer:
        # cond = jnp.logical_and(mpi_config.layer_idx == 1, jnp.any(data!=-2))
        # jax.lax.cond(cond,
        #              lambda _: jax.debug.print("rank {} sending {} to rank {}, it {}", mpi_config.rank, data, process, it),
        #              lambda _: None, None)
        send(data, dest=process, tag=0, comm=mpi_config.comm)

@partial(jax.jit, static_argnums=(1,))
def forward_recv(mpi_config: MPIConfig, input_shape):
    return recv(jnp.zeros((input_shape,)), source=MPI.ANY_SOURCE, tag=0, comm=mpi_config.comm)

def send_labels(mpi_config: MPIConfig, labels, source):
    # Use mpi4py directly (not mpi4jax) to avoid the per-batch memory leak.
    if mpi_config.rank == source:
        buf = np.ascontiguousarray(np.asarray(labels, dtype=np.float32))
        mpi_config.comm.Send(buf, dest=mpi_config.batch_first_and_last_rank[1], tag=10)

def recv_labels(mpi_config: MPIConfig):
    if mpi_config.rank == mpi_config.batch_first_and_last_rank[1]:
        buf = np.empty((mpi_config.batch_part.get_size,), dtype=np.float32)
        mpi_config.comm.Recv(buf, source=mpi_config.batch_first_and_last_rank[0], tag=10)
        y = jnp.asarray(buf)
    else:
        y = jnp.zeros((mpi_config.batch_part.get_size,))

    y = leader_share_to_whole_layer(mpi_config, y)
    return y

def backward_send(mpi_config: MPIConfig, data):
    for process, partition in mpi_config.previous_layer:
        # cond = jnp.logical_and(mpi_config.rank != -1, jnp.any(data!=-1))
        # jax.lax.cond(cond,
        #              lambda _: jax.debug.print("rank {} sending {} to rank {}", mpi_config.rank, data.shape, process),
        #              lambda _: None, None)
        # jax.debug.print("rank {} sending start idx{} end idx{} resulting in shape {} to rank {} with total shape {}", mpi_config.rank, partition.start_idx, partition.end_idx, data[:, partition.start_idx:partition.end_idx+1].shape, process, data.shape)
        data_part = data[:, partition.start_idx:partition.end_idx+1]
        send(data_part, dest=process, tag=2, comm=mpi_config.comm)

def backward_recv(mpi_config: MPIConfig):
    data_shape = (mpi_config.batch_part.get_size, mpi_config.model_part.get_size)
    # data_shape = (mpi_config.batch_part.get_size, mpi_config.model_part.total_size)
    data = jnp.zeros(data_shape)
    for process, model_partition in mpi_config.next_layer:
        # jax.debug.print("rank {} waiting to receive from {} data of shape {}", mpi_config.rank, process, data_shape)
        data_part = recv(jnp.zeros(data_shape), source=process, tag=2, comm=mpi_config.comm)
        data += data_part
        # jax.debug.print("rank {} waiting to receive from {} data of shape {}", mpi_config.rank, process, data_part)

        # cond = jnp.logical_and(process != -1, jnp.any(data!=-1))
        # jax.lax.cond(cond,
        #              lambda _: jax.debug.print("rank {} received {}", mpi_config.rank, data.shape),
        #              lambda _: None, None)
    return data

#region Splitting func
def data_split(rank, comm, mpi_size, batch_size, layer_sizes: tuple[int, ...]):
    mpi_process_distribution = MPIProcessDistribution(mpi_size, batch_size, layer_sizes)
    mpi_process_distribution.data_split_uniform()
    mpi_config = mpi_process_distribution.build(rank, comm)
    return mpi_config

def model_split_custom(rank, comm, mpi_size, batch_size, layer_sizes: tuple[int, ...], processes_per_layer: tuple[int, ...]):
    mpi_process_distribution = MPIProcessDistribution(mpi_size, batch_size, layer_sizes)
    mpi_process_distribution.model_split_custom(processes_per_layer)
    mpi_config = mpi_process_distribution.build(rank, comm)
    return mpi_config

def model_split(rank, comm, mpi_size, batch_size, layer_sizes: tuple[int, ...]):
    mpi_process_distribution = MPIProcessDistribution(mpi_size, batch_size, layer_sizes)
    mpi_process_distribution.model_split_uniform()
    mpi_config = mpi_process_distribution.build(rank, comm)
    return mpi_config

#region main
if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()      # Real rank
    size = comm.Get_size()

    batch_size = 39
    layer_sizes = (700, 128, 128, 20)

    # mpi_config = data_split(rank, comm, size, batch_size, layer_sizes)
    mpi_config = model_split(rank, comm, size, batch_size, layer_sizes)

    mpi_config.print()
