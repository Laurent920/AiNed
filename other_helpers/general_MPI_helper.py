import dataclasses
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from mpi4py import MPI
from mpi4jax import send, recv
import mpi4jax

from other_helpers.event_pooling import pool_output_size

try:
    from other_helpers.helpers import NeuronStates, BaseParams
except ModuleNotFoundError:
    from helpers import NeuronStates, BaseParams


@dataclasses.dataclass(frozen=True)
class MPIHelperParams(BaseParams):
    """Minimal params subclass used inside general_MPI_helper (split_batch, gather_w_it_th)."""
    pass

#region PAD_BATCH
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

@dataclass(frozen=True, init=False)
class CNN_layer_Partition(Partition):
    c_start_idx: int
    c_end_idx: int
    c_total_size: int
    
    x_start_idx: int
    x_end_idx: int
    x_total_size: int
    
    y_start_idx: int
    y_end_idx: int
    y_total_size: int

    def __init__(self,
                 c_start_idx: int,
                 c_end_idx: int,
                 c_total_size: int,
                 x_start_idx: int,
                 x_end_idx: int,
                 x_total_size: int,
                 y_start_idx: int,
                 y_end_idx: int,
                 y_total_size: int):
        total_size = c_total_size * x_total_size * y_total_size
        start_idx = self._flatten_index(c_start_idx, x_start_idx, y_start_idx, x_total_size, y_total_size)
        end_idx = self._flatten_index(c_end_idx, x_end_idx, y_end_idx, x_total_size, y_total_size)

        super().__init__(start_idx=start_idx, end_idx=end_idx, total_size=total_size)
        object.__setattr__(self, "c_start_idx", c_start_idx)
        object.__setattr__(self, "c_end_idx", c_end_idx)
        object.__setattr__(self, "c_total_size", c_total_size)
        object.__setattr__(self, "x_start_idx", x_start_idx)
        object.__setattr__(self, "x_end_idx", x_end_idx)
        object.__setattr__(self, "x_total_size", x_total_size)
        object.__setattr__(self, "y_start_idx", y_start_idx)
        object.__setattr__(self, "y_end_idx", y_end_idx)
        object.__setattr__(self, "y_total_size", y_total_size)

    @staticmethod
    def _flatten_index(c_idx: int, x_idx: int, y_idx: int, x_total_size: int, y_total_size: int):
        return c_idx * x_total_size * y_total_size + x_idx * y_total_size + y_idx

    @property
    def get_c_size(self):
        return self.c_end_idx - self.c_start_idx+1

    @property
    def get_x_size(self):
        return self.x_end_idx - self.x_start_idx+1
    
    @property
    def get_y_size(self):
        return self.y_end_idx - self.y_start_idx+1

    @property
    def get_size(self):
        return self.get_c_size * self.get_x_size * self.get_y_size

    def contain(self, data):
            coords = jnp.atleast_1d(data)
            c_idx = coords[0]
            x_idx = coords[1]
            y_idx = coords[2]

            in_c = jnp.logical_and(c_idx >= self.c_start_idx, c_idx <= self.c_end_idx)
            in_x = jnp.logical_and(x_idx >= self.x_start_idx, x_idx <= self.x_end_idx)
            in_y = jnp.logical_and(y_idx >= self.y_start_idx, y_idx <= self.y_end_idx)
            return jnp.logical_and(in_c, jnp.logical_and(in_x, in_y))
            
    def __hash__(self):
        # Custom hash using the immutable fields
        return hash((
            self.c_start_idx, self.c_end_idx, self.c_total_size,
            self.x_start_idx, self.x_end_idx, self.x_total_size,
            self.y_start_idx, self.y_end_idx, self.y_total_size,
        ))
    
    def print(self, name: str = "Partition", indent: int = 0):
        prefix = " " * indent
        print(f"{prefix}{name}:")
        print(f"{prefix}  Channels: [{self.c_start_idx}:{self.c_end_idx + 1})")
        print(f"{prefix}  X range: [{self.x_start_idx}:{self.x_end_idx + 1})")
        print(f"{prefix}  Y range: [{self.y_start_idx}:{self.y_end_idx + 1})")
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

    #region Send and Rcv
    def send_labels(self, labels, source):
        # Use mpi4py directly (not mpi4jax) to avoid the per-batch memory leak.
        if self.rank == source:
            buf = np.ascontiguousarray(np.asarray(labels, dtype=np.float32))
            self.comm.Send(buf, dest=self.batch_first_and_last_rank[1], tag=10)

    def recv_labels(self):
        if self.rank == self.batch_first_and_last_rank[1]:
            buf = np.empty((self.batch_part.get_size,), dtype=np.float32)
            self.comm.Recv(buf, source=self.batch_first_and_last_rank[0], tag=10)
            y = jnp.asarray(buf)
        else:
            y = jnp.zeros((self.batch_part.get_size,))

        y = self.leader_share_to_whole_layer(y)
        return y
    
    def forward_send(self, data, it=0):
        for process, partition in self.next_layer:
            send(data, dest=process, tag=0, comm=self.comm)

    def forward_send_cnn(self, data, event_c, event_x, event_y, k_h, k_w, event_pad_h, event_pad_w, it=0):
        """
        Selective CNN event routing: only send to next-layer ranks whose owned output
        region is touched by the kernel window of this input event.

        An event at input (c_in, x_in, y_in) updates output positions
          x_out ∈ [x_in, x_in + k_h - 1]  (padded coords)
          y_out ∈ [y_in, y_in + k_w - 1]  (padded coords)
        across all output channels (output-channel splits always receive every event).

        A next-layer rank with CNN_layer_Partition owns padded output range
          x ∈ [x_start + event_pad_h, x_end + event_pad_h]
          y ∈ [y_start + event_pad_w, y_end + event_pad_w]

        Two ranges [a, b] and [c, d] overlap iff a <= d and c <= b.
        """
        for process, partition in self.next_layer:
            if not isinstance(partition, CNN_layer_Partition):
                # FC layer or no partition info — always send
                send(data, dest=process, tag=0, comm=self.comm)
                continue

            # Kernel window in padded output coordinates
            win_x_lo = event_x
            win_x_hi = event_x + k_h - 1
            win_y_lo = event_y
            win_y_hi = event_y + k_w - 1

            # Rank's owned range in padded output coordinates
            rank_x_lo = partition.x_start_idx + event_pad_h
            rank_x_hi = partition.x_end_idx   + event_pad_h
            rank_y_lo = partition.y_start_idx + event_pad_w
            rank_y_hi = partition.y_end_idx   + event_pad_w

            x_overlap = jnp.logical_and(win_x_lo <= rank_x_hi, rank_x_lo <= win_x_hi)
            y_overlap = jnp.logical_and(win_y_lo <= rank_y_hi, rank_y_lo <= win_y_hi)
            affected  = jnp.logical_and(x_overlap, y_overlap)

            def _do_send(_):
                send(data, dest=process, tag=0, comm=self.comm)
                return None

            jax.lax.cond(affected, _do_send, lambda _: None, operand=None)

    def residual_send(self, data, event_c, event_x, event_y, it=0, broadcast=False):
        """Send event as a residual (identity skip) to every layer in self.res_connect_next.
        The channel index is negated to mark the event as residual at the receiver:
        out[0] = -event_c - 1.   Spatial coords (x, y) and value are unchanged.
        For identity skip there is no kernel window — overlap test is a single-point check.

        broadcast=True disables that test and sends to every destination rank. Required for
        weight-shared (projected) skips, where the event is remapped into the destination’s
        input frame and spread over a kernel window, so its source coordinates say nothing
        about which destination rank is affected. The receiver masks to its own slice.
        """
        neg_c = (-event_c - 1).astype(data.dtype)
        payload = data.at[0].set(neg_c)

        for process, _src_part, dst_partition in self.res_connect_next:
            if broadcast or not isinstance(dst_partition, CNN_layer_Partition):
                send(payload, dest=process, tag=0, comm=self.comm)
                continue

            in_c = jnp.logical_and(event_c >= dst_partition.c_start_idx, event_c <= dst_partition.c_end_idx)
            in_x = jnp.logical_and(event_x >= dst_partition.x_start_idx, event_x <= dst_partition.x_end_idx)
            in_y = jnp.logical_and(event_y >= dst_partition.y_start_idx, event_y <= dst_partition.y_end_idx)
            affected = jnp.logical_and(in_c, jnp.logical_and(in_x, in_y))

            def _do_send(_):
                send(payload, dest=process, tag=0, comm=self.comm)
                return None
            jax.lax.cond(affected, _do_send, lambda _: None, operand=None)

    def residual_send_end_signal(self, END_SIGNAL):
        """Send END_SIGNAL = [-1,-1,-1,-1] on tag=0 to every dst in res_connect_next.
        Distinguishable from residual events because END has x = -1 while residuals have x >= 0."""
        for process, _src_part, _dst_part in self.res_connect_next:
            send(END_SIGNAL, dest=process, tag=0, comm=self.comm)

    def forward_recv(self, input_shape):
        src = MPI.ANY_SOURCE
        data = recv(jnp.zeros((input_shape,)), source=src, tag=0, comm=self.comm)

        # jax.debug.print("rank {}, received: {}", self.rank, data)
        return data

    def forward_send_bulk(self, array):
        """Send a full (N, 4) event array in one MPI message to every next-layer rank."""
        for process, _ in self.next_layer:
            send(array, dest=process, tag=0, comm=self.comm)

    def forward_recv_bulk(self, bulk_size, cols=4):
        """Receive a (bulk_size, cols) event array in one MPI message.
        cols=4 for CNN (c,x,y,value), cols=2 for MLP (neuron_idx, value)."""
        return recv(jnp.zeros((bulk_size, cols)), source=MPI.ANY_SOURCE, tag=0, comm=self.comm)

    def backward_send(self, data):
        for process, partition in self.previous_layer:
            # cond = jnp.logical_and(self.rank != -1, jnp.any(data!=-1))
            # jax.lax.cond(cond,
            #              lambda _: jax.debug.print("rank {} sending {} to rank {}", self.rank, data.shape, process),
            #              lambda _: None, None)
            # jax.debug.print("rank {} sending start idx{} end idx{} resulting in shape {} to rank {} with total shape {}", self.rank, partition.start_idx, partition.end_idx, data[:, partition.start_idx:partition.end_idx+1].shape, process, data.shape)
            data_part = data[:, partition.start_idx:partition.end_idx+1]
            send(data_part, dest=process, tag=2, comm=self.comm)

    def backward_recv(self):
        data_shape = (self.batch_part.get_size, self.model_part.get_size)
        # data_shape = (self.batch_part.get_size, self.model_part.total_size)
        data = jnp.zeros(data_shape)
        for process, model_partition in self.next_layer:
            # jax.debug.print("rank {} waiting to receive from {} data of shape {}", self.rank, process, data_shape)
            data_part = recv(jnp.zeros(data_shape), source=process, tag=2, comm=self.comm)
            data += data_part
            # jax.debug.print("rank {} waiting to receive from {} data of shape {}", self.rank, process, data_part)

            # cond = jnp.logical_and(process != -1, jnp.any(data!=-1))
            # jax.lax.cond(cond,
            #              lambda _: jax.debug.print("rank {} received {}", self.rank, data.shape),
            #              lambda _: None, None)
        return data
    
    #region combine_batch_avg
    def combine_batch_avg(self, data):
        '''
        Concatenate all the data from one split_rank onto one rank to reconstruct the batch and 
        resharing the averaged result to the corresponding split_ranks
        '''
        rank = self.rank
        comm = self.comm
        batch_distrib = self.batch_distribution

        data = jnp.array(data)   
        if len(batch_distrib) == 1:
            # jax.debug.print("rank {} and data shape {} in combine batch", rank, data.shape)
            return jnp.mean(data, axis=0)
        
        # jax.debug.print("rank {} data shape {}", rank, data.shape)
        if self.is_batch_leader:
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
            send(data, dest=self.get_batch_leader, tag=20, comm=comm)
            avg = recv(jnp.zeros((data.shape[1:])), source=self.get_batch_leader, tag=20, comm=comm)
        # jax.debug.print(f"Rank {rank} finished combining batch avg shape: {avg.shape}")
        return avg

    #region gather_batch
    def gather_batch(self, data, average=True):
        '''
        Gather all the data from one split_rank onto one rank and resharing the average result to the corresponding split_ranks
        '''
        rank = self.rank
        comm = self.comm
        batch_distrib = self.batch_distribution

        data = jnp.array(data)
        if len(batch_distrib) == 1:
            return data
        
        if self.is_batch_leader:
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
            send(data, dest=self.get_batch_leader, tag=20, comm=comm)
            avg = recv(data, source=self.get_batch_leader, tag=20, comm=comm)
        return avg

    #region sum_model_parallel
    def sum_model_parallel(self, data):
        '''
        Sum `data` across all model-parallel ranks in the current layer that share the same
        batch partition (i.e. the ranks in self.current_layer), and broadcast the sum back.
        No-op when there is a single rank in the layer.

        Use this for gradients that each rank computed as a partial sum over its owned
        output slice (e.g. conv weight gradient under x/y splits, threshold gradient).
        '''
        rank = self.rank
        comm = self.comm
        cur_layer = self.current_layer

        data = jnp.array(data)
        if len(cur_layer) == 1:
            return data

        leader = self.get_current_group_leader
        if rank == leader:
            total = data
            for r, _ in cur_layer:
                if r == rank:
                    continue
                received = recv(data, source=r, tag=22, comm=comm)
                total = total + received
            for r, _ in cur_layer:
                if r == rank:
                    continue
                send(total, dest=r, tag=22, comm=comm)
        else:
            send(data, dest=leader, tag=22, comm=comm)
            total = recv(data, source=leader, tag=22, comm=comm)
        return total

    #region split_batch
    def split_batch(self, params, batch_iterator, tuple_size, label_shape=None, label_pad_value=-1.0):
        # tuple_size =2 for MLP and =4 for CNN 
        rank = self.rank
        comm = self.comm
        batch_part = self.batch_part
        batch_size = batch_part.total_size
        batch_distrib = self.batch_distribution

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

            # Input-layer model parallelism: model-split siblings share the same (full) batch
            # partition and so are absent from batch_distrib. Send each of them the full batch
            # (they emit only their own spatial region). No-op when the input is not split.
            _distrib_ranks = {p for p, _ in batch_distrib}
            for process, _ in self.current_layer:
                if process == rank or process in _distrib_ranks:
                    continue
                send(all_batch_x[batch_part.start_idx:batch_part.end_idx+1], dest=process, tag=4, comm=comm)
                send(all_batch_y[batch_part.start_idx:batch_part.end_idx+1], dest=process, tag=4, comm=comm)
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

    #region gather_model_part
    def gather_model_partition(self, data):
        """
        Gather all the model partitions of the current layer and reconstruct the full layer data and share the full data to all 
        processes of the current layer.
        E.g. last layer with 2 processes and 10 neurons: Gather partitions [0-4] and [5-9] to reconstruct the full layer [0-9]
        """

        batch_size = self.batch_part.get_size
        full_layer_data = jnp.zeros((batch_size, self.current_layer[0][1].total_size))
        leader = self.get_current_group_leader
        rank = self.rank
        # print(f"rank {rank} has full layer data shape {full_layer_data.shape} and data shape {data.shape}")
        if rank == leader: 
            for i, (process, partition) in enumerate(self.current_layer):
                start, end = partition.start_idx, partition.end_idx+1
                
                if process != rank: 
                    rcv_data = recv(jnp.zeros((batch_size, self.current_layer[i][1].get_size)), source=process, tag=21, comm=self.comm)
                else:
                    rcv_data = data 
                # print(f"rank {rank} received shape {rcv_data.shape}, second dim {self.current_layer[i][1].get_size}")
                full_layer_data = full_layer_data.at[:, start:end].set(rcv_data)
            for i, (process, partition) in enumerate(self.current_layer):
                if process == rank: continue
                send(full_layer_data, dest=process, tag=21, comm=self.comm)
        else:
            send(data, dest=leader, tag=21, comm=self.comm)
            full_layer_data = recv(full_layer_data, source=leader, tag=21, comm=self.comm)
        return full_layer_data

    #region concat_model_part
    def concatenate_model_partition(self, data, dim):
        """
        Concatenate all the data from the model partitions of the current layer and share the resulting data to all 
        processes of the current layer within the same batch partition.
        E.g. hidden layer with 2 processes and weights of shape (128, 5): Reconstruct the full weights as (128, 10)
        """
        full_data_shape = data.shape
        if dim == 1:
            full_data_shape = self.current_layer[0][1].total_size
        elif dim == 2:
            full_data_shape = (data.shape[0], self.current_layer[0][1].total_size)
        full_layer_data = jnp.zeros(full_data_shape)
        
        leader = self.get_current_group_leader
        rank = self.rank
        # print(f"rank {rank} has full layer data shape {full_layer_data.shape} and data shape {data.shape}")
        if rank == leader: 
            for i, (process, partition) in enumerate(self.current_layer):
                start, end = partition.start_idx, partition.end_idx+1
                
                if process != rank: 
                    if dim == 2:
                        rcv_data = recv(jnp.zeros((data.shape[0], self.current_layer[i][1].get_size)), source=process, tag=21, comm=self.comm)
                    elif dim == 1:
                        rcv_data = recv(jnp.zeros((self.current_layer[i][1].get_size,)), source=process, tag=21, comm=self.comm)
                else:
                    rcv_data = data 
                # print(f"rank {rank} received shape {rcv_data.shape}, second dim {self.current_layer[i][1].get_size}")
                if dim == 2:
                    full_layer_data = full_layer_data.at[:, start:end].set(rcv_data)
                elif dim == 1:
                    full_layer_data = full_layer_data.at[start:end].set(rcv_data)

            for i, (process, partition) in enumerate(self.current_layer):
                if process == rank: continue
                send(full_layer_data, dest=process, tag=21, comm=self.comm)
        else:
            send(data, dest=leader, tag=21, comm=self.comm)
            full_layer_data = recv(full_layer_data, source=leader, tag=21, comm=self.comm)
        return full_layer_data

    #region leader_share
    def leader_share_to_whole_layer(self, data):
        # Use plain mpi4py (not mpi4jax) to avoid the per-batch memory leak from
        # mpi4jax's per-callsite cache when called once per batch with the same shape.
        leader = self.get_current_group_leader
        rank = self.rank

        if rank == leader:
            buf = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
            for process, _ in self.current_layer:
                if process != rank:
                    self.comm.Send(buf, dest=process, tag=21)
        else:
            buf = np.empty(np.asarray(data).shape, dtype=np.float32)
            self.comm.Recv(buf, source=leader, tag=21)
        return jnp.asarray(buf)

    #region share_iteration
    def share_iteration_to_whole_layer(self, data):
        full_layer_data = jnp.zeros(data.shape)
        leader = self.get_current_group_leader
        rank = self.rank
        # print(f"rank {rank} has data {data} and data shape {data.shape}")
        if rank == leader: 
            for i, (process, partition) in enumerate(self.current_layer):
                start, end = partition.start_idx, partition.end_idx+1
                
                if process != rank: 
                    rcv_data = recv(full_layer_data, source=process, tag=21, comm=self.comm)
                else:
                    rcv_data = data 
                # print(f"rank {rank} received shape {rcv_data.shape}, second dim {self.current_layer[i][1].get_size}")
                full_layer_data += rcv_data
            for i, (process, partition) in enumerate(self.current_layer):
                if process == rank: continue
                send(full_layer_data, dest=process, tag=21, comm=self.comm)
        else:
            send(jnp.array(data), dest=leader, tag=21, comm=self.comm)
            full_layer_data = recv(full_layer_data, source=leader, tag=21, comm=self.comm)
        return full_layer_data

    #region gather w, it, th
    def gather_w_it_th(self, params, weights, mean_iterations, thresholds):
        """ 
        Gather all the weights, iteration values and thresholds at the last layer's leader rank to store them
        """
        rank = self.rank
        layer_idx = self.layer_idx
        last_layer = self.last_layer_idx
        comm = self.comm
        weights_dict = {}
        all_iteration_mean = []
        thresholds_dict = {}    

        if layer_idx == 0:
            mean_iterations = self.share_iteration_to_whole_layer(mean_iterations)
        weights = self.concatenate_model_partition(weights, dim=len(weights.shape))
        thresholds = self.concatenate_model_partition(thresholds, dim=len(thresholds.shape))

        # print(rank, thresholds.shape, mean_iterations)
        if not self.is_last_layer and self.is_batch_leader:
            dest = self.get_last_layer_batch_leader

            # print(f"rank {rank}, iterations: {mean_iterations}")
            send(jnp.array(mean_iterations), dest=dest, tag=5,comm=comm)
            if layer_idx != 0:
                send(weights, dest=dest, tag=5,comm=comm)
                send(thresholds, dest=dest, tag=5,comm=comm)

        elif self.is_last_layer and self.is_batch_leader:
            for i, leader_rank in enumerate(self.all_leader_ranks):
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

    #region MPI_partition
    def MPI_partition(self, weights, empty_neuron_states):
        start = self.model_part.start_idx
        end = self.model_part.end_idx+1
        svec = empty_neuron_states.sync_rate_vector
        # Preserve optional fields used by the CNN script (absent in the MLP script's states)
        extra_fields = {}
        for field in ("weights_shape", "is_conv"):
            if field in empty_neuron_states._fields:
                extra_fields[field] = getattr(empty_neuron_states, field)
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
            **extra_fields,
        )
        # print(f"weights shape: {weights.shape}, resulting size s-e: {start}-{end}")
        try:
            part_weights = weights[:,start:end]
        except:
            part_weights = weights[start:end]
        return part_weights, part_empty_neuron_states

    def print(self):
        """Print a compact one-line summary of the configuration."""
        if isinstance(self.model_part, CNN_layer_Partition):
            model_part_repr = (
                f"C[{self.model_part.c_start_idx}:{self.model_part.c_end_idx}] "
                f"X[{self.model_part.x_start_idx}:{self.model_part.x_end_idx}] "
                f"Y[{self.model_part.y_start_idx}:{self.model_part.y_end_idx}]"
            )
        else:
            model_part_repr = f"[{self.model_part.start_idx}:{self.model_part.end_idx}]"

        print(f"Rank {self.rank}: Layer {self.layer_idx} \n| "
              f"Batch [{self.batch_part.start_idx}:{self.batch_part.end_idx}] \n| "
              f"Model {model_part_repr} \n| "
              f"Current layer {list(self.current_layer)} \n| "
              f"Previous layer {list(self.previous_layer)} \n| "
              f"Next layer {list(self.next_layer)} \n| "
              f"Residual prev {list(self.res_connect_prev)} \n| "
              f"Residual next {list(self.res_connect_next)} \n| "
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

#region MPIProcessDistribution
class MPIProcessDistribution:
    """
    Builder for creating flexible MPI topologies for 
    - Model parallelism (splitting each layer across processes)
    - Data parallelism (splitting batches across processes)
    - ResNet Connections
    """
    def __init__(self, mpi_size: int, batch_size: int, layer_sizes: tuple[int, ...], residual_connections=()):
            assert mpi_size >= len(layer_sizes), f"The number of processes ({mpi_size}) needs to at least match the number of layers({len(layer_sizes)})"

            self.nb_layers = len(layer_sizes)
            self.mpi_size = mpi_size
            self.layer_sizes = layer_sizes
            self.batch_size = batch_size
            self.layer_assignments = {}  # rank -> (layer_idx, bach_part, model_part)
            self.residual_connections = tuple(tuple(p) for p in residual_connections)

    def _get_batch_partition_bounds(self, batch_partition_idx: int, total_batch_partitions: int):
        batch_part_size = self.batch_size // total_batch_partitions
        batch_remainder = self.batch_size % total_batch_partitions

        current_batch_size = batch_part_size + int(batch_partition_idx < batch_remainder)
        batch_part_start = batch_partition_idx * batch_part_size + min(batch_partition_idx, batch_remainder)
        batch_part_end = batch_part_start + current_batch_size - 1
        return batch_part_start, batch_part_end

    def _get_cnn_layer_shapes(self):
        cached_shapes = getattr(self, "_cnn_layer_shapes", None)
        if cached_shapes is not None:
            return cached_shapes

        cnn_layer_shapes = []
        flat_layer_shapes = []   # layer outputs, i.e. after pooling
        previous_layer = None
        for layer_idx, raw_layer in enumerate(self.layer_sizes):
            layer = tuple(raw_layer)

            if len(layer) == 1:
                previous_layer = int(layer[0])
                cnn_layer_shapes.append(previous_layer)
                flat_layer_shapes.append(previous_layer)
                continue

            if layer_idx == 0:
                previous_layer = tuple(int(dim) for dim in layer[:3])
                cnn_layer_shapes.append(previous_layer)
                flat_layer_shapes.append(previous_layer)
                continue

            if not isinstance(previous_layer, tuple) or len(previous_layer) != 3:
                raise ValueError(f"CNN layer {layer_idx} expects a 3D input shape, got {previous_layer}")

            pool_size = (2, 2)
            pool_stride = (2, 2)
            pooling = ""
            if len(layer) > 4:
                pooling = layer[4]
                if len(layer) > 5:
                    pool_size = tuple(int(dim) for dim in layer[5])
                if len(layer) > 6:
                    pool_stride = tuple(int(dim) for dim in layer[6])

            out_chan = int(layer[0])
            kernel = tuple(int(dim) for dim in layer[1])
            padding = tuple(int(dim) for dim in layer[2])
            stride = tuple(int(dim) for dim in layer[3])

            h_out = (previous_layer[1] + 2 * padding[0] - kernel[0]) // stride[0] + 1
            w_out = (previous_layer[2] + 2 * padding[1] - kernel[1]) // stride[1] + 1
            conv_shape = (out_chan, h_out, w_out)
            cnn_layer_shapes.append(conv_shape)

            if pooling != "":
                h_out = pool_output_size(h_out, pool_size[0], pool_stride[0])
                w_out = pool_output_size(w_out, pool_size[1], pool_stride[1])
            previous_layer = (out_chan, h_out, w_out)
            flat_layer_shapes.append(previous_layer)

        self._cnn_layer_shapes = tuple(cnn_layer_shapes)
        self._cnn_flat_layer_shapes = tuple(flat_layer_shapes)
        return self._cnn_layer_shapes

    def _get_cnn_flat_layer_shapes(self):
        """Layer output shapes (post-pool). _get_cnn_layer_shapes returns pre-pool shapes;
        the two differ exactly on layers that pool."""
        self._get_cnn_layer_shapes()   # populates both caches
        return self._cnn_flat_layer_shapes

    def _get_cnn_assignment(self, layer_idx: int):
        layer_shape = self._get_cnn_layer_shapes()[layer_idx]
        if isinstance(layer_shape, tuple):
            return (
                (0, layer_shape[0] - 1),
                (0, layer_shape[1] - 1),
                (0, layer_shape[2] - 1),
            )
        return (0, layer_shape - 1)

    def _build_cnn_partition(self, layer_idx: int, model_parts, flat=False):
        shapes = self._get_cnn_flat_layer_shapes() if flat else self._get_cnn_layer_shapes()
        layer_shape = shapes[layer_idx]
        if isinstance(layer_shape, tuple):
            (c_start, c_end), (x_start, x_end), (y_start, y_end) = model_parts
            return CNN_layer_Partition(
                c_start_idx=c_start,
                c_end_idx=c_end,
                c_total_size=layer_shape[0],
                x_start_idx=x_start,
                x_end_idx=x_end,
                x_total_size=layer_shape[1],
                y_start_idx=y_start,
                y_end_idx=y_end,
                y_total_size=layer_shape[2],
            )
        return Partition(
            start_idx=model_parts[0],
            end_idx=model_parts[1],
            total_size=layer_shape,
        )

    def _build_residual_src_partition(self, src_idx: int, model_parts):
        """Partition of a residual source layer, in the space the residual actually travels in.

        A residual carries the src layer's OUTPUT (post-pool), so both the forward identity
        add and the backward gradient live in flat space. model_parts, however, is indexed in
        pre-pool space. The two coincide only when src does not pool -- which is why the only
        previously-run residual ([1,2], layer 1 unpooled) worked and a pooled src crashed.
        """
        conv_shape = self._get_cnn_layer_shapes()[src_idx]
        flat_shape = self._get_cnn_flat_layer_shapes()[src_idx]
        if not isinstance(flat_shape, tuple) or conv_shape == flat_shape:
            return self._build_cnn_partition(src_idx, model_parts)

        # src pools: a pre-pool shard cannot be mapped onto the pooled output in general,
        # so only a rank owning the whole layer (data-parallel only) can be re-expressed.
        if tuple(model_parts) != tuple(self._get_cnn_assignment(src_idx)):
            raise ValueError(
                f"Residual source layer {src_idx} both pools and is model-split "
                f"(model_parts={model_parts}); mapping a pre-pool shard onto the pooled "
                f"output is not supported. Use split_dims=null for this layer."
            )
        flat_parts = ((0, flat_shape[0] - 1), (0, flat_shape[1] - 1), (0, flat_shape[2] - 1))
        return self._build_cnn_partition(src_idx, flat_parts, flat=True)

    # region CNN data uniform
    def CNN_data_split_uniform(self):
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
        self._get_cnn_layer_shapes()

        for rank in range(self.mpi_size):
            batch_partition_idx = rank // self.nb_layers
            layer_idx = rank % self.nb_layers
            batch_part_start, batch_part_end = self._get_batch_partition_bounds(batch_partition_idx, process_per_layer)
            self.layer_assignments[rank] = (
                layer_idx,
                (batch_part_start, batch_part_end),
                self._get_cnn_assignment(layer_idx),
            )
    
    #region CNN Model custom
    def CNN_model_split_custom(self, processes_per_layer: tuple[int, ...],
                               split_dims: tuple[str, ...] | None = None):
        """
        General CNN model parallelism: split each layer across any combination of
        output channel (c), height (x), and width (y) dimensions.

        Args:
            processes_per_layer: tuple of ints (one per layer) summing to mpi_size.
                For data-parallel layers use 1; for model-parallel layers use > 1.
            split_dims: tuple of strings (one per layer) choosing the split dimension.
                Each element is one of 'c', 'x', 'y', 'cx', 'cy', 'xy', 'cxy', or None.
                None/'c' defaults to splitting on c (channel).
                FC layers always split on their single neuron dimension regardless.

        Example: processes_per_layer=(1, 4, 2, 1), split_dims=(None, 'cx', 'x', None)
            layer 0: 1 process, full shape
            layer 1: 4 processes split across c and x (2 c-splits × 2 x-splits)
            layer 2: 2 processes split across x
            layer 3: 1 process, full shape
        """
        assert len(processes_per_layer) == self.nb_layers, \
            f"processes_per_layer length ({len(processes_per_layer)}) must match number of layers ({self.nb_layers})"
        assert sum(processes_per_layer) == self.mpi_size, \
            f"sum of processes_per_layer ({sum(processes_per_layer)}) must equal total MPI size ({self.mpi_size})"
        assert all(p >= 1 for p in processes_per_layer), "each layer needs at least 1 process"
        if split_dims is None:
            split_dims = tuple(['c'] * self.nb_layers)
        assert len(split_dims) == self.nb_layers

        cnn_layer_shapes = self._get_cnn_layer_shapes()
        process_rank = 0

        for layer_idx, nb_processes in enumerate(processes_per_layer):
            layer_shape = cnn_layer_shapes[layer_idx]
            dim = split_dims[layer_idx] or 'x'

            if not isinstance(layer_shape, tuple):
                # FC layer: split on neuron dimension
                layer_neurons = layer_shape
                part_size = layer_neurons // nb_processes
                remain = layer_neurons % nb_processes
                start = 0
                end = start + part_size - 1
                for _ in range(nb_processes):
                    self.layer_assignments[process_rank] = (
                        layer_idx,
                        (0, self.batch_size - 1),
                        (start, end),
                    )
                    start = end + 1
                    end = start + part_size - 1
                    if remain != 0:
                        end += 1
                        remain -= 1
                    process_rank += 1
                continue

            c_total, x_total, y_total = layer_shape

            # Determine how many splits per active dimension
            active_dims = [d for d in ('c', 'x', 'y') if d in dim]
            nb_active = len(active_dims)
            assert nb_active >= 1, f"split_dims entry '{dim}' must contain at least one of c/x/y"

            # Distribute nb_processes across the active dims as evenly as possible.
            # E.g. 4 processes with 'cx' → 2 c-splits × 2 x-splits.
            # Distribute greedily: largest-remainder among active dims.
            splits_per_dim = {d: 1 for d in ('c', 'x', 'y')}
            remaining = nb_processes
            for i, d in enumerate(active_dims):
                dim_size = {'c': c_total, 'x': x_total, 'y': y_total}[d]
                share = round(remaining ** (1.0 / (nb_active - i)))
                share = min(share, dim_size, remaining)
                splits_per_dim[d] = max(1, share)
                remaining = (remaining + splits_per_dim[d] - 1) // splits_per_dim[d]

            # Adjust so total product equals nb_processes
            # (simple approach: recompute last dim from the product of others)
            product_others = 1
            for d in active_dims[:-1]:
                product_others *= splits_per_dim[d]
            splits_per_dim[active_dims[-1]] = nb_processes // product_others

            nc = splits_per_dim['c']
            nx = splits_per_dim['x']
            ny = splits_per_dim['y']
            assert nc * nx * ny == nb_processes, \
                f"Layer {layer_idx}: {nc}×{nx}×{ny}={nc*nx*ny} ≠ {nb_processes}. " \
                f"Choose processes_per_layer divisible by the active split dimensions."

            def _ranges(total, n):
                part = total // n
                rem = total % n
                start = 0
                ranges = []
                for i in range(n):
                    end = start + part - 1
                    if i < rem:
                        end += 1
                    ranges.append((start, end))
                    start = end + 1
                return ranges

            c_ranges = _ranges(c_total, nc)
            x_ranges = _ranges(x_total, nx)
            y_ranges = _ranges(y_total, ny)

            for c_s, c_e in c_ranges:
                for x_s, x_e in x_ranges:
                    for y_s, y_e in y_ranges:
                        model_parts = (
                            (c_s, c_e),
                            (x_s, x_e),
                            (y_s, y_e),
                        )
                        self.layer_assignments[process_rank] = (
                            layer_idx,
                            (0, self.batch_size - 1),
                            model_parts,
                        )
                        process_rank += 1

        print(self.layer_assignments)

    #region MLP Data uniform
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

    #region MLP Model custom
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

    #region MLP Model uniform
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

    #region MPI Build
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

        # Populate residual connections
        res_prev_list = []  # ranks in src layers that point AT this layer (this rank receives from them)
        res_next_list = []  # ranks in dst layers that this rank's layer points AT (this rank sends to them)

        for src_idx, dst_idx in self.residual_connections:
            # If this rank is in the destination layer, every rank in the source layer that shares
            # this batch partition becomes a residual previous-layer entry.
            if layer_idx == dst_idx:
                for r_other in self.layer_assignments.keys():
                    l_other, b_other, m_other = self.layer_assignments[r_other]
                    if l_other != src_idx:
                        continue
                    if not (b_other[0] == batch_part.start_idx and b_other[1] == batch_part.end_idx):
                        continue
                    src_part = self._build_residual_src_partition(src_idx, m_other)
                    dst_part = self._build_cnn_partition(dst_idx, model_parts)
                    res_prev_list.append((r_other, src_part, dst_part))

            if layer_idx == src_idx:
                for r_other in self.layer_assignments.keys():
                    l_other, b_other, m_other = self.layer_assignments[r_other]
                    if l_other != dst_idx:
                        continue
                    if not (b_other[0] == batch_part.start_idx and b_other[1] == batch_part.end_idx):
                        continue
                    src_part = self._build_residual_src_partition(src_idx, model_parts)
                    dst_part = self._build_cnn_partition(dst_idx, m_other)
                    res_next_list.append((r_other, src_part, dst_part))

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
            nb_previous=len(prev) + len(res_prev_list),
            all_leader_ranks=all_leaders_rank,
            batch_distribution=tuple(batch_distrib),
            batch_first_and_last_rank=batch_first_and_last_rank,
            res_connect_next=tuple(res_next_list),
            res_connect_prev=tuple(res_prev_list)
        )
    
    def CNN_build(self, rank, comm):
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
        model_part = self._build_cnn_partition(layer_idx, model_parts)
        

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
                m_part = self._build_cnn_partition(l_idx, m_parts)
                if m_part.start_idx == 0: # Only consider the first model partition of the layer (leader ranks)
                    b_last_rank = max(b_last_rank, r)   # Last rank of the batch partition where to send the labels
                    b_first_rank = min(b_first_rank, r) # First rank of the batch partition from which we receive the labels
                
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

        # Populate residual connections
        res_prev_list = []  # ranks in src layers that point AT this layer (this rank receives from them)
        res_next_list = []  # ranks in dst layers that this rank's layer points AT (this rank sends to them)

        for src_idx, dst_idx in self.residual_connections:
            # If this rank is in the destination layer, every rank in the source layer that shares
            # this batch partition becomes a residual previous-layer entry.
            if layer_idx == dst_idx:
                for r_other in self.layer_assignments.keys():
                    l_other, b_other, m_other = self.layer_assignments[r_other]
                    if l_other != src_idx:
                        continue
                    if not (b_other[0] == batch_part.start_idx and b_other[1] == batch_part.end_idx):
                        continue
                    src_part = self._build_residual_src_partition(src_idx, m_other)
                    dst_part = self._build_cnn_partition(dst_idx, model_parts)
                    res_prev_list.append((r_other, src_part, dst_part))

            if layer_idx == src_idx:
                for r_other in self.layer_assignments.keys():
                    l_other, b_other, m_other = self.layer_assignments[r_other]
                    if l_other != dst_idx:
                        continue
                    if not (b_other[0] == batch_part.start_idx and b_other[1] == batch_part.end_idx):
                        continue
                    src_part = self._build_residual_src_partition(src_idx, model_parts)
                    dst_part = self._build_cnn_partition(dst_idx, m_other)
                    res_next_list.append((r_other, src_part, dst_part))

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
            nb_previous=len(prev) + len(res_prev_list),
            all_leader_ranks=all_leaders_rank,
            batch_distribution=tuple(batch_distrib),
            batch_first_and_last_rank=batch_first_and_last_rank,
            res_connect_next=tuple(res_next_list),
            res_connect_prev=tuple(res_prev_list)
        )

#region Splitting func
def data_split(rank, comm, mpi_size, batch_size, layer_sizes: tuple[int, ...]):
    mpi_process_distribution = MPIProcessDistribution(mpi_size, batch_size, layer_sizes)
    mpi_process_distribution.data_split_uniform()
    mpi_config = mpi_process_distribution.build(rank, comm)
    return mpi_config

def CNN_data_split(rank, comm, mpi_size, batch_size, layer_sizes: tuple[int, ...]):
    mpi_process_distribution = MPIProcessDistribution(mpi_size, batch_size, layer_sizes)
    mpi_process_distribution.CNN_data_split_uniform()
    mpi_config = mpi_process_distribution.CNN_build(rank, comm)
    return mpi_config

def model_split_custom(rank, comm, mpi_size, batch_size, layer_sizes: tuple[int, ...], processes_per_layer: tuple[int, ...]):
    mpi_process_distribution = MPIProcessDistribution(mpi_size, batch_size, layer_sizes)
    mpi_process_distribution.model_split_custom(processes_per_layer)
    mpi_config = mpi_process_distribution.build(rank, comm)
    return mpi_config

def CNN_model_split_custom_x(rank, comm, mpi_size, batch_size, layer_sizes: tuple[int, ...], processes_per_layer: tuple[int, ...]):
    mpi_process_distribution = MPIProcessDistribution(mpi_size, batch_size, layer_sizes)
    mpi_process_distribution.CNN_model_split_custom_x(processes_per_layer)
    mpi_config = mpi_process_distribution.CNN_build(rank, comm)
    return mpi_config

def CNN_model_split_custom(rank, comm, mpi_size, batch_size, layer_sizes: tuple[int, ...],
                           processes_per_layer: tuple[int, ...],
                           split_dims: tuple[str, ...] | None = None):
    """General CNN model parallelism supporting splits on channel (c), x, y, or combinations."""
    mpi_process_distribution = MPIProcessDistribution(mpi_size, batch_size, layer_sizes)
    mpi_process_distribution.CNN_model_split_custom(processes_per_layer, split_dims)
    mpi_config = mpi_process_distribution.CNN_build(rank, comm)
    return mpi_config

def model_split(rank, comm, mpi_size, batch_size, layer_sizes: tuple[int, ...]):
    mpi_process_distribution = MPIProcessDistribution(mpi_size, batch_size, layer_sizes)
    mpi_process_distribution.model_split_uniform()
    mpi_config = mpi_process_distribution.build(rank, comm)
    return mpi_config

def ResNet_data_split(rank, comm, mpi_size, batch_size, layer_sizes, residual_connections=()):
    d = MPIProcessDistribution(mpi_size, batch_size, layer_sizes, residual_connections)
    d.CNN_data_split_uniform()
    return d.CNN_build(rank, comm)

def ResNet_model_split_custom(rank, comm, mpi_size, batch_size, layer_sizes,
                              processes_per_layer, split_dims=None, residual_connections=()):
    d = MPIProcessDistribution(mpi_size, batch_size, layer_sizes, residual_connections)
    d.CNN_model_split_custom(processes_per_layer, split_dims)
    return d.CNN_build(rank, comm)

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
