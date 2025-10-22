import jax
import jax.numpy as jnp
from jax import grad, custom_vjp
from mpi4py import MPI
import mpi4jax
import numpy as np

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Network configuration
INPUT_DIM = 784
HIDDEN_DIM = 128
OUTPUT_DIM = 10
BATCH_SIZE = 36

# =============================================================================
# SIMPLE EXAMPLE: 6 Processes mapping to [784, 128, 10] network
# =============================================================================

def create_simple_mapping(num_processes=6):
    """
    Example mapping for 6 processes:
    p0: layer0, neurons[0:392]
    p1: layer0, neurons[392:784]
    p2: layer1, neurons[0:64]
    p3: layer1, neurons[64:128]
    p4: layer2, neurons[0:5]
    p5: layer2, neurons[5:10]
    """
    layer_sizes = [784, 128, 10]
    mapping = []
    
    # Layer 0: split 784 neurons across 2 processes
    mapping.append({'process': 0, 'layer': 0, 'neuron_range': (0, 392)})
    mapping.append({'process': 1, 'layer': 0, 'neuron_range': (392, 784)})
    
    # Layer 1: split 128 neurons across 2 processes
    mapping.append({'process': 2, 'layer': 1, 'neuron_range': (0, 64)})
    mapping.append({'process': 3, 'layer': 1, 'neuron_range': (64, 128)})
    
    # Layer 2: split 10 neurons across 2 processes
    mapping.append({'process': 4, 'layer': 2, 'neuron_range': (0, 5)})
    mapping.append({'process': 5, 'layer': 2, 'neuron_range': (5, 10)})
    
    return mapping, layer_sizes


def get_layer_info(mapping, my_rank):
    """Extract info for current rank"""
    for m in mapping:
        if m['process'] == my_rank:
            return m
    return None


def create_layer_communicators(mapping):
    """Create separate communicator for each layer"""
    layers = {}
    
    for m in mapping:
        layer_idx = m['layer']
        if layer_idx not in layers:
            layers[layer_idx] = []
        layers[layer_idx].append(m['process'])
    
    # Create MPI communicators for each layer
    layer_comms = {}
    for layer_idx, processes in layers.items():
        group = comm.Get_group().Incl(processes)
        layer_comms[layer_idx] = comm.Create(group)
    
    return layer_comms


# =============================================================================
# DISTRIBUTED LAYER OPERATIONS
# =============================================================================

@custom_vjp
def distributed_matmul_send(W_local, x, neuron_start, neuron_end, next_layer_processes):
    """
    Compute local matmul and send results neuron-by-neuron to next layer.
    
    Args:
        W_local: [local_neurons, input_dim] - local weight slice
        x: [batch, input_dim] - full input (replicated)
        neuron_start, neuron_end: global neuron indices for this process
        next_layer_processes: list of process ranks in next layer
    """
    # Compute local output: [local_neurons, batch]
    local_out = W_local @ x.T
    
    # Send each neuron to appropriate process in next layer
    local_neurons = neuron_end - neuron_start
    
    for local_idx in range(local_neurons):
        global_idx = neuron_start + local_idx
        neuron_value = local_out[local_idx, :]  # [batch]
        
        # Determine which next-layer process should receive this
        # For simplicity, send to all next-layer processes (they'll select what they need)
        for dest_proc in next_layer_processes:
            packet = jnp.array([float(global_idx)] + neuron_value.tolist())
            mpi4jax.send(packet, dest=dest_proc, tag=0, comm=comm)
    
    return local_out


def distributed_matmul_send_fwd(W_local, x, neuron_start, neuron_end, next_layer_processes):
    local_out = distributed_matmul_send(W_local, x, neuron_start, neuron_end, next_layer_processes)
    return local_out, (W_local, x, neuron_start, neuron_end, next_layer_processes)


def distributed_matmul_send_bwd(residuals, g):
    W_local, x, neuron_start, neuron_end, next_layer_processes = residuals
    
    # Receive gradient for local neurons
    local_neurons = neuron_end - neuron_start
    grad_local = jnp.zeros((local_neurons, x.shape[0]))
    
    for local_idx in range(local_neurons):
        global_idx = neuron_start + local_idx
        # Receive gradient from next layer processes
        grad_packet = mpi4jax.recv(
            jnp.zeros(x.shape[0] + 1),
            source=MPI.ANY_SOURCE,
            tag=1,
            comm=comm
        )
        grad_local = grad_local.at[local_idx].set(grad_packet[1:])
    
    # Compute gradients
    grad_W = grad_local @ x  # [local_neurons, input_dim]
    grad_x = W_local.T @ grad_local  # [input_dim, batch]
    
    return (grad_W, grad_x, None, None, None)


distributed_matmul_send.defvjp(distributed_matmul_send_fwd, distributed_matmul_send_bwd)


@custom_vjp
def distributed_receive_accumulate(prev_layer_size, batch_size, prev_layer_processes, W_local, neuron_start, neuron_end):
    """
    Receive neurons from previous layer and accumulate relevant contributions.
    
    This process owns neurons [neuron_start:neuron_end] of current layer.
    It receives all neurons from previous layer and computes:
    output[i] = sum_j(W[i, j] * prev_layer_output[j])
    
    But only for neurons i in [neuron_start:neuron_end].
    """
    local_neurons = neuron_end - neuron_start
    local_output = jnp.zeros((local_neurons, batch_size))
    
    # Receive neurons from previous layer
    prev_values = jnp.zeros((prev_layer_size, batch_size))
    
    for _ in range(prev_layer_size):
        for src_proc in prev_layer_processes:
            packet = mpi4jax.recv(
                jnp.zeros(batch_size + 1),
                source=src_proc,
                tag=0,
                comm=comm
            )
            prev_neuron_idx = int(packet[0])
            prev_neuron_value = packet[1:]
            prev_values = prev_values.at[prev_neuron_idx].set(prev_neuron_value)
    
    # Compute local output using local weights
    # W_local: [local_neurons, prev_layer_size]
    local_output = W_local @ prev_values  # [local_neurons, batch]
    
    return local_output


def distributed_receive_accumulate_fwd(prev_layer_size, batch_size, prev_layer_processes, W_local, neuron_start, neuron_end):
    local_output = distributed_receive_accumulate(
        prev_layer_size, batch_size, prev_layer_processes, W_local, neuron_start, neuron_end
    )
    
    # Save for backward pass
    return local_output, (prev_layer_size, batch_size, prev_layer_processes, W_local, neuron_start, neuron_end, local_output)


def distributed_receive_accumulate_bwd(residuals, g):
    prev_layer_size, batch_size, prev_layer_processes, W_local, neuron_start, neuron_end, local_output = residuals
    
    # g: [local_neurons, batch] - gradient w.r.t. local output
    
    # Gradient w.r.t. W_local
    # Need prev_layer_values which we should have saved
    # For simplicity, receive them again (in practice, save them)
    prev_values = jnp.zeros((prev_layer_size, batch_size))
    # ... receive logic similar to forward pass ...
    
    grad_W_local = g @ prev_values.T  # [local_neurons, prev_layer_size]
    
    # Gradient w.r.t. prev layer values
    grad_prev = W_local.T @ g  # [prev_layer_size, batch]
    
    # Send gradients back to previous layer processes
    for prev_neuron_idx in range(prev_layer_size):
        grad_packet = jnp.array([float(prev_neuron_idx)] + grad_prev[prev_neuron_idx].tolist())
        for dest_proc in prev_layer_processes:
            mpi4jax.send(grad_packet, dest=dest_proc, tag=1, comm=comm)
    
    return (None, None, None, grad_W_local, None, None)


distributed_receive_accumulate.defvjp(distributed_receive_accumulate_fwd, distributed_receive_accumulate_bwd)


# =============================================================================
# SIMPLIFIED EXAMPLE: Just demonstrate the concept
# =============================================================================

def simple_distributed_layer_example():
    """
    Simplified example showing key concepts without full implementation.
    
    Key insights:
    1. Each process owns a slice of neurons
    2. Use MPI subcommunicators for intra-layer operations
    3. Use all-reduce for operations that need full layer view
    """
    
    mapping, layer_sizes = create_simple_mapping(6)
    my_info = get_layer_info(mapping, rank)
    
    if my_info is None:
        print(f"Rank {rank}: Not assigned to any layer")
        return
    
    layer_idx = my_info['layer']
    neuron_start, neuron_end = my_info['neuron_range']
    local_neurons = neuron_end - neuron_start
    
    print(f"Rank {rank}: Layer {layer_idx}, neurons [{neuron_start}:{neuron_end}]")
    
    # Create layer-specific communicators
    layer_comms = create_layer_communicators(mapping)
    my_layer_comm = layer_comms[layer_idx]
    
    # Example: Initialize local weights
    if layer_idx == 0:
        input_dim = INPUT_DIM
        W_local = jax.random.normal(jax.random.PRNGKey(rank), (local_neurons, input_dim)) * 0.01
    elif layer_idx == 1:
        input_dim = layer_sizes[0]  # 784
        W_local = jax.random.normal(jax.random.PRNGKey(rank), (local_neurons, input_dim)) * 0.01
    else:  # layer_idx == 2
        input_dim = layer_sizes[1]  # 128
        W_local = jax.random.normal(jax.random.PRNGKey(rank), (local_neurons, input_dim)) * 0.01
    
    print(f"Rank {rank}: W_local shape: {W_local.shape}")
    
    # Example: Distributed top-k within layer
    # Generate random activations for demonstration
    local_activations = jax.random.normal(jax.random.PRNGKey(rank + 100), (local_neurons,))
    
    # Find top-5 neurons in this layer
    k = 5
    if layer_idx < len(layer_sizes):  # Valid layer
        # Method 1: Gather to root of layer communicator
        rank_in_layer = my_layer_comm.Get_rank()
        layer_size = my_layer_comm.Get_size()
        
        # Each process contributes its top local candidates
        local_k = min(k, local_neurons)
        local_top_idx = jnp.argsort(local_activations)[-local_k:][::-1]
        local_top_vals = local_activations[local_top_idx]
        global_top_idx = local_top_idx + neuron_start
        
        # Gather all candidates to root
        if rank_in_layer == 0:
            all_vals = np.zeros(layer_size * local_k, dtype=np.float32)
            all_idx = np.zeros(layer_size * local_k, dtype=np.int32)
        else:
            all_vals = None
            all_idx = None
        
        # Use Gather
        my_layer_comm.Gather(np.array(local_top_vals, dtype=np.float32), all_vals, root=0)
        my_layer_comm.Gather(np.array(global_top_idx, dtype=np.int32), all_idx, root=0)
        
        if rank_in_layer == 0:
            # Select global top-k
            final_k = min(k, len(all_vals))
            sorted_indices = np.argsort(all_vals)[-final_k:][::-1]
            top_k_vals = all_vals[sorted_indices]
            top_k_idx = all_idx[sorted_indices]
            
            print(f"\nRank {rank} (Layer {layer_idx} root): Top-{k} neurons globally:")
            print(f"  Indices: {top_k_idx}")
            print(f"  Values: {top_k_vals}")


# =============================================================================
# PRACTICAL EXAMPLE: Layer-wise norm computation
# =============================================================================

def distributed_layer_norm(values_local, layer_comm):
    """
    Compute layer normalization across distributed neurons.
    
    Formula: (x - mean) / sqrt(var + eps)
    Requires: all-reduce for mean and variance
    """
    local_sum = jnp.sum(values_local)
    local_count = len(values_local)
    
    # All-reduce to get global sum and count
    global_sum = mpi4jax.allreduce(local_sum, op=MPI.SUM, comm=layer_comm)[0]
    global_count = mpi4jax.allreduce(jnp.array(local_count), op=MPI.SUM, comm=layer_comm)[0]
    
    mean = global_sum / global_count
    
    # Compute variance
    local_sq_diff = jnp.sum((values_local - mean) ** 2)
    global_sq_diff = mpi4jax.allreduce(local_sq_diff, op=MPI.SUM, comm=layer_comm)[0]
    
    variance = global_sq_diff / global_count
    std = jnp.sqrt(variance + 1e-5)
    
    # Normalize local values
    normalized_local = (values_local - mean) / std
    
    return normalized_local


# =============================================================================
# COMPLETE MINI EXAMPLE: 2-layer network with 3 processes
# =============================================================================

def mini_example_3_processes():
    """
    Minimal working example with 3 processes:
    p0: layer0 neurons [0:4]
    p1: layer0 neurons [4:8]  
    p2: layer1 neurons [0:8] (all neurons, receives from p0 and p1)
    
    Network: [8, 8] (8 input, 8 hidden)
    """
    if size != 3:
        if rank == 0:
            print(f"This example requires exactly 3 processes, got {size}")
        return
    
    MINI_INPUT = 8
    MINI_HIDDEN = 8
    
    # Define mapping
    if rank == 0:
        my_layer = 0
        neuron_start, neuron_end = 0, 4
        next_layer_procs = [2]
        prev_layer_procs = []
    elif rank == 1:
        my_layer = 0
        neuron_start, neuron_end = 4, 8
        next_layer_procs = [2]
        prev_layer_procs = []
    else:  # rank == 2
        my_layer = 1
        neuron_start, neuron_end = 0, 8
        next_layer_procs = []
        prev_layer_procs = [0, 1]
    
    local_neurons = neuron_end - neuron_start
    
    print(f"Rank {rank}: Layer {my_layer}, neurons [{neuron_start}:{neuron_end}]")
    
    # Create layer communicators
    if my_layer == 0:
        layer_group = comm.Get_group().Incl([0, 1])
        layer_comm = comm.Create(layer_group)
    else:
        layer_group = comm.Get_group().Incl([2])
        layer_comm = comm.Create(layer_group)
    
    # Initialize weights
    key = jax.random.PRNGKey(rank)
    
    if my_layer == 0:
        # Input layer: no weights, just forward input
        x = jax.random.normal(key, (MINI_INPUT,))
        
        # Send neurons one by one
        for i in range(neuron_start, neuron_end):
            packet = jnp.array([float(i), x[i]])
            for dest in next_layer_procs:
                mpi4jax.send(packet, dest=dest, tag=0, comm=comm)
        
        print(f"Rank {rank}: Sent neurons {neuron_start} to {neuron_end-1}")
        
    else:  # Hidden layer
        # Receive all input neurons and compute output
        W = jax.random.normal(key, (local_neurons, MINI_INPUT)) * 0.01
        
        received_values = jnp.zeros(MINI_INPUT)
        for _ in range(MINI_INPUT):
            packet = mpi4jax.recv(jnp.zeros(2), source=MPI.ANY_SOURCE, tag=0, comm=comm)
            idx = int(packet[0])
            val = packet[1]
            received_values = received_values.at[idx].set(val)
        
        # Compute output
        output = W @ received_values
        
        print(f"Rank {rank}: Received {MINI_INPUT} neurons, computed output shape {output.shape}")
        print(f"Rank {rank}: Output mean: {jnp.mean(output):.6f}")
    
    comm.Barrier()


# =============================================================================
# BEST PRACTICES SUMMARY
# =============================================================================

def print_best_practices():
    """Print key best practices for distributed neural networks"""
    if rank == 0:
        print("\n" + "="*80)
        print("BEST PRACTICES FOR DISTRIBUTED NEURAL NETWORKS")
        print("="*80)
        print("""
1. PROCESS MAPPING STRATEGIES:
   
   a) Neuron Parallelism (Model Parallelism):
      - Split neurons across processes
      - Best for: Large layers that don't fit in single GPU
      - Pros: Can handle huge models
      - Cons: Requires all-reduce for full layer operations
      - Example: GPT-3 uses this for large linear layers
   
   b) Batch Parallelism (Data Parallelism):
      - Split batch across processes
      - Best for: Small models, large batches
      - Pros: No model communication during forward/backward
      - Cons: Only scales up to batch_size processes
      - Example: ImageNet training uses this
   
   c) Hybrid Parallelism:
      - Mix neuron and batch parallelism
      - Best for: Most practical scenarios
      - Large layers → neuron split
      - Small layers → batch split
      - Example: Megatron-LM uses pipeline + tensor parallelism

2. HANDLING NON-DIVISIBLE LAYERS:
   
   - Use remainder distribution: first k processes get +1 neuron
   - Example: 10 neurons, 3 processes → [4, 3, 3]
   
   Code pattern:
   neurons_per_proc = layer_size // num_procs
   remainder = layer_size % num_procs
   
   for i in range(num_procs):
       extra = 1 if i < remainder else 0
       size = neurons_per_proc + extra

3. DISTRIBUTED LAYER OPERATIONS:
   
   a) Top-K selection:
      - Method 1: Gather all to root → O(n) communication
      - Method 2: Tournament reduction → O(k*log(P)) communication
      - Use Method 2 for large P, small k
   
   b) Normalization (LayerNorm, BatchNorm):
      - Requires all-reduce for mean/variance
      - 2 all-reduce calls per normalization
      - Consider GroupNorm to reduce communication
   
   c) Softmax:
      - Max: single all-reduce
      - Sum: single all-reduce  
      - Total: 2 all-reduce operations

4. COMMUNICATION PATTERNS:
   
   - Forward: Point-to-point or Broadcast
   - Backward: All-reduce for gradient aggregation
   - Use MPI subcommunicators for layer-local operations
   - Overlap communication with computation when possible

5. FRAMEWORKS USING THESE TECHNIQUES:
   
   - Megatron-LM (NVIDIA): Tensor + Pipeline parallelism
   - DeepSpeed (Microsoft): ZeRO optimizer + Pipeline
   - GPipe (Google): Pipeline parallelism
   - Mesh TensorFlow: N-dimensional mesh parallelism
   - FSDP (PyTorch): Fully Sharded Data Parallel

6. JAX-SPECIFIC CONSIDERATIONS:
   
   - Use jax.pmap for SPMD parallelism
   - Use custom_vjp for fine-grained MPI control
   - jax.experimental.maps.xmap for mesh parallelism
   - mpi4jax for explicit MPI in JAX functions
        """)
        print("="*80 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    comm.Barrier()
    
    # Print best practices
    print_best_practices()
    
    comm.Barrier()
    
    # Run simple distributed layer example
    if rank == 0:
        print("\n" + "="*80)
        print("RUNNING SIMPLE DISTRIBUTED LAYER EXAMPLE")
        print("="*80 + "\n")
    
    if size == 6:
        simple_distributed_layer_example()
    
    comm.Barrier()
    
    # Run mini example with 3 processes
    if rank == 0:
        print("\n" + "="*80)
        print("RUNNING MINI EXAMPLE (requires 3 processes)")
        print("="*80 + "\n")
    
    mini_example_3_processes()
    
    comm.Barrier()
    
    if rank == 0:
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED")
        print("="*80 + "\n")