import jax
import jax.numpy as jnp
from functools import partial

# region Sparse pooling 
@partial(jax.jit, static_argnames=['pad_len',])
def compact_nonzero_and_pad(events, pad_len=0):
    """
    Reorders (c, x, y, v) events so that all v != 0 come first, 
    and pads the rest with -2. Keeps output shape identical to input if pad_len is 0.

    Args:
        events: jnp.ndarray of shape (N, 4), each row [c, x, y, v].
        pad_len: int, number of extra rows to pad at the end (default 0)

    Returns:
        compacted: jnp.ndarray of shape (N, 4)
        nonzero_count: number of nonzero v entries (scalar int32)
    """
    values = events[:, 3]
    
    # Boolean mask: 1 for nonzero values
    mask = values != 0

    # Get indices that would sort mask so that True come first
    # (~mask) is used so that True (1) goes before False (0)
    sort_keys = ~mask
    perm = jnp.argsort(sort_keys.astype(jnp.int32))

    # Reorder events
    compacted = events[perm] #TODO Order the events from highest to lowest value

    # Count nonzero
    nonzero_count = jnp.sum(mask).astype(jnp.int32)

    # 1. Calculate the final target shape (pad_len must be static)
    total_rows = compacted.shape[0] + pad_len

    # 2. Pad the array to the final size in one go
    # We pad with 0 (or anything) because we will overwrite the mask anyway
    full_compacted = jnp.pad(
        compacted, 
        pad_width=((0, pad_len), (0, 0)), 
        mode="constant", 
        constant_values=0.0
    )

    # 3. Create a single index array for the entire padded height
    indices = jnp.arange(total_rows)

    # 4. Apply the mask once across the whole array
    # Pad the remaining and additional rows with -2
    # For zero entries, overwrite everything with -2
    compacted = jnp.where(
        indices[:, None] < nonzero_count, 
        full_compacted, 
        -2.0
    )
    # jax.debug.print("input events {}, nonzero count {}, compacted {}", events.shape, nonzero_count, compacted.shape)

    return nonzero_count, compacted


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5,))
def sparse_pool(events, input_shape, mode="max", pool_size=(2, 2), stride=(2, 2), pad_value=0.0):
    """
    Sparse pooling on (c,x,y,value) event region.

    Args:
        events: jnp.ndarray (N,4) containing (c, x, y, v). Use sentinel rows
                with c < 0 to indicate padding (they will be removed).
        input_shape: (C, H, W) (integers)
        mode: "max" or "avg"
        pool_size: (ph, pw)
        stride: (sh, sw)
        pad_value: value for empty pooled windows (default 0.0)

    Returns:
        pooled_events: jnp.ndarray shape (C * Hp * Wp, 4) with rows (c, x, y, value)
        pooled_shape: (C, Hp, Wp)
    """
    C, H, W = map(int, input_shape)
    ph, pw = map(int, pool_size)
    sh, sw = map(int, stride)
    
    # Pooling layer size (assumes H and W are divisible by stride if not we just ignore one row/column)
    Hp = H // sh
    Wp = W // sw
    # jax.debug.print("Pool output sizes: {} {} {} {} {}", Hp, Wp, C, H, W)

    coords = events[:, :3].astype(jnp.int32)   # (N,3) integers
    values = events[:, 3].astype(jnp.float32)  # (N,)
    # jax.debug.print("in events in sparse pool: {}", events.shape)

    # Map each event to pooling layer cell
    pooled_c = coords[:, 0].astype(jnp.int32)
    pooled_x = (coords[:, 1] // sh).astype(jnp.int32)
    pooled_y = (coords[:, 2] // sw).astype(jnp.int32)

    # Compute the index of the corresponding pooling layer cell 
    pooled_idx = pooled_c * (Hp * Wp) + pooled_x * Wp + pooled_y    # integer index (N,)
    valid = (
        (coords[:, 0] >= 0) &                       # valid channel
        (coords[:, 1] >= 0) & (coords[:, 2] >= 0) & # valid spatial coords
        (coords[:, 1] < Hp * sh) &                  # ignore last H row
        (coords[:, 2] < Wp * sw)                    # ignore last W column
    )
    safe_pooled_idx = jnp.where(valid, pooled_idx, 0) # Replace invalid indices with a safe dummy (0)

    num_segments = C * Hp * Wp # Number of cells in pooling layer
    # jax.debug.print("After pooling indexes {} {} {} {} {}", pooled_c, pooled_x, pooled_y, pooled_idx, num_segments)

    # --- pooling ---
    if mode == "max":
        # Replace invalid values so they never affect max
        safe_values = jnp.where(valid, values, -jnp.inf)

        pooled_values = jax.ops.segment_max(
            safe_values,
            safe_pooled_idx,
            num_segments=num_segments
        )
        # jax.debug.print("valid {}, \nsafe pooled idx {}, \nsafe values {}, \npooled values {} ",valid,  safe_pooled_idx, safe_values, pooled_values)
        
        # replace empty segments (-inf) with pad_value (0.0)
        pooled_values = jnp.where(jnp.isneginf(pooled_values), pad_value, pooled_values)

        # Compute the original index of the max values
        is_max = values == pooled_values[safe_pooled_idx]
        idx = jnp.arange(values.shape[0])
        argmax_idx = jax.ops.segment_min(jnp.where(is_max, idx, values.shape[0]),
                                        safe_pooled_idx,
                                        num_segments=num_segments)
        
        # Get the unpooled values for bpp
        unpooled = jnp.zeros_like(values)
        unpooled = unpooled.at[argmax_idx].set(pooled_values) # out of bound indexes are ignored in JAX
        # jax.debug.print("is max {}, idx {}, argmax_idx {}, segmin data {}, unpooled {} unpooled coords {} safe pooled idx {}, max pooled idx {} values {}", 
        #                 is_max, idx, argmax_idx, jnp.where(is_max, idx, values.shape[0]), unpooled, coords, safe_pooled_idx, 
        #                 jnp.where(argmax_idx < 1000000000, safe_pooled_idx[argmax_idx], -1), safe_values)
        unpooled_vals = unpooled
    elif mode == "avg":
        # Replace invalid values so they never affect avg
        safe_values = jnp.where(valid, values, 0)

        # segment_sum returns 0 for empty segments
        sums = jax.ops.segment_sum(safe_values, safe_pooled_idx, num_segments=num_segments)

        # average over full kernel area (include zeros)
        area = ph * pw
        pooled_values = sums / float(area)
        jax.debug.print("safe values {} after segment sum and avg {} {} ", safe_values, sums, pooled_values)
        unpooled_vals = safe_values
    else:
        raise ValueError("mode must be 'max' or 'avg'")

    # # reconstruct coords for every pooled cell in canonical order
    # pooled_c_full = jnp.repeat(jnp.arange(C, dtype=jnp.int32), Hp * Wp)
    # pooled_x_full = jnp.tile(jnp.repeat(jnp.arange(Hp, dtype=jnp.int32), Wp), C)
    # pooled_y_full = jnp.tile(jnp.arange(Wp, dtype=jnp.int32), C * Hp)

    # coords_full = jnp.stack([pooled_c_full, pooled_x_full, pooled_y_full], axis=-1)  # (num_segments, 3)
    # jax.debug.print("pooled coords full {}, coords {} pooled values {}", coords_full, coords, pooled_values)
    # out = jnp.concatenate([coords_full, pooled_values[:, None]], axis=-1)  # (num_segments, 4)

    max_pooled_idx = jnp.where(argmax_idx < 1000000000, safe_pooled_idx[argmax_idx], -1)
    # Convert flat indices to (c, x, y) coordinates
    pooled_c = max_pooled_idx // (Hp * Wp)
    pooled_x = (max_pooled_idx % (Hp * Wp)) // Wp
    pooled_y = max_pooled_idx % Wp

    # Stack to get coordinates
    coords_sparse = jnp.stack([pooled_c, pooled_x, pooled_y], axis=-1)  # (num_events, 3)

    # Concatenate with pooled values
    out = jnp.concatenate([coords_sparse, pooled_values[:, None]], axis=-1)  # (num_events, 4)

    # jax.debug.print("new pooled coords sparse {}, pooled values {}, out shape {}", coords_sparse, pooled_values, out.shape)

    nb_valid_el, compact_out = compact_nonzero_and_pad(out)

    return nb_valid_el, compact_out, coords, unpooled_vals

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def output_to_event_array_with_pooling(activated_output, start_indices, end_indices, kernel_span, pooling="", pool_size=(2,2), pool_stride=(2,2), rank=None):
    '''
    Transforms the activated output matrix into a list with format (c, x, y, value)
    to send to the next layer. And apply pooling if required.
    
    activated_output: (c, k_h, k_w) - the activated output corresponding to the input neuron
    start_indices: (c, x, y) - the starting indices of the slice in the padded neuron states
    end_indices: (c, h, w) - the shape of the original neuron states
    kernel_padding: (c, k_h_pad, k_w_pad) - the padding of the kernel
    '''
    c, h, w = activated_output.shape
    kernel_h_span, kernel_w_span = kernel_span
    
    # Step 1: Create coordinate grid
    c_grid, x_grid, y_grid = jnp.meshgrid(
        jnp.arange(c),
        jnp.arange(h),
        jnp.arange(w),
        indexing='ij'
    )
    # Flatten everything
    coords = jnp.stack([c_grid.ravel(), x_grid.ravel(), y_grid.ravel()], axis=-1)

    # Step 2: Adjust coordinates with start indices and kernel span for padding offset
    adjusted_coords = coords + jnp.array(start_indices) + jnp.array([0, -kernel_h_span, -kernel_w_span]) 
    # jax.debug.print("coords {}, start indices {}, adjusted coords: {}", coords, start_indices, adjusted_coords)

    # Step 3: Filter out-of-bounds coordinates
    is_valid = jnp.all(
        (adjusted_coords >= jnp.zeros((3,))) &
        (adjusted_coords < jnp.array(end_indices)),
        axis=-1
    )
    values = activated_output.ravel()
    values_masked = values * is_valid.astype(values.dtype)
    # jax.debug.print("values: {}, is valid: {}, values masked: {}", values, is_valid, values_masked)

    # Step 4: Combine adjusted coordinates and valid values
    out_events = jnp.concatenate([adjusted_coords, values_masked[:, None]], axis=-1)
    # jax.debug.print("out events: {}", out_events)
    
    # jax.debug.print("activated output: \n{}", activated_output)

    # Step 5: Apply pooling if needed
    if pooling != "":
        nb_valid_el, padded_out_events, unpooled_coords, unpooled_vals = sparse_pool(out_events, end_indices, pooling, pool_size, pool_stride)
        # jax.debug.print("after pool el: {} and out shape: {}", nb_valid_el, padded_out_events.shape)
    else:
        # Compute full size and valid size
        target_size = (end_indices[0]) * (end_indices[1]) * (end_indices[2])
        out_size = activated_output.size
        nb_valid_el = jnp.count_nonzero(out_events[:, 3])
        
        # jax.lax.cond
        # Pad out_events to full size
        # pad_to_full_size = jnp.full((target_size-nb_valid_el, 4), -2)
        # padded_out_events = jnp.concatenate([out_events, pad_to_full_size])
        
        pad_len = target_size - out_size

        padded_out_events = jnp.pad(
            out_events,
            pad_width=((0, pad_len), (0, 0)),  # pad rows only
            mode="constant",
            constant_values=-2
        )
        nb_valid_el, padded_out_events = compact_nonzero_and_pad(out_events, pad_len)

        unpooled_coords = out_events[:, :3].astype(jnp.int32) # (N,3) integers
        unpooled_vals = out_events[:, 3].astype(jnp.float32) # (N,)

        # jax.debug.print("rank {} nbvalid el: {}, values {} compact out {}", rank, nb_valid_el, out_events, compact_out)
    # jax.debug.print("nbvalid el: {}, padded out events shape: {}, out events: {}", nb_valid_el, padded_out_events.shape, out_events)

    return nb_valid_el, padded_out_events, unpooled_coords, unpooled_vals

# region Full matrix pooling
@partial(jax.jit, static_argnums=(2, 3, 4,))
def unpool(input_matrix, pooled, shape, pool_size=(2, 2), pool_stride=(2, 2)):
    C, H, W = shape
    sh, sw = pool_stride
    kh, kw = pool_size

    # Create upsampled version
    x_up_h = jnp.repeat(pooled, sh, axis=1)
    matrix_upsampled = jnp.repeat(x_up_h, sw, axis=2)
    
    # Pad if necessary to match original dimensions
    pad_h = H - matrix_upsampled.shape[1]
    pad_w = W - matrix_upsampled.shape[2]
    if pad_h > 0 or pad_w > 0:
        matrix_upsampled = jnp.pad(matrix_upsampled, 
                                ((0, 0), (0, pad_h), (0, pad_w)), 
                                constant_values=0)
    
    # Create mask for matching values
    mask = (input_matrix == matrix_upsampled)
    
    # Create priority matrix: prefer top-left positions
    priority_base = jnp.arange(H * W, dtype=jnp.float32).reshape(H, W)
    priority = jnp.broadcast_to(priority_base[None, :, :], (C, H, W))
    
    # Apply mask: set non-matching positions to -inf (not 0!)
    priority = jnp.where(mask, priority, -jnp.inf)
    
    # For each pooling window, keep only the position with highest priority
    max_priorities = jax.lax.reduce_window(
        priority,
        init_value=-jnp.inf,
        computation=jax.lax.max,
        window_dimensions=(1, kh, kw),
        window_strides=(1, sh, sw),
        padding="VALID",
    )
    
    max_priorities_upsampled = jnp.repeat(jnp.repeat(max_priorities, sh, axis=1), sw, axis=2)
    
    if pad_h > 0 or pad_w > 0:
        max_priorities_upsampled = jnp.pad(max_priorities_upsampled, 
                                            ((0, 0), (0, pad_h), (0, pad_w)), 
                                            constant_values=-jnp.inf)
    
    # Keep only positions that have the maximum priority in their window
    unique_mask = (priority == max_priorities_upsampled) & mask
    unpooled = matrix_upsampled * unique_mask

    # jax.debug.print("rank {}, input {}, pooled {}, unpooled {}", rank, jnp.count_nonzero(input_matrix), jnp.count_nonzero(pooled), jnp.count_nonzero(unpooled))
    return unpooled

@partial(jax.jit, static_argnums=(1, 2, 3, 4,))
def full_matrix_to_event_array_with_pooling(matrix, shape, pooling="", pool_size=(2, 2), pool_stride=(2, 2), rank=None):
    """
    Convert a full (C, H, W) matrix into event array format (N, 4),
    keeping only nonzero values at the beginning, padded with -2s at the end.

    Args:
        matrix: jnp.ndarray of shape (C, H, W)
        shape: tuple (C, H, W) – static shape of the full matrix

    Returns:
        (num_nonzero, padded_events)
        - num_nonzero: scalar, number of nonzero entries
        - padded_events: (C*H*W, 4) array of [c, x, y, value]
                         first num_nonzero rows are valid, rest are -2
    """
    C, H, W = shape

    unpooled = jnp.zeros_like(matrix)
    if pooling != "":
        kh, kw = pool_size
        sh, sw = pool_stride

        input_matrix = matrix
        # Compute output dimensions
        out_h = (H - kh) // sh + 1
        out_w = (W - kw) // sw + 1
        # io_callback(lambda arr, name: save_to_file(arr, name), None, input_matrix, 0)
        
        @jax.jit
        def pool_fn(x):
            # Extract all pooling windows efficiently
            windows = jax.lax.reduce_window(
                x,
                init_value=-jnp.inf if pooling == "max" else 0.0,
                computation=jax.lax.max if pooling == "max" else jax.lax.add,
                window_dimensions=(1, kh, kw),
                window_strides=(1, sh, sw),
                padding="VALID"
            )
            if pooling == "avg":
                windows = windows / (kh * kw)
            return windows

        # Apply pooling channel-wise
        matrix = pool_fn(input_matrix)
        # io_callback(lambda arr, name: save_to_file(arr, name), None, matrix, 1)

        if pooling == "max":
            unpooled = unpool(input_matrix, matrix, shape, pool_size, pool_stride)
        # io_callback(lambda arr, name: save_to_file(arr, name), None, unpooled, 2)
        
        # Update the shape for output computation
        H, W = matrix.shape[1:]
        shape = (C, H, W)
    
    N = C * H * W
    # Coordinate grid
    c_grid, x_grid, y_grid = jnp.meshgrid(
        jnp.arange(C), jnp.arange(H), jnp.arange(W), indexing='ij'
    )
    coords = jnp.stack([c_grid.ravel(), x_grid.ravel(), y_grid.ravel()], axis=-1)  # (N,3)

    # Build event array
    values = matrix.ravel()  # (N,)
    out_events = jnp.concatenate([coords, values[:, None]], axis=-1)  # (N,4)

    num_nonzero, padded_out_events = compact_nonzero_and_pad(out_events)

    if pooling == "":
        unpooled = matrix
    # jax.debug.print("Non zeros after pooling: {}", num_nonzero)
    # jax.debug.print("rank {}, output matrix \n{}, out events {}, nb non zero {} padded_events {}", 
    #                 rank, matrix, out_events.shape, num_nonzero, padded_out_events)
    return num_nonzero, padded_out_events, unpooled
    