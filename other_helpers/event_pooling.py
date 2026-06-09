import jax
import jax.numpy as jnp
from functools import partial


def pool_output_size(in_size, pool, stride):
    """Ceil-mode pooled output size: ceil((in_size - pool) / stride) + 1.

    Keeps a partial last window for odd dimensions instead of dropping the
    trailing row/column (floor mode). Equals the floor formula when the
    dimension is evenly covered. All static ints.
    """
    return -(-(in_size - pool) // stride) + 1


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
    
    # Ceil-mode pooling size: keep a partial last window for odd dims.
    Hp = pool_output_size(H, ph, sh)
    Wp = pool_output_size(W, pw, sw)
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
        (coords[:, 1] < H) &                        # ceil-mode: keep edge (partial window)
        (coords[:, 2] < W)
    )
    safe_pooled_idx = jnp.where(valid, pooled_idx, 0) # Replace invalid indices with a safe dummy (0)

    N = events.shape[0]
    num_segments = C * Hp * Wp

    if mode == "max":
        safe_values = jnp.where(valid, values, -jnp.inf)
        pooled_max = jax.ops.segment_max(safe_values, safe_pooled_idx, num_segments=num_segments)  # (C*Hp*Wp,)

        # For each input event, check whether it is the unique winner of its pooling cell.
        # "Unique" breaks ties by keeping the smallest event index.
        per_event_max = pooled_max[safe_pooled_idx]  # (N,)
        is_candidate = valid & (values > 0) & (values == per_event_max)
        first_winner = jax.ops.segment_min(
            jnp.where(is_candidate, jnp.arange(N), N),
            safe_pooled_idx,
            num_segments=num_segments,
        )  # (C*Hp*Wp,)
        is_unique_winner = is_candidate & (jnp.arange(N) == first_winner[safe_pooled_idx])

        out_val = jnp.where(is_unique_winner, values, 0.0)
        unpooled_vals = out_val  # pre-pool value at the winning cell, 0 elsewhere

    elif mode == "avg":
        safe_values = jnp.where(valid, values, 0.0)
        sums = jax.ops.segment_sum(safe_values, safe_pooled_idx, num_segments=num_segments)
        pooled_avg = sums / float(ph * pw)
        first_event = jax.ops.segment_min(
            jnp.where(valid, jnp.arange(N), N),
            safe_pooled_idx,
            num_segments=num_segments,
        )
        is_unique_winner = valid & (jnp.arange(N) == first_event[safe_pooled_idx])
        out_val = jnp.where(is_unique_winner, pooled_avg[safe_pooled_idx], 0.0)
        unpooled_vals = safe_values
    else:
        raise ValueError("mode must be 'max' or 'avg'")

    # Build (N, 4) output — one row per input event, using its pre-computed pooled coords.
    # Sort winners to front ordered by pooling cell index so downstream emit order matches
    # what the original full-map implementation produced (stable, ascending cell index).
    out = jnp.stack([pooled_c.astype(jnp.float32),
                     pooled_x.astype(jnp.float32),
                     pooled_y.astype(jnp.float32),
                     out_val], axis=-1)  # (N, 4)

    sort_key = jnp.where(is_unique_winner, safe_pooled_idx, num_segments)
    compact_out = out[jnp.argsort(sort_key)]
    nb_valid_el = jnp.sum(is_unique_winner).astype(jnp.int32)

    return nb_valid_el, compact_out, coords, unpooled_vals

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def output_to_event_array_with_pooling(activated_output, start_indices, end_indices, event_padding, pooling="", pool_size=(2,2), pool_stride=(2,2), rank=None):
    '''
    Transforms the activated output matrix into a list with format (c, x, y, value)
    to send to the next layer. And apply pooling if required.
    
    activated_output: (c, k_h, k_w) - the activated output corresponding to the input neuron
    start_indices: (c, x, y) - the starting indices of the slice in the padded neuron states
    end_indices: (c, h, w) - the shape of the original neuron states
    kernel_padding: (c, k_h_pad, k_w_pad) - the padding of the kernel
    '''
    c, h, w = activated_output.shape
    event_pad_h, event_pad_w = event_padding
    
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
    adjusted_coords = coords + jnp.array(start_indices) + jnp.array([0, -event_pad_h, -event_pad_w]) 
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
        nb_valid_el, padded_out_events = compact_nonzero_and_pad(out_events)

        unpooled_coords = out_events[:, :3].astype(jnp.int32)
        unpooled_vals = out_events[:, 3].astype(jnp.float32)

        # jax.debug.print("rank {} nbvalid el: {}, values {} compact out {}", rank, nb_valid_el, out_events, compact_out)
    # jax.debug.print("nbvalid el: {}, padded out events shape: {}, out events: {}", nb_valid_el, padded_out_events.shape, out_events)

    return nb_valid_el, padded_out_events, unpooled_coords, unpooled_vals

# region Full matrix pooling
@partial(jax.jit, static_argnums=(2, 3, 4,))
def unpool(input_matrix, pooled, shape, pool_size=(2, 2), pool_stride=(2, 2)):
    C, H, W = shape
    sh, sw = pool_stride
    kh, kw = pool_size

    # Pad the original map to the ceil-mode grid so windows align with `pooled`.
    out_h, out_w = pooled.shape[1], pooled.shape[2]
    pad_h = (out_h - 1) * sh + kh - H
    pad_w = (out_w - 1) * sw + kw - W
    inp = jnp.pad(input_matrix, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-jnp.inf)
    Hp, Wp = inp.shape[1], inp.shape[2]

    # Create upsampled version on the padded grid
    matrix_upsampled = jnp.repeat(jnp.repeat(pooled, sh, axis=1), sw, axis=2)[:, :Hp, :Wp]

    # Create mask for matching values (padded cells are -inf, never match)
    mask = (inp == matrix_upsampled)

    # Create priority matrix: prefer top-left positions
    priority_base = jnp.arange(Hp * Wp, dtype=jnp.float32).reshape(Hp, Wp)
    priority = jnp.broadcast_to(priority_base[None, :, :], (C, Hp, Wp))

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

    max_priorities_upsampled = jnp.repeat(jnp.repeat(max_priorities, sh, axis=1), sw, axis=2)[:, :Hp, :Wp]

    # Keep only positions that have the maximum priority in their window
    unique_mask = (priority == max_priorities_upsampled) & mask
    unpooled = jnp.where(unique_mask, matrix_upsampled, 0.0)

    # Crop padding back to the original resolution
    unpooled = unpooled[:, :H, :W]

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
        # Ceil-mode output dimensions (keep a partial last window for odd dims)
        out_h = pool_output_size(H, kh, sh)
        out_w = pool_output_size(W, kw, sw)
        # Pad so VALID windows realize the ceil-mode output without dropping the edge.
        pad_h = (out_h - 1) * sh + kh - H
        pad_w = (out_w - 1) * sw + kw - W
        pad_val = -jnp.inf if pooling == "max" else 0.0
        padded_input = jnp.pad(input_matrix, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=pad_val)
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
        matrix = pool_fn(padded_input)
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
    