import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

import mpi4jax
from mpi4jax import send, recv, bcast

from other_helpers.helpers import activation_func, keep_top_k
from other_helpers.helpers import update_history
from other_helpers.event_pooling import output_to_event_array_with_pooling, full_matrix_to_event_array_with_pooling
from other_helpers.general_MPI_helper import CNN_layer_Partition

# Set to 1/2/3 to enable Bug Tests 1/2/3 in conv runs.
# 1 = BT1: output event coordinate assertion per input event
# 2 = BT2: total-events-received summary after while_loop
# 3 = BT3: END_SIGNAL receipt tracing
_DEBUG_LEVEL = 0

@partial(jax.jit, static_argnames=['params', 'mpi_config'])
def process_activated_output(params, mpi_config, key, arr: jnp.ndarray):
    '''
    Processed the output of a layer from (1d array) to (2d array) with [(neuron idx, value)]
    value == 0 are filled with index==-2
    '''
    # max_len = params.layer_sizes[mpi_config.layer_idx]
    max_len = mpi_config.model_part.get_size

    # indices of nonzero values (padded with -2)
    idx = jnp.nonzero(arr, size=max_len, fill_value=-2)[0]
    vals = jnp.where(idx != -2, arr[idx], -2)

    # Offset local idx by model partition start so downstream layers see global neuron indices
    idx = jnp.where(idx != -2, idx + mpi_config.model_part.start_idx, idx)

    # stack before shuffle
    pairs = jnp.stack([idx, vals], axis=1)

    def do_shuffle(pairs):
         # mask: 1 for valid entries, 0 for padded (-2, 0)
        mask = (idx != -2).astype(jnp.int32)
        
        # assign random keys for sorting
        rand_keys = jax.random.uniform(key, (max_len,))

        # ensure valid entries come first, shuffled within themselves
        sort_keys = jnp.where(mask == 1, rand_keys, rand_keys + 2.0)  
        permuted = pairs[jnp.argsort(sort_keys)]

        return permuted
    
    def do_sort_by_value(pairs):
        # Sort by value (descending), with padding entries at the end
        # Use negative values for descending sort, add large number to padding
        mask = (idx != -2).astype(jnp.int32)
        sort_keys = jnp.where(mask == 1, -vals, 1e10)  # valid: -value, padding: large number
        sorted_pairs = pairs[jnp.argsort(sort_keys)]
        return sorted_pairs

    # pairs_out = jax.lax.cond(
    #     params.shuffle_activations,
    #     do_shuffle,
    #     do_sort_by_value, #lambda pairs: pairs,
    #     operand=pairs
    # )
    if params.shuffle_activations:
        pairs_out = do_shuffle(pairs)
    else:
        pairs_out = do_sort_by_value(pairs)

    return pairs_out

# region layer_computation
@partial(jax.jit, static_argnames=['params', 'mpi_config', 'grad'])
def layer_computation(params, mpi_config, key, neuron_idx, layer_input, weights, neuron_states, iteration=0, grad=False, is_residual=False):
    # is_residual accepted for call-site parity with conv_layer_computation; FC layers ignore it.
    layer_idx= mpi_config.layer_idx
    last_layer = mpi_config.last_layer_idx

    is_MLP = neuron_idx.ndim == 0
    # jax.debug.print("rank {}, is_MLP: {}, neuron idx: {}", mpi_config.rank, is_MLP, neuron_idx)

    if is_MLP: # MLP network
        neuron_idx = neuron_idx.astype(jnp.int32)
    else: # Linear layer in CNN network
        c, x, y = neuron_idx

        # Compute the flattened neuron index
        C, H, W = 0, 0, 0
        flat_layer_size = params.flat_layer_sizes[layer_idx-1]
        if len(flat_layer_size) == 3: # Fail safe in case there is no hidden layer
            C, H, W = flat_layer_size
        
        # if layer_idx == 2:
        #     jax.debug.print("rank {}, is_MLP: {}, H {}, W {}, neuron idx: {}, flat layer sizes {}", mpi_config.rank, is_MLP, H, W, neuron_idx, flat_layer_size)

        neuron_idx = c * (H * W) + x * W + y 

        # if layer_idx == 2:
        #     jax.debug.print("rank {}, is_MLP: {}, H {}, W {}, neuron idx: {}, flat layer sizes {}", mpi_config.rank, is_MLP, H, W, neuron_idx, flat_layer_size)

    invalid_idx = neuron_idx < 0  # True when end-of-stream signal received
    
    # Compute the new values of the neuron states
    activations = jax.lax.cond(invalid_idx,
                            lambda _: neuron_states.values,
                            lambda _: jnp.dot(layer_input, weights[neuron_idx]) + neuron_states.values,
                            None
                            )
    #TODO being able to compute multiple incoming index neurons
    #TODO store the weight residuals of last layer and neuron state of input layer in sparse matrix representation to reduce space utilization because unused
    
    # jax.lax.cond(neuron_idx == -1,
    #                 lambda _: jax.debug.print("rank {}, iteration: {}, neuron idx: {}", rank, iteration, neuron_idx),
    #                 lambda _: None,
    #                 None)

    if grad:
        new_input_residuals = jax.lax.cond(invalid_idx,
                                lambda _: neuron_states.input_residuals,
                                lambda _: neuron_states.input_residuals.at[neuron_idx].add(layer_input),
                                None
                                )
        new_input_activity = jax.lax.cond(invalid_idx,
                                lambda _: neuron_states.input_activity,
                                lambda _: neuron_states.input_activity.at[neuron_idx].add(1),
                                None
                                )
    else:
        new_input_residuals = neuron_states.input_residuals
        new_input_activity = neuron_states.input_activity

    def last_layer_case(): # No need for additional computation at the output layer
        new_values_history, new_history_index = neuron_states.values_history, neuron_states.history_index

        # Apply per-event weight decay: contribution at event t is scaled by output_decay^t.
        # (activations - neuron_states.values) isolates the new dot-product contribution;
        # for END_SIGNAL (invalid_idx) this term is zero, so no special casing needed.
        decayed_activations = neuron_states.values + (activations - neuron_states.values) * (params.output_decay ** iteration)

        if params.history_size > 0:
            new_values_history, new_history_index = update_history(new_values_history, new_history_index, decayed_activations)

        shape = 2
        if not is_MLP:
            shape = 4 # CNN network output format: (c, x, y, v)

        dummy_activations = jnp.zeros((activations.shape[0], shape))

        # if layer_idx == 2:
        #     jax.debug.print("rank {}, is_MLP: {}, H {}, W {}, neuron idx: {}, dummy_activations {}", mpi_config.rank, is_MLP, H, W, neuron_idx, dummy_activations)
        return jnp.array(0), dummy_activations, neuron_states.replace(  values=decayed_activations,
                                                                        input_residuals=new_input_residuals,
                                                                        input_activity=new_input_activity,
                                                                        values_history=new_values_history,
                                                                        history_index=new_history_index,)
    
    def hidden_layer_case():
        # APPLY THE SYNC RATE
        if params.global_sync:
            # Global per-layer (per-shard) gate: the layer can fire at all only if
            # sync_rate events have elapsed since its last firing (max over owned
            # neurons of last_sent_iteration). All-or-nothing, so raising sync_rate
            # to the split factor S cleanly divides each shard's firing rate by S.
            layer_last = jnp.max(neuron_states.last_sent_iteration)
            layer_eligible = (iteration - layer_last >= neuron_states.sync_rate_vector[0]).astype(jnp.int32)
            sync_fire = jnp.full(neuron_states.last_sent_iteration.shape, layer_eligible, dtype=jnp.int32)
        else:
            # Per-neuron gate: only neurons that fired within the last sync_rate
            # events are blocked; the top-k among the rest still fires.
            sync_fire = (iteration - neuron_states.last_sent_iteration >= neuron_states.sync_rate_vector).astype(jnp.int32)
        sync_fire = jax.lax.cond(invalid_idx, lambda _: jnp.ones(sync_fire.shape, dtype=jnp.int32), lambda _: sync_fire, None)
        activated_output = activations * sync_fire

        # APPLY ACTIVATION FUNCTION
        activated_output = activation_func(neuron_states.thresholds, activated_output)

        # APPLY PER-SAMPLE DROPOUT (training only): mask candidate neurons before top-k.
        # The mask is drawn once per sample in loop_over_batches and held fixed across
        # all of that sample's events, so a dropped neuron stays out of the running for
        # the whole sample (works with firing_nb=1: the runner-up fires instead).
        _drop_p = params.dropout[layer_idx] if getattr(params, "dropout", None) is not None else 0.0
        if grad and _drop_p > 0.0:
            activated_output = activated_output * neuron_states.dropout_mask

        # APPLY THE FIRING NUMBER
        f_nb = params.firing_nb
        k = f_nb if isinstance(f_nb, int) else f_nb[layer_idx]
        activated_output = keep_top_k(activated_output, k)

        # APPLY THE RESTRICTION
        reset = params.restrict
        if not isinstance(reset, int) and not isinstance(reset, float):
            reset = reset[layer_idx]
        
        # penalty = jax.lax.cond(reset <= 0,
        #                        lambda _: activated_output,
        #                        lambda _: activated_output * reset, None)
        penalty = activated_output * reset if reset > 0 else activated_output

        active_mask = (activated_output > 0)
        fire = jnp.logical_and(sync_fire.astype(bool), active_mask)
        new_last_sent_iteration = jnp.where(fire, iteration, neuron_states.last_sent_iteration)

        if grad:
            active_indexes = active_mask.astype(neuron_states.layer_activity.dtype)
            last_neuron_idx = jnp.argmax(neuron_states.input_order)
            new_neuron_idx = jax.lax.cond(invalid_idx, lambda _: last_neuron_idx, lambda _: neuron_idx, None)

            new_neuron_states = neuron_states.replace(
                values=activations - penalty,
                input_residuals=new_input_residuals,
                input_activity=new_input_activity,
                layer_activity=neuron_states.layer_activity + active_indexes,
                input_order=neuron_states.input_order.at[new_neuron_idx].set(iteration),
                output_activity=neuron_states.output_activity.at[new_neuron_idx].add(active_indexes),
                input_vector=neuron_states.input_vector.at[neuron_idx].set(iteration + 1),
                output_vector=jnp.where(active_mask, iteration + 1, neuron_states.output_vector),
                last_sent_iteration=new_last_sent_iteration,
            )
        else:
            new_neuron_states = neuron_states.replace(
                values=activations - penalty,
                input_residuals=new_input_residuals,
                input_activity=new_input_activity,
                last_sent_iteration=new_last_sent_iteration,
            )

        valid_elements = jnp.count_nonzero(activated_output)
        # process_activated_output emits this rank's owned slice with global neuron indices
        # (offset by model_part.start_idx), so model-split FC layers work in both networks.
        processed_output = process_activated_output(params, mpi_config, key, activated_output)
        if not is_MLP:
            # if layer_idx == 3:
            #     jax.debug.print("rank {}, activated_output: {}", mpi_config.rank, activated_output)
            # Pad to CNN format
            processed_output = jnp.pad(processed_output, ((0, 0), (2, 0)), constant_values=-2)

        return valid_elements, processed_output, new_neuron_states
    
    if layer_idx == last_layer:
        return last_layer_case()
    else:
        return hidden_layer_case()
    
    # TEST MPI WITH CONTROLLED NUMBER OF ACTIVATIONS
    def first_hidden(activations):
        return jnp.ones(activations.shape), neuron_states
    
    def other_hidden(activations):
        half_ones = jnp.ones(1)  # half ones
        half_zeros = jnp.zeros(activations.shape[0]-1)  # half zeros

        # Concatenate them
        arr = jnp.concatenate([half_ones, half_zeros])
        return arr, neuron_states
    
    return jax.lax.cond(rank == 1, first_hidden, other_hidden, (activations))

#region predict
@partial(jax.jit, static_argnames=['params', 'mpi_config', 'layer_computation', 'message_size', 'grad', 'BUFFER_SIZE'])
def predict(params, 
            mpi_config, 
            key, 
            weights, 
            empty_neuron_states, 
            layer_computation,
            batch_data: jnp.ndarray,
            message_size=2, 
            grad=False, 
            END_SIGNAL=jnp.array([-1.0, -1.0], dtype=jnp.float32),
            BUFFER_SIZE=0):
    '''
    Inference loop, each layer sends each event separately in the format: 
        MLP: (neuron idx, value)
        CNN: (c, x, y, value)

    value = -1 means end of data from previous layer (END_SIGNAL)
            -2 means placeholder data in the input layer to match the shape
    '''
    layer_idx= mpi_config.layer_idx
    last_layer = mpi_config.last_layer_idx
    
    # Compute first hidden layer's kernel geometry for selective routing from the input layer.
    # Static at trace time (params and mpi_config are static args).
    _first_hidden_has_cnn_partition = (
        len(mpi_config.next_layer) > 0
        and isinstance(mpi_config.next_layer[0][1], CNN_layer_Partition)
    )
    if _first_hidden_has_cnn_partition and len(params.layer_sizes) > 1:
        _fh_spec = params.layer_sizes[1]  # layer 1 = first hidden (input is layer 0)
        _fh_k_h, _fh_k_w = _fh_spec[1]
        _fh_pad_h, _fh_pad_w = _fh_spec[2]
        _fh_event_pad_h = _fh_k_h - 1 - _fh_pad_h
        _fh_event_pad_w = _fh_k_w - 1 - _fh_pad_w
    else:
        _fh_k_h = _fh_k_w = _fh_event_pad_h = _fh_event_pad_w = 0

    # Input-layer model parallelism: at trace time, decide whether this rank owns only a
    # spatial sub-region of the input. When split, each input rank emits only the events
    # inside its model_part; the union over ranks is the full (non-overlapping) input.
    _input_mp = mpi_config.model_part
    _input_is_split = (
        isinstance(_input_mp, CNN_layer_Partition)
        and _input_mp.get_size != _input_mp.total_size
    )

    def input_layer(x):
        x_p = jnp.array(x)

        if params.shuffle_input:
            perm = jax.random.permutation(key, x_p.shape[0])
            x_p = x_p[perm]

        if _input_is_split:
            # Keep only this rank's region; compact those events to the front so the send
            # loop below (which forwards the first `loop_iterations` rows) emits exactly them.
            ev_c, ev_x, ev_y = x_p[:, 0], x_p[:, 1], x_p[:, 2]
            active = ev_c != -2
            in_c = (ev_c >= _input_mp.c_start_idx) & (ev_c <= _input_mp.c_end_idx)
            in_x = (ev_x >= _input_mp.x_start_idx) & (ev_x <= _input_mp.x_end_idx)
            in_y = (ev_y >= _input_mp.y_start_idx) & (ev_y <= _input_mp.y_end_idx)
            keep = active & in_c & in_x & in_y
            x_p = jnp.where(keep[:, None], x_p, -2.0)
            x_p = x_p[jnp.argsort(~keep)]

        _next_sr = params.sync_rate if isinstance(params.sync_rate, int) else params.sync_rate[1]
        if _next_sr == 10000:
            # Bulk path: send all events in one MPI message instead of one per event.
            # The receiver (sr=10000 layer) expects a (max_nonzero, 4) array followed by END_SIGNAL.
            mask = x_p[:, 0] != -2
            loop_iterations = jnp.count_nonzero(mask)
            mpi_config.forward_send_bulk(x_p)
        else:
            def send_input(i, _):
                data = x_p[i]
                if _first_hidden_has_cnn_partition:
                    # data = [c, x, y, value] — route only to affected next-layer ranks
                    mpi_config.forward_send_cnn(
                        data,
                        data[0].astype(jnp.int32),
                        data[1].astype(jnp.int32),
                        data[2].astype(jnp.int32),
                        _fh_k_h, _fh_k_w,
                        _fh_event_pad_h, _fh_event_pad_w,
                    )
                else:
                    mpi_config.forward_send(data)
                return i

            mask = (x_p != -2)
            loop_iterations = (jnp.count_nonzero(mask)/message_size).astype(int)
            loop_iterations = jax.lax.fori_loop(0, loop_iterations, send_input, (0))

        # Send end signal to ALL next-layer ranks regardless of selective routing
        mpi_config.forward_send(END_SIGNAL)
        if len(mpi_config.res_connect_next) > 0:
            mpi_config.residual_send_end_signal(END_SIGNAL)

        return loop_iterations, jnp.zeros((BUFFER_SIZE, message_size))

    def other_layers(neuron_states):
        # Determine next-layer kernel geometry for selective CNN event routing.
        # All static at trace time — zero runtime overhead.
        _next_layer_idx = layer_idx + 1
        _next_has_cnn_partition = (
            len(mpi_config.next_layer) > 0
            and isinstance(mpi_config.next_layer[0][1], CNN_layer_Partition)
        )
        if _next_has_cnn_partition and _next_layer_idx < len(params.layer_sizes):
            _next_spec = params.layer_sizes[_next_layer_idx]
            _next_k_h, _next_k_w = _next_spec[1]
            _next_pad_h, _next_pad_w = _next_spec[2]
            _next_event_pad_h = _next_k_h - 1 - _next_pad_h
            _next_event_pad_w = _next_k_w - 1 - _next_pad_w
        else:
            _next_k_h = _next_k_w = _next_event_pad_h = _next_event_pad_w = 0

        _sr = params.sync_rate if isinstance(params.sync_rate, int) else params.sync_rate[layer_idx]

        if _sr == 10000 and layer_idx == 1 and mpi_config.nb_previous == 1:
            # Bulk path: receive all events in one MPI message, process locally, then
            # receive END_SIGNAL to trigger the full-layer firing (last_input).
            # Requires the input layer to have used forward_send_bulk (gated by _next_sr == 10000).

            # Compute dummy activated_output shape for the no-op branch in jax.lax.cond.
            # Mirrors event_array_size logic in conv_layer_computation.
            _out_ch, _, _k_h, _k_w = weights.shape
            _pad_h, _pad_w = neuron_states.padding
            _ep_h, _ep_w = _k_h - 1 - _pad_h, _k_w - 1 - _pad_w
            _, _pH, _pW = neuron_states.values.shape
            if neuron_states.pooling != "":
                _C_f, _H_f, _W_f = params.flat_layer_sizes[layer_idx]
                _dummy_event_size = _C_f * _H_f * _W_f
            else:
                _dummy_event_size = _out_ch * (_pH - 2 * _ep_h) * (_pW - 2 * _ep_w)

            # One recv for the entire event array (sender sent forward_send_bulk)
            bulk = mpi_config.forward_recv_bulk(params.max_nonzero)  # (max_nonzero, 4)

            # Process each event locally — no MPI per event
            def process_event(i, ns):
                event = bulk[i]
                c_raw = event[0].astype(jnp.int32)
                x_raw = event[1].astype(jnp.int32)
                y_raw = event[2].astype(jnp.int32)
                neuron_idx_e = jnp.stack([c_raw, x_raw, y_raw])
                _, _, new_ns = jax.lax.cond(
                    c_raw >= 0,  # -2 = padding sentinel
                    lambda ns: layer_computation(params, mpi_config, key, neuron_idx_e, event[3], weights, ns, i, grad, is_residual=jnp.array(False)),
                    lambda ns: (jnp.array(0), jnp.zeros((_dummy_event_size, 4)), ns),
                    ns,
                )
                return new_ns

            neuron_states = jax.lax.fori_loop(0, params.max_nonzero, process_event, neuron_states)

            # Receive END_SIGNAL → conv_layer_computation routes to last_input, firing all neurons
            end_data = mpi_config.forward_recv(message_size)
            end_idx = jnp.stack([end_data[0].astype(jnp.int32), end_data[1].astype(jnp.int32), end_data[2].astype(jnp.int32)])
            loop_iterations, activated_output, neuron_states = layer_computation(
                params, mpi_config, key, end_idx, end_data[3], weights, neuron_states, params.max_nonzero, grad, is_residual=jnp.array(False)
            )
            layer_input = end_data[3]

            if layer_idx != last_layer:
                def send_act(i, _):
                    out_val = activated_output[i]
                    if _next_has_cnn_partition:
                        mpi_config.forward_send_cnn(
                            out_val,
                            out_val[0].astype(jnp.int32),
                            out_val[1].astype(jnp.int32),
                            out_val[2].astype(jnp.int32),
                            _next_k_h, _next_k_w,
                            _next_event_pad_h, _next_event_pad_w,
                        )
                    else:
                        mpi_config.forward_send(out_val)
                    if len(mpi_config.res_connect_next) > 0:
                        mpi_config.residual_send(
                            out_val,
                            out_val[0].astype(jnp.int32),
                            out_val[1].astype(jnp.int32),
                            out_val[2].astype(jnp.int32),
                        )
                    return None
                jax.lax.fori_loop(0, loop_iterations, send_act, None)
                mpi_config.forward_send(END_SIGNAL, params.max_nonzero)
                if len(mpi_config.res_connect_next) > 0:
                    mpi_config.residual_send_end_signal(END_SIGNAL)

            return layer_input, neuron_states, jnp.array(params.max_nonzero, dtype=jnp.int32), jnp.zeros((BUFFER_SIZE, 2))

        def cond(state): # Stop when all previous-layer senders have signaled end-of-stream
            _, _, finished, _, _ = state
            return finished < mpi_config.nb_previous

        def forward_pass(state):
            layer_input, neuron_states, finished, iteration, buffer = state
            def hidden_layers(loop_iterations, activated_output):
                def send_activation(i, _):
                    out_val = activated_output[i]
                    if _next_has_cnn_partition:
                        # out_val = [c_out, x_out, y_out, value] — use output coords
                        # to route only to next-layer ranks whose kernel window overlaps
                        mpi_config.forward_send_cnn(
                            out_val,
                            out_val[0].astype(jnp.int32),
                            out_val[1].astype(jnp.int32),
                            out_val[2].astype(jnp.int32),
                            _next_k_h, _next_k_w,
                            _next_event_pad_h, _next_event_pad_w,
                        )
                    else:
                        mpi_config.forward_send(out_val)

                    # Residual: also fan out as residual to any res_connect_next destinations.
                    if len(mpi_config.res_connect_next) > 0:
                        mpi_config.residual_send(
                            out_val,
                            out_val[0].astype(jnp.int32),
                            out_val[1].astype(jnp.int32),
                            out_val[2].astype(jnp.int32),
                        )
                    return None
                jax.lax.fori_loop(0, loop_iterations, send_activation, None)
                return None
            
            received_data = mpi_config.forward_recv(message_size)

            # if layer_idx == 4:
            # jax.debug.print("rank {}, received: {}", mpi_config.rank, received_data)
            if message_size == 2:
                neuron_idx, layer_input = received_data[0], received_data[1]
                is_residual = jnp.array(False)
            else:
                c_raw = received_data[0].astype(jnp.int32)
                x_raw = received_data[1].astype(jnp.int32)
                y_raw = received_data[2].astype(jnp.int32)
                layer_input = received_data[3]

                is_end = jnp.logical_and(c_raw == -1, x_raw == -1)         # full END_SIGNAL
                is_residual = jnp.logical_and(c_raw < 0, x_raw >= 0)        # residual marker
                actual_c = jnp.where(is_residual, -c_raw - 1, c_raw)
                neuron_idx = jnp.stack([actual_c, x_raw, y_raw])

            neuron_idx = jnp.asarray(neuron_idx, dtype=jnp.int32)

            loop_iterations, activated_output, new_neuron_states = layer_computation(
                params,
                mpi_config,
                key,
                neuron_idx,
                layer_input,
                weights,
                neuron_states,
                iteration,
                grad,
                is_residual=is_residual,
            )

            neg_idx = jnp.any(neuron_idx == -1) if message_size == 2 else is_end
            finished = jax.lax.cond(neg_idx, lambda _: finished + 1, lambda _: finished, operand=None)
            iteration = jax.lax.cond(neg_idx, lambda _: iteration, lambda _: iteration + 1, operand=None)

            # Bug Test 3: trace each END_SIGNAL receipt vs nb_previous.
            if _DEBUG_LEVEL >= 3:
                jax.lax.cond(
                    neg_idx,
                    lambda _: jax.debug.print(
                        "BT3 rank={} layer={} END_SIG #{}/{} nb_prev={}",
                        mpi_config.rank, mpi_config.layer_idx, finished, finished, mpi_config.nb_previous,
                    ),
                    lambda _: None,
                    operand=None,
                )

            if layer_idx != last_layer:
                hidden_layers(loop_iterations, activated_output)
            
            # if layer_idx == 2:
            #     jax.debug.print("rank {}, loop_iterations: {}, neuron activated_output: {}, iteration {}", mpi_config.rank, loop_iterations, activated_output, iteration)

            return layer_input, new_neuron_states, finished, iteration, buffer

        finished = jnp.array(0)
        layer_input = jnp.zeros(())
        initial_state = (layer_input, neuron_states, finished, 0, jnp.zeros((BUFFER_SIZE, 2)))

        layer_input, neuron_states, finished, iteration, buffer = jax.lax.while_loop(cond, forward_pass, initial_state)

        # Bug Test 2: print total events received and owned-region info for each rank.
        if _DEBUG_LEVEL >= 2:
            _mp = mpi_config.model_part
            _is_cnn = isinstance(_mp, CNN_layer_Partition)
            _xs = _mp.x_start_idx if _is_cnn else -1
            _xe = _mp.x_end_idx   if _is_cnn else -1
            _ys = _mp.y_start_idx if _is_cnn else -1
            _ye = _mp.y_end_idx   if _is_cnn else -1
            jax.debug.print(
                "BT2 rank={} layer={} total_events={} nb_prev={} x=[{},{}] y=[{},{}]",
                mpi_config.rank, mpi_config.layer_idx,
                iteration - mpi_config.nb_previous,  # subtract END_SIGNALs from total loop iters
                mpi_config.nb_previous,
                _xs, _xe, _ys, _ye,
            )

        if layer_idx != last_layer:
            mpi_config.forward_send(END_SIGNAL, iteration)
            if len(mpi_config.res_connect_next) > 0:
                mpi_config.residual_send_end_signal(END_SIGNAL)
            # jax.debug.print("rank {} finished forward pass, starting backward pass", mpi_config.rank)

        return layer_input, neuron_states, iteration-1, buffer

    # jax.debug.print("rank {} data has shape {}", rank, batch_data.shape)

    # Per-sample neuron dropout: static gate (params/grad/layer_idx all static at trace).
    _drop_p = params.dropout[layer_idx] if getattr(params, "dropout", None) is not None else 0.0
    _apply_dropout = grad and (layer_idx != 0) and (layer_idx != last_layer) and (_drop_p > 0.0)

    # Loop over batches, accumulate output values and return them
    @jit
    def loop_over_batches(carry_key, x):
        neuron_states = empty_neuron_states
        if _apply_dropout:
            # Fresh Bernoulli keep-mask per sample, fixed across the sample's events.
            carry_key, sample_key = jax.random.split(carry_key)
            keep = jax.random.bernoulli(sample_key, p=1.0 - _drop_p,
                                        shape=neuron_states.values.shape).astype(neuron_states.values.dtype)
            if params.dropout_invert_scaling:
                keep = keep / (1.0 - _drop_p)
            neuron_states = neuron_states.replace(dropout_mask=keep)
        if layer_idx==0:
            iterations, buffer = input_layer(x)
            layer_input, new_neuron_states = jnp.zeros(()), neuron_states
        else:
            layer_input, new_neuron_states, iterations, buffer = other_layers(neuron_states)
        # Barrier between samples prevents events from bleeding across sample boundaries
        # when a layer has multiple senders and the receiver uses MPI.ANY_SOURCE.
        mpi4jax.barrier(comm=mpi_config.comm)
        return carry_key, (new_neuron_states.values, iterations, new_neuron_states, buffer)

    _, (all_outputs, all_iterations, all_neuron_states, buffer) = jax.lax.scan(loop_over_batches, key, batch_data)

    # Synchronize all ranks before starting the backward pass
    mpi4jax.barrier(comm=mpi_config.comm)

    return all_outputs, all_iterations, all_neuron_states, buffer

#region Conv computation
@partial(jax.jit, static_argnames=['params', 'mpi_config', 'grad',])
def conv_layer_computation(params, mpi_config, key, neuron_idx, layer_input, weights, neuron_states, iteration=0, grad=False, is_residual=False):
    '''
    Apply the convolution for an incoming event in the event-driven manner described in "Optimizing event-based neural networks on digital neuromorphic architecture: a comprehensive design space exploration"
    This convolution only supports 'SAME' padding scheme with stride 1

    weights: (out_ch, in_ch, k_h, k_w)

    Model parallelism: when mpi_config.model_part is a CNN_layer_Partition, each rank owns a sub-region
    (c_start:c_end+1, x_start:x_end+1, y_start:y_end+1) of the output. All ranks receive every input event
    but only compute and emit outputs for their owned region. The full values/thresholds/activity arrays
    are kept at their original size; only the owned sub-region is updated. No halo exchange is needed
    because every rank independently accumulates partial sums for its own output positions.

    Splits supported:
      - Output channel split: each rank owns a contiguous range of output channels. Input channel
        splits are not supported here (would require allreduce across ranks sharing input channels).
      - X (height) split: each rank owns a contiguous range of output rows.
      - Y (width) split: each rank owns a contiguous range of output columns.
      - Any combination of the above.
    '''
    out_ch, in_ch, k_h, k_w = weights.shape
    c, x, y = neuron_idx
    pad_value = jnp.asarray(-10000.0, dtype=neuron_states.values.dtype)
    pad_h, pad_w = neuron_states.padding
    event_pad_h, event_pad_w = k_h - 1 - pad_h, k_w - 1 - pad_w
    H = neuron_states.values.shape[1] - 2 * event_pad_h
    W = neuron_states.values.shape[2] - 2 * event_pad_w

    # Determine owned region from model partition (static at trace time)
    model_part = mpi_config.model_part
    is_partitioned = isinstance(model_part, CNN_layer_Partition)
    if is_partitioned:
        c_start = model_part.c_start_idx
        c_end   = model_part.c_end_idx
        x_start = model_part.x_start_idx
        x_end   = model_part.x_end_idx
        y_start = model_part.y_start_idx
        y_end   = model_part.y_end_idx
    else:
        c_start, c_end = 0, out_ch - 1
        x_start, x_end = 0, H - 1
        y_start, y_end = 0, W - 1

    sr = params.sync_rate if isinstance(params.sync_rate, int) else params.sync_rate[mpi_config.layer_idx]
    if sr == 10000:
        if neuron_states.pooling != "":
            C_f, H_f, W_f = params.flat_layer_sizes[mpi_config.layer_idx]
            event_array_size = C_f * H_f * W_f
        else:
            event_array_size = out_ch * H * W
    else:
        event_array_size = out_ch * k_h * k_w

    @jit
    def regular_input(neuron_states):
        # Step 1: Multiply the input value by the flipped kernel to obtain partial output values
        # layer_input is scalar for event-driven conv; scalar multiply is cheaper than dot.
        partial_activations = layer_input * jnp.flip(weights[:, c, :, :], axis=(1, 2)) # Shape (out_ch, k_h, k_w)

        # Step 2: Build masks for owned region — zero out kernel positions outside this rank's slice.
        # Kernel position (oc, kx, ky) maps to output padded position (x + kx, y + ky) and output channel oc.
        # event_pad_h/w and H/W are captured from the outer scope (computed once, not per event).

        # Output channel mask: kernel rows for owned output channels only
        oc_indices = jnp.arange(out_ch)  # (out_ch,)
        c_mask = (oc_indices >= c_start) & (oc_indices <= c_end)  # (out_ch,)
        partial_activations = partial_activations * c_mask[:, None, None]

        # X mask: kernel rows that land in the owned output-x range (in padded coordinates)
        kernel_x_pos = x + jnp.arange(k_h)  # padded output x positions, shape (k_h,)
        x_mask = (kernel_x_pos >= x_start + event_pad_h) & (kernel_x_pos <= x_end + event_pad_h)  # (k_h,)
        partial_activations = partial_activations * x_mask[None, :, None]

        # Y mask: kernel columns that land in the owned output-y range (in padded coordinates)
        kernel_y_pos = y + jnp.arange(k_w)  # padded output y positions, shape (k_w,)
        y_mask = (kernel_y_pos >= y_start + event_pad_w) & (kernel_y_pos <= y_end + event_pad_w)  # (k_w,)
        partial_activations = partial_activations * y_mask[None, None, :]

        # Input channel mask: skip this event if input channel c is not handled by this rank.
        # For output-channel / spatial splits the weights cover all input channels, so no masking needed.
        # For input-channel splits (not currently supported), the caller would need to restructure weights.

        values_padded = neuron_states.values
        thresholds_padded = neuron_states.thresholds

        start_indices = (0, x, y)
        slice_shape = partial_activations.shape  # (out_ch, k_h, k_w)

        current_values_sliced = jax.lax.dynamic_slice(values_padded, start_indices, slice_shape)
        thresholds_sliced = jax.lax.dynamic_slice(thresholds_padded, start_indices, slice_shape)

        padding_mask = jnp.where(current_values_sliced == pad_value, 0.0, 1.0)

        # Step 3: Add the partial output values to the current values
        activations = (current_values_sliced + partial_activations) * padding_mask
        updated_values_slice = activations

        # Step 4: Apply sync rate
        activity_slice = jax.lax.dynamic_slice(neuron_states.output_activity, start_indices, slice_shape)
        ne_activity_slice = activity_slice + 1
        sr = params.sync_rate
        sr = sr if isinstance(sr, int) else sr[mpi_config.layer_idx]
        activations = jnp.where(ne_activity_slice >= sr, activations, 0.0)

        # Step 5: Apply activation function (ReLU / threshold)
        activated_output = activation_func(thresholds_sliced, activations)

        # Step 6: Apply firing number — owned region only fires top-k
        f_nb = params.firing_nb
        k = f_nb if isinstance(f_nb, int) else f_nb[mpi_config.layer_idx]
        activated_output = keep_top_k(activated_output, k, max_kernel=params.max_kernel)

        # Zero out any activations outside owned region (in case partial_activations mask was not enough)
        activated_output = activated_output * c_mask[:, None, None] * x_mask[None, :, None] * y_mask[None, None, :]

        # Step 7: Update activity counter
        activation_mask = jnp.where(activated_output > 0, 0.0, 1.0)
        new_activity_slice = ne_activity_slice * activation_mask

        new_output_activity = jax.lax.dynamic_update_slice(
            neuron_states.output_activity,
            new_activity_slice,
            start_indices
        )

        reset = params.restrict
        if not isinstance(reset, int):
            reset = reset[mpi_config.layer_idx]
        # Step 8: Apply restriction
        penalty = jax.lax.cond( reset <= 0,
                                lambda _: activated_output,
                                lambda _: activated_output*reset, None)

        # Step 9: Compute remaining values
        remaining_value = updated_values_slice - penalty

        remaining_value = jnp.where(padding_mask == 0, pad_value, remaining_value)
        values_padded = jax.lax.dynamic_update_slice(values_padded, remaining_value, start_indices)
        new_values = values_padded

        # Step 10: Apply pooling and compute output events
        nb_valid_elements, out_events, unpooled_coords, unpooled_vals = output_to_event_array_with_pooling(activated_output,
                                                                       start_indices,
                                                                       (out_ch, H, W),
                                                                       (event_pad_h, event_pad_w),
                                                                       neuron_states.pooling,
                                                                       neuron_states.pool_size,
                                                                       neuron_states.pool_stride,
                                                                       mpi_config.rank)

        # Sync layer (sr==10000) never emits per-event (the sync gate above zeros everything
        # until the END_SIGNAL burst in last_input). Pad this per-event buffer up to the
        # full-layer event_array_size so all jax.lax.cond branches share one static shape.
        if sr == 10000 and neuron_states.pooling == "":
            out_events = jnp.pad(out_events, ((0, event_array_size - out_events.shape[0]), (0, 0)), constant_values=-2.0)

        # Bug Test 1: assert all fired output events have coords inside the owned region.
        if _DEBUG_LEVEL >= 1 and is_partitioned:
            valid_mask = jnp.arange(out_events.shape[0]) < nb_valid_elements
            out_c_ok = jnp.all(jnp.where(valid_mask, (out_events[:, 0] >= c_start) & (out_events[:, 0] <= c_end), True))
            out_x_ok = jnp.all(jnp.where(valid_mask, (out_events[:, 1] >= x_start) & (out_events[:, 1] <= x_end), True))
            out_y_ok = jnp.all(jnp.where(valid_mask, (out_events[:, 2] >= y_start) & (out_events[:, 2] <= y_end), True))
            jax.debug.print(
                "BT1 rank={} layer={} in=(c={},x={},y={}) fired={} c_ok={} x_ok={} y_ok={}",
                mpi_config.rank, mpi_config.layer_idx, c, x, y,
                nb_valid_elements, out_c_ok, out_x_ok, out_y_ok,
            )

        if grad:
            # Step 11: Update gradient tracking state
            valid_els = jnp.where(unpooled_vals != 0, 1, 0)
            new_weight_res = neuron_states.weight_res.at[   unpooled_coords[:, 0],
                                                            c,
                                                            unpooled_coords[:, 1]-x+event_pad_h,
                                                            unpooled_coords[:, 2]-y+event_pad_w
                                                        ].add(valid_els)

            new_layer_activity = neuron_states.layer_activity.at[   unpooled_coords[:, 0],
                                                                    unpooled_coords[:, 1],
                                                                    unpooled_coords[:, 2]
                                                                ].add(jnp.where(unpooled_vals != 0, 1, 0))

            # Record the max winning (pooled) value per pre-pool cell. This is the
            # signal used by the backward pass to route the max-pool gradient to the
            # true argmax cell, instead of the most-frequent winner (layer_activity).
            new_output_vector = neuron_states.output_vector.at[ unpooled_coords[:, 0],
                                                                unpooled_coords[:, 1],
                                                                unpooled_coords[:, 2]
                                                            ].max(jnp.where(unpooled_vals != 0, unpooled_vals, 0.0))

            input_act = neuron_states.input_activity
            new_input_activity = jax.lax.cond(nb_valid_elements > 0, lambda _: input_act.at[neuron_idx].add(1), lambda _: input_act, None)
            new_input_residuals = neuron_states.input_residuals.at[tuple(neuron_idx)].add(layer_input)
        else:
            new_input_residuals = neuron_states.input_residuals
            new_input_activity = neuron_states.input_activity
            new_layer_activity = neuron_states.layer_activity
            new_weight_res = neuron_states.weight_res
            new_output_vector = neuron_states.output_vector

        new_neuron_states = neuron_states.replace(
            values=new_values,
            input_residuals=new_input_residuals,
            input_activity=new_input_activity,
            layer_activity=new_layer_activity,
            output_activity=new_output_activity,
            output_vector=new_output_vector,
            weight_res=new_weight_res,)

        return nb_valid_elements, out_events, new_neuron_states

    @jit
    def last_input(neuron_states):
        if sr != 10000:
            return jnp.array(0), jnp.zeros((event_array_size, 4)), neuron_states

        # For full sync case, fire all neurons that are above the threshold.
        # values/thresholds are stored pre-padded (border = pad sentinel); operate on the
        # valid (unpadded) region only, so the fired events, layer_activity and the emitted
        # event buffer are all in the unpadded (out_ch, H, W) frame.
        valid_slice = (0, event_pad_h, event_pad_w)
        valid_shape = (out_ch, H, W)
        neuron_val = jax.lax.dynamic_slice(neuron_states.values, valid_slice, valid_shape)
        thresholds = jax.lax.dynamic_slice(neuron_states.thresholds, valid_slice, valid_shape)
        activated_output = activation_func(thresholds, neuron_val)

        # Mask to owned region only (unpadded output coords)
        if is_partitioned:
            c_indices = jnp.arange(activated_output.shape[0])
            x_indices = jnp.arange(activated_output.shape[1])
            y_indices = jnp.arange(activated_output.shape[2])
            region_mask = (
                (c_indices >= c_start)[:, None, None] &
                (c_indices <= c_end)[:, None, None] &
                (x_indices >= x_start)[None, :, None] &
                (x_indices <= x_end)[None, :, None] &
                (y_indices >= y_start)[None, None, :] &
                (y_indices <= y_end)[None, None, :]
            )
            activated_output = jnp.where(region_mask, activated_output, 0.0)

        # Compute remaining values and write them back into the pre-padded values array
        remaining_value = neuron_val - activated_output
        new_values = jax.lax.dynamic_update_slice(neuron_states.values, remaining_value, valid_slice)
        nb_valid_elements, out_events, unpooled = full_matrix_to_event_array_with_pooling(activated_output, activated_output.shape,
                                                                                          neuron_states.pooling, neuron_states.pool_size,
                                                                                          neuron_states.pool_stride, mpi_config.rank)

        # Add unpooled values to layer activity
        mask = unpooled != 0
        new_layer_activity = jnp.where(
            mask,
            neuron_states.layer_activity + unpooled,
            neuron_states.layer_activity
        )

        new_neuron_states = neuron_states.replace(
            values=new_values,
            input_activity=jnp.ones(neuron_states.input_activity.shape, dtype=int),
            layer_activity=new_layer_activity,)

        return nb_valid_elements, out_events, new_neuron_states

    def residual_input(neuron_states):
        """Identity skip: add layer_input directly to neuron_states.values[c, x, y].
        No conv weights, no thresholds, no firing — just accumulation."""
        c, x, y = neuron_idx
        new_values = neuron_states.values.at[c, x, y].add(layer_input)
        if grad:
            new_layer_activity = neuron_states.layer_activity.at[c, x, y].add(1.0)
        else:
            new_layer_activity = neuron_states.layer_activity
        new_ns = neuron_states.replace(values=new_values, layer_activity=new_layer_activity)
        # No new fired events — firing happens when a future regular event crosses threshold.
        return jnp.array(0), jnp.zeros((event_array_size, 4)), new_ns

    nb_valid_elements, out_events, neuron_states = jax.lax.cond(
        is_residual,
        residual_input,
        lambda ns: jax.lax.cond(jnp.any(neuron_idx < 0), last_input, regular_input, ns),
        neuron_states,
    )

    return nb_valid_elements, out_events, neuron_states