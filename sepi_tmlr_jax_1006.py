"""
sepi_tmlr_jax_1006.py  
12062026
─────────────────────────────────────────────────────────────────────────────
Frame-based N-MNIST inference using Laurent's JAX + mpi4jax infrastructure.

Architecture
────────────
N MPI ranks form a pipeline, one rank per layer:
  rank 0        — input feeder: reads event frames, sends (neuron_idx, value) pairs
  rank 1 … N-2  — hidden LIF layers: integrate input, fire binary spikes
  rank N-1      — output accumulator: integrates spikes, argmax for class

Number of ranks = len(layer_sizes).  Examples:
  layer_sizes: [2312, 128, 10]         →  mpirun -n 3
  layer_sizes: [2312, 64, 64, 64, 10]  →  mpirun -n 5

Extends async_MLP.py predict() with:
  1. Frame-end sentinel -3  →  apply exp(-dt/tau_m) membrane decay per frame
  2. Binary spikes (1.0) + hard reset  →  matches asynctorch LIF training
  3. Momentum ordering of output spikes  →  sort by pre-spike membrane desc
  4. Optional refractory period  →  (V+I)*~is_refrac after firing
  5. Optional synchronous first layer  →  fire once per frame at -3 boundary

Run (adjust -n to match len(layer_sizes)):
  JAX_PLATFORMS=cpu mpirun -n <N> python sepi_tmlr_jax_1006.py \\
      --config sepi_tmlr_conf_1006.yaml [--max_samples N] [--sync_first_layer]

Accuracy on full 10k N-MNIST test set — [2312,128,10], fgs=128, momentum:
  async+async,    use_refrac=False : 96.19%
  async+async,    use_refrac=True  : 96.43%
  sync-all-frames, use_refrac=True : 96.29%
  sync-frame1+async, use_refrac=True : 96.43%
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.pop("JAX_TRACEBACK_FILTERING", None)

from mpi4py import MPI
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np
import time, sys, yaml, argparse
from tqdm import tqdm

import mpi4jax
from mpi4jax import send, recv

import tonic
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader

from other_helpers.helpers import NeuronStates, keep_top_k, Params

jax.config.update("jax_debug_nans", True)

# ── MPI globals (set in main) ──────────────────────────────────────────────────
comm             = None   # MPI communicator
rank             = None   # this process's MPI rank (0=input, 1…N-2=hidden, N-1=output)
size             = None   # total number of MPI processes
layer_idx        = None   # same as rank; which layer this process owns
process_per_layer = 1     # one MPI rank per layer (no model parallelism)
last_layer       = None   # rank index of the output layer (= n_layers - 1)

# ── Frame-based sentinels (plain Python ints for JIT-static comparisons) ───────
_FRAME_END  = -3   # inter-frame boundary: trigger membrane decay and fire
_SAMPLE_END = -1   # end of sample: exit while-loop
_PADDING    = -2   # padding rows in fixed-length event arrays; skipped by input_layer


# ── Params builder ─────────────────────────────────────────────────────────────
def make_params(cfg):
    """
    Build a frozen Params dataclass from a YAML config dict.

    Parameters
    ----------
    cfg : dict
        YAML config with keys:
          layer_sizes       : list[int]  — neuron counts per layer, e.g. [2312, 128, 10]
          threshold         : float      — LIF firing threshold (default 0.3)
          firing_nb         : int        — max spikes per firing event; -1 = no cap
          forward_group_size: int        — sync_rate: fire every N events (async hidden layer)
          tau_m             : float      — membrane time constant in ms (default 1000.0)
          time_window       : float      — frame duration = dt for decay (default 10000.0)
          use_refrac        : bool       — apply refractory: membrane = (V+I)*~is_refrac
          weight_file       : str        — path to .npz weight file

    Returns
    -------
    Params
        Frozen dataclass consumed by layer_computation() and predict().
    """
    return Params(
        dataset          = "nmnist",
        random_seed      = 42,
        layer_sizes      = tuple(cfg["layer_sizes"]),
        init_thresholds  = float(cfg.get("threshold",          0.3)),
        num_epochs       = 0,
        learning_rate    = 0.0,
        batch_size       = 1,
        load_file        = True,
        shuffle_activations = False,   # momentum ordering handled in format_frames_as_events
        restrict         = 0,          # unused: hard reset is done explicitly
        firing_nb        = int(cfg.get("firing_nb",           -1)),
        sync_rate        = int(cfg.get("forward_group_size", 8)),
        max_nonzero      = 0,
        shuffle_input    = False,      # input ordering done in format_frames_as_events
        threshold_lr     = 0.0,
        sparsity_impact  = (0.0, 0.0, 0.0),
        w_reg            = 0.0,
        rerun            = "",
        top_weights      = -1,         # -1 = use all weights (no sparsification)
        history_size     = 0,
        tau_m            = float(cfg.get("tau_m",        1000.0)),
        dt               = float(cfg.get("time_window", 10000.0)),
        use_refrac       = bool(cfg.get("use_refrac",   False)),
    )


# ── Binary momentum output events ─────────────────────────────────────────────
@partial(jax.jit, static_argnames=["n_neurons", "firing_nb_static"])
def binary_output_events(activations, thresholds, n_neurons, firing_nb_static):
    """
    Convert membrane potentials to a padded (n_neurons, 2) spike array.

    Produces momentum-ordered binary spikes: neurons with the highest pre-spike
    membrane fire first, matching asynctorch training convention.

    Parameters
    ----------
    activations : jnp.ndarray, shape (n_neurons,)
        Current membrane potentials.
    thresholds : jnp.ndarray, shape (n_neurons,)
        Per-neuron firing thresholds.
    n_neurons : int (static)
        Number of neurons; also the output array length.
    firing_nb_static : int (static)
        Maximum spikes to emit per firing event. -1 means no cap.
        Values < 32 sharply reduce accuracy (see firing_nb sweep results).

    Returns
    -------
    pairs : jnp.ndarray, shape (n_neurons, 2)
        Each row is [neuron_index, 1.0] for fired neurons (momentum-ordered),
        padded with [-2, -2] for unfired slots.
    effective_count : jnp.ndarray, scalar int
        Number of actual spikes emitted (≤ firing_nb_static if capped).
    """
    fired_mask  = activations > thresholds
    fired_count = jnp.sum(fired_mask.astype(jnp.int32))

    # Sort by -activations: highest membrane fires first (momentum ordering)
    sort_key   = jnp.where(fired_mask, -activations, 1e10)
    sorted_idx = jnp.argsort(sort_key)   # shape: (n_neurons,)

    if firing_nb_static >= 0:
        effective_count = jnp.minimum(fired_count, firing_nb_static)
    else:
        effective_count = fired_count

    slots   = jnp.arange(n_neurons)
    out_idx = jnp.where(slots < effective_count, sorted_idx, jnp.full((), -2))
    out_val = jnp.where(out_idx >= 0, 1.0, -2.0)   # binary spikes (1.0), pad (-2.0)
    pairs   = jnp.stack([out_idx.astype(jnp.float32), out_val], axis=1)
    return pairs, effective_count


# ── layer_computation ──────────────────────────────────────────────────────────
@partial(jax.jit, static_argnames=["params", "grad"])
def layer_computation(params, key, neuron_idx, layer_input, weights,
                      neuron_states, is_refrac, iteration=0, grad=False):
    """
    One-event update for a single layer; called inside the async while-loop.

    Routing by neuron_idx
    ─────────────────────
    neuron_idx >= 0   →  integrate: membrane += W[neuron_idx] * layer_input
    neuron_idx == -3  →  frame-end: apply decay, fire, reset, build output spikes
    neuron_idx == -1  →  sample-end: no integration (loop exits after this call)

    Parameters
    ----------
    params : Params (static / frozen dataclass)
        Hyperparameters. Frozen so JAX recompiles for each unique config.
    key : jax.random.KeyArray
        PRNG key (currently unused; kept for API compatibility with Laurent's code).
    neuron_idx : jnp.ndarray, scalar int
        Index of the pre-synaptic neuron that fired, or a sentinel value.
    layer_input : jnp.ndarray, scalar float
        Spike value from the sending neuron (1.0 for binary spikes).
    weights : jnp.ndarray, shape (in_size, out_size)
        Synaptic weight matrix for this layer.
    neuron_states : NeuronStates
        Current membrane potentials, thresholds, and bookkeeping state.
    is_refrac : jnp.ndarray, shape (n_out,), dtype bool
        True for neurons that fired in the previous frame and are refractory.
        Ignored when params.use_refrac=False (compiled away at trace time).
    iteration : int, default 0
        Event counter; used to decide whether to fire based on sync_rate.
    grad : bool (static), default False
        Unused gradient flag; kept for API compatibility.

    Returns
    -------
    valid_elements : jnp.ndarray, scalar int
        Number of output spikes emitted this step.
    pairs : jnp.ndarray, shape (n_out, 2)
        Spike array [(fired_idx, 1.0), ...] padded with (-2, -2).
    new_neuron_states : NeuronStates
        Updated membrane potentials and bookkeeping.
    new_is_refrac : jnp.ndarray, shape (n_out,), dtype bool
        Updated refractory mask (set after firing, cleared at frame-end).
    """
    # ── Integrate (skip for any sentinel) ────────────────────────────────────
    filtered_weights = keep_top_k(weights[neuron_idx], params.top_weights, apply_abs=True)
    raw_activations = jax.lax.cond(
        neuron_idx < 0,
        lambda _: neuron_states.values,
        lambda _: neuron_states.values + jnp.dot(layer_input, filtered_weights),
        None
    )
    # Refractory mask: (V + I) * ~is_refrac  —  matches asynctorch training model.
    # Python-level branch so JAX compiles two separate graphs (no runtime cost).
    if params.use_refrac:
        activations = raw_activations * (~is_refrac)
    else:
        activations = raw_activations

    n_out = params.layer_sizes[layer_idx]

    # ── Output layer: accumulate total signal; no inter-frame decay ───────────
    # The output layer classifies by argmax of total accumulated membrane —
    # decaying it would attenuate late-frame evidence and hurt accuracy.
    @jit
    def last_layer_case(_):
        """Accumulate membrane without decay; output is used for argmax only."""
        dummy_out = jnp.zeros((n_out, 2))
        ns = neuron_states.replace(values=activations)
        return jnp.array(0), dummy_out, ns, is_refrac   # pass is_refrac unchanged

    # ── Hidden layer: async fire + hard reset + inter-frame decay ────────────
    @jit
    def hidden_layer_case(_):
        """
        Fire when sync_rate events have passed or at a sentinel boundary.
        Hard-reset fired neurons, apply inter-frame decay at -3 boundary,
        update refractory mask.
        """
        # Fire condition: enough events since last fire, OR forced at frame-end.
        # Use == _FRAME_END (not neuron_idx < 0) so sample-end (-1) does NOT trigger a fire.
        fire = (iteration - neuron_states.last_sent_iteration) >= params.sync_rate
        fire = jnp.logical_or(fire, neuron_idx == _FRAME_END)

        fired_mask = jnp.logical_and(activations > neuron_states.thresholds, fire)

        # Build momentum-ordered binary spike array
        pairs, valid_elements = binary_output_events(
            activations, neuron_states.thresholds, n_out, params.firing_nb
        )
        # Suppress spikes if not in a firing window
        pairs          = jax.lax.cond(fire, lambda _: pairs,
                                      lambda _: jnp.full_like(pairs, -2.0), None)
        valid_elements = jax.lax.cond(fire, lambda _: valid_elements,
                                      lambda _: jnp.array(0), None)

        # Hard reset: set fired neurons to 0
        new_vals = jnp.where(fired_mask, 0.0, activations)

        # Apply inter-frame decay only at the -3 frame boundary
        decay    = jnp.exp(-params.dt / params.tau_m)
        new_vals = jax.lax.cond(
            neuron_idx == _FRAME_END,
            lambda _: new_vals * decay,
            lambda _: new_vals,
            None
        )

        # Refractory update (Python-level branch for JIT efficiency)
        if params.use_refrac:
            # Mark fired neurons as refractory for the next event
            new_is_refrac = jax.lax.cond(
                fire,
                lambda _: is_refrac | fired_mask,
                lambda _: is_refrac,
                None
            )
            # Clear refractory at frame-end so the next frame starts fresh
            new_is_refrac = jax.lax.cond(
                neuron_idx == _FRAME_END,
                lambda _: jnp.zeros_like(is_refrac),
                lambda _: new_is_refrac,
                None
            )
        else:
            new_is_refrac = is_refrac

        new_last_sent = jax.lax.cond(
            fire, lambda _: iteration, lambda _: neuron_states.last_sent_iteration, None
        )
        ns = neuron_states.replace(values=new_vals, last_sent_iteration=new_last_sent)
        return valid_elements, pairs, ns, new_is_refrac

    return jax.lax.cond(layer_idx == last_layer, last_layer_case, hidden_layer_case, None)


# ── predict (extended from Laurent's async_MLP.py) ────────────────────────────
@partial(jax.jit, static_argnames=["params", "grad", "sync_first_layer", "sync_first_frame"])
def predict(params, key, weights, empty_neuron_states, batch_data,
            grad=False, sync_first_layer=False, sync_first_frame=False):
    """
    JIT-compiled inference for a batch of samples via MPI pipeline.

    Each rank runs a different inner function determined at trace time:
      rank 0        →  input_layer    (sends events downstream)
      rank 1        →  sync_lif_layer (if sync_first_layer or sync_first_frame)
                    or other_layers   (default async)
      rank 2…N-2    →  other_layers   (hidden layers)
      rank N-1      →  other_layers   (output accumulator, last_layer)

    Parameters
    ----------
    params : Params (static)
        Frozen hyperparameters; changes force recompilation.
    key : jax.random.KeyArray
        PRNG key (unused; kept for API compatibility).
    weights : jnp.ndarray, shape (in_size, out_size)
        Weight matrix for this rank's layer (rank 0 uses dummy zeros).
    empty_neuron_states : NeuronStates
        Fresh zero-initialized state; reset at the start of every sample.
    batch_data : jnp.ndarray, shape (B, max_events, 2)
        Event array for rank 0; other ranks pass zeros (ignored).
    grad : bool (static), default False
        Unused; kept for API compatibility with Laurent's training code.
    sync_first_layer : bool (static), default False
        If True, rank 1 uses synchronous LIF for ALL frames: accumulates all
        frame events then fires once at each -3 boundary.
    sync_first_frame : bool (static), default False
        If True, rank 1 uses synchronous LIF only for the FIRST frame, then
        switches to async (fire every sync_rate events) for frames 2–N.
        Mutually exclusive with sync_first_layer.

    Returns
    -------
    all_outputs : jnp.ndarray, shape (B, n_out)
        Accumulated output membrane per sample (rank 2 only; zeros elsewhere).
    all_iters : jnp.ndarray, shape (B,)
        Number of while-loop iterations per sample.
    all_ns : NeuronStates
        Neuron states after the last sample in the batch.
    buffer : jnp.ndarray, shape (B, 1, 2)
        Unused output buffer (kept for scan shape consistency).
    all_stat_a : jnp.ndarray, shape (B,)
        Per-sample spike count:
          rank 0      → input events sent
          rank 1      → hidden spikes fired (sync or async)
          rank 2…N-1  → 0 (intermediate hidden and output layers do not fire via this counter)
    all_stat_b : jnp.ndarray, shape (B,)
        Per-sample event count received by this rank (neuron_idx >= 0 only).
    """

    @jit
    def input_layer(args):
        """
        Rank 0: iterate over the fixed-length event array and send non-padding
        rows downstream.  Sends a final (-1, -1) sample-end sentinel.

        Parameters
        ----------
        args : tuple (NeuronStates, jnp.ndarray)
            neuron_states : unused (rank 0 has no weights to integrate)
            x             : jnp.ndarray, shape (max_events, 2) — event array

        Returns
        -------
        6-tuple matching the shared scan output structure:
          (zeros, neuron_states, 0, zeros_buffer, events_sent, 0)
        """
        neuron_states, x = args
        x_p = jnp.array(x)

        # Count rows that are NOT (-2,-2) padding; includes real events + -3 sentinels
        def not_padding(row):
            return row != _PADDING   # element-wise comparison
        mask = jax.vmap(not_padding)(x_p)
        loop_iterations = (jnp.count_nonzero(mask) / 2).astype(int)

        def send_input(i, carry):
            """Send the i-th event row (neuron_idx, value) to rank 1."""
            send(x_p[i], dest=rank + process_per_layer, tag=0, comm=comm)
            return i

        jax.lax.fori_loop(0, loop_iterations, send_input, 0)
        # Signal end-of-sample so rank 1's while-loop exits
        send(jnp.array([-1.0, -1.0]), dest=rank + process_per_layer, tag=0, comm=comm)
        # stat_a = total events sent (pixel events + frame separators), stat_b = 0
        return jnp.zeros(()), neuron_states, 0, jnp.zeros((1, 2)), loop_iterations, jnp.array(0, dtype=jnp.int32)

    # ── Synchronous LIF layer (rank 1 when sync_first_layer or sync_first_frame) ─
    @jit
    def sync_lif_layer(args):
        """
        Synchronous-first LIF processing for rank 1.

        Two modes, selected at trace time via Python-level branch:

        sync_first_layer=True  (all frames sync)
          State: (membrane, neuron_idx, event_count, spike_count, is_refrac)
          Every frame: accumulate all events, fire once at -3, decay residual.

        sync_first_frame=True  (first frame sync, rest async)
          State adds: first_frame_done (bool), events_since_fire (int)
          Frame 1  : no mid-frame firing; fire + decay once at first -3.
          Frames 2+: fire every params.sync_rate events OR at -3 (async mode).
          The switch happens automatically when first_frame_done flips to True.

        Firing order (both modes): fire on raw accumulated membrane (pre-decay),
        then apply inter-frame decay to the reset residual for the next frame.
        Decaying BEFORE firing kills the signal (decay ≈ 4.5e-5 for default
        tau_m=1000, dt=10000).

        Parameters
        ----------
        args : tuple (NeuronStates, ignored)
            neuron_states.values     : initial membrane (zeros for fresh sample)
            neuron_states.thresholds : per-neuron thresholds

        Returns
        -------
        6-tuple matching the shared scan output structure:
          (zeros, new_ns, 0, zeros_buffer, spike_count, event_count)
        """
        neuron_states, _ = args
        membrane   = neuron_states.values       # shape (n_hidden,)
        thresholds = neuron_states.thresholds   # shape (n_hidden,)
        n_hidden   = membrane.shape[0]
        # Inter-frame decay factor: exp(-dt / tau_m)
        decay      = jnp.exp(-params.dt / params.tau_m)

        # ── Shared helper: fire + reset + decay + send spikes + send -3 ─────────
        def fire_and_send(membrane_in, is_refrac_in):
            """
            Fire neurons, hard-reset, decay residual, send spikes downstream.

            Called at every -3 boundary in both sync modes.

            Parameters
            ----------
            membrane_in : jnp.ndarray (n_hidden,)
                Accumulated membrane at the moment of firing (pre-decay).
            is_refrac_in : bool array (n_hidden,)
                Refractory mask from the previous frame.

            Returns
            -------
            carry_mem     : jnp.ndarray (n_hidden,) — decayed residual for next frame
            fired_count   : int scalar              — number of spikes sent
            new_is_refrac : bool array (n_hidden,)  — refractory mask for next frame
            """
            if params.use_refrac:
                eff = membrane_in * (~is_refrac_in)
            else:
                eff = membrane_in
            # Fire on raw accumulated membrane (pre-decay)
            fired_mask  = eff > thresholds
            fired_count = jnp.sum(fired_mask.astype(jnp.int32))
            # Hard-reset fired neurons; decay the residual for the next frame
            reset_mem   = jnp.where(fired_mask, 0.0, eff)
            carry_mem   = reset_mem * decay
            # Momentum order: highest pre-spike membrane fires first
            sort_key    = jnp.where(fired_mask, -eff, 1e10)
            sorted_idx  = jnp.argsort(sort_key)

            def send_spike(i, _):
                """Send the i-th momentum-ordered spike (neuron_idx, 1.0) downstream."""
                send(jnp.stack([sorted_idx[i].astype(jnp.float32), jnp.array(1.0)]),
                     dest=rank + process_per_layer, tag=0, comm=comm)
                return None

            jax.lax.fori_loop(0, fired_count, send_spike, None)
            # Forward frame-end sentinel so downstream ranks apply their own decay
            send(jnp.array([-3.0, 0.0]), dest=rank + process_per_layer, tag=0, comm=comm)
            if params.use_refrac:
                new_is_refrac = fired_mask   # fired neurons blocked next frame
            else:
                new_is_refrac = is_refrac_in
            return carry_mem, fired_count, new_is_refrac

        # ── Mode A: all frames synchronous ───────────────────────────────────────
        # Python-level if: compiled away at trace time; two separate JAX graphs.
        if not sync_first_frame:

            def cond_a(state):
                """Exit when sample-end sentinel (-1) is received."""
                _, neuron_idx, _, _, _ = state
                return neuron_idx != _SAMPLE_END

            @jit
            def forward_pass_a(state):
                """
                All-frames-sync forward pass.

                State: (membrane, neuron_idx, event_count, spike_count, is_refrac)
                  membrane    : (n_hidden,)  — accumulated membrane
                  neuron_idx  : int scalar   — last received index
                  event_count : int scalar   — total pixel events received
                  spike_count : int scalar   — total spikes fired
                  is_refrac   : (n_hidden,) bool — refractory mask
                """
                membrane, _, event_count, spike_count, is_refrac = state

                (neuron_idx, layer_input) = recv(
                    jnp.zeros((2,)), source=rank - process_per_layer, tag=0, comm=comm
                )
                neuron_idx = neuron_idx.astype(int)

                # Accumulate pixel events; sentinels leave membrane unchanged
                new_membrane = jax.lax.cond(
                    neuron_idx >= 0,
                    lambda _: membrane + weights[neuron_idx] * layer_input,
                    lambda _: membrane,
                    None
                )

                @jit
                def on_frame_end(_):
                    """Fire once at -3, then carry decayed residual to next frame."""
                    carry, fired_count, new_is_refrac = fire_and_send(new_membrane, is_refrac)
                    return carry, fired_count, new_is_refrac

                @jit
                def no_frame_end(_):
                    """Non-sentinel: membrane updated, nothing to fire."""
                    return new_membrane, jnp.array(0, dtype=jnp.int32), is_refrac

                final_mem, n_spikes, new_is_refrac = jax.lax.cond(
                    neuron_idx == _FRAME_END, on_frame_end, no_frame_end, None
                )

                new_event_count = event_count + jnp.where(
                    neuron_idx >= 0, jnp.array(1, jnp.int32), jnp.array(0, jnp.int32)
                )
                new_spike_count = spike_count + n_spikes
                return final_mem, neuron_idx, new_event_count, new_spike_count, new_is_refrac

            init_a = (membrane,
                      jnp.array(0, dtype=jnp.int32),    # neuron_idx placeholder
                      jnp.array(0, dtype=jnp.int32),    # event_count
                      jnp.array(0, dtype=jnp.int32),    # spike_count
                      jnp.zeros(n_hidden, dtype=bool))  # is_refrac

            final_mem, _, event_count, spike_count, _ = jax.lax.while_loop(
                cond_a, forward_pass_a, init_a
            )

        # ── Mode B: first frame sync, subsequent frames async ─────────────────
        else:

            def cond_b(state):
                """Exit when sample-end sentinel (-1) is received."""
                _, neuron_idx, _, _, _, _, _ = state
                return neuron_idx != _SAMPLE_END

            @jit
            def forward_pass_b(state):
                """
                First-frame-sync forward pass.

                State: (membrane, neuron_idx, event_count, spike_count,
                        is_refrac, first_frame_done, events_since_fire)
                  membrane          : (n_hidden,)  — accumulated membrane
                  neuron_idx        : int scalar   — last received index
                  event_count       : int scalar   — total pixel events
                  spike_count       : int scalar   — total spikes fired
                  is_refrac         : (n_hidden,) bool — refractory mask
                  first_frame_done  : bool scalar  — True after first -3 is processed;
                                                     gate that enables async firing
                  events_since_fire : int scalar   — events since last firing window;
                                                     used for sync_rate check in async mode
                """
                membrane, _, event_count, spike_count, is_refrac, first_frame_done, events_since_fire = state

                (neuron_idx, layer_input) = recv(
                    jnp.zeros((2,)), source=rank - process_per_layer, tag=0, comm=comm
                )
                neuron_idx = neuron_idx.astype(int)

                # Integrate pixel events; sentinels leave membrane unchanged
                new_membrane = jax.lax.cond(
                    neuron_idx >= 0,
                    lambda _: membrane + weights[neuron_idx] * layer_input,
                    lambda _: membrane,
                    None
                )

                # Track events since last firing window (used for async sync_rate check)
                new_esf = events_since_fire + jnp.where(
                    neuron_idx >= 0, jnp.array(1, jnp.int32), jnp.array(0, jnp.int32)
                )

                # Fire conditions:
                #   - Frame 1 (first_frame_done=False): only fire at -3 boundary (sync)
                #   - Frames 2+ (first_frame_done=True): fire every sync_rate events
                #     OR at -3 boundary (async — matches layer_computation behavior)
                fire_by_rate = first_frame_done & (new_esf >= params.sync_rate)
                fire_at_end  = (neuron_idx == _FRAME_END)
                fire = fire_by_rate | fire_at_end

                @jit
                def do_fire(_):
                    """Fire neurons, send spikes, and if at -3 also forward sentinel."""
                    if params.use_refrac:
                        eff = new_membrane * (~is_refrac)
                    else:
                        eff = new_membrane
                    fired_mask  = eff > thresholds
                    fired_count = jnp.sum(fired_mask.astype(jnp.int32))
                    reset_mem   = jnp.where(fired_mask, 0.0, eff)

                    # Decay only at frame-end; mid-frame fires keep raw reset membrane
                    carry_mem = jax.lax.cond(
                        fire_at_end,
                        lambda _: reset_mem * decay,
                        lambda _: reset_mem,
                        None
                    )
                    sort_key   = jnp.where(fired_mask, -eff, 1e10)
                    sorted_idx = jnp.argsort(sort_key)

                    def send_spike(i, _):
                        """Send one momentum-ordered spike downstream."""
                        send(jnp.stack([sorted_idx[i].astype(jnp.float32), jnp.array(1.0)]),
                             dest=rank + process_per_layer, tag=0, comm=comm)
                        return None

                    jax.lax.fori_loop(0, fired_count, send_spike, None)

                    # Forward frame-end sentinel only when actually at -3
                    jax.lax.cond(
                        fire_at_end,
                        lambda _: send(jnp.array([-3.0, 0.0]),
                                       dest=rank + process_per_layer, tag=0, comm=comm),
                        lambda _: [],
                        None
                    )

                    if params.use_refrac:
                        # Set refractory for fired neurons; clear at frame-end for next frame
                        new_ref = is_refrac | fired_mask
                        new_ref = jax.lax.cond(
                            fire_at_end,
                            lambda _: jnp.zeros_like(is_refrac),
                            lambda _: new_ref,
                            None
                        )
                    else:
                        new_ref = is_refrac

                    return carry_mem, fired_count, new_ref

                @jit
                def no_fire(_):
                    """No firing this step: propagate -3 only if at frame-end."""
                    jax.lax.cond(
                        fire_at_end,
                        lambda _: send(jnp.array([-3.0, 0.0]),
                                       dest=rank + process_per_layer, tag=0, comm=comm),
                        lambda _: [],
                        None
                    )
                    # Apply decay even when no neurons fired (carry for next frame)
                    carry_mem = jax.lax.cond(
                        fire_at_end,
                        lambda _: new_membrane * decay,
                        lambda _: new_membrane,
                        None
                    )
                    return carry_mem, jnp.array(0, dtype=jnp.int32), is_refrac

                final_mem, n_spikes, new_is_refrac = jax.lax.cond(
                    fire, do_fire, no_fire, None
                )

                # After processing the first -3, switch to async mode for all later frames
                new_ffd = first_frame_done | fire_at_end

                # Reset counter after any firing event so sync_rate counts from scratch
                new_esf_out = jax.lax.cond(
                    fire,
                    lambda _: jnp.array(0, dtype=jnp.int32),
                    lambda _: new_esf,
                    None
                )

                new_event_count = event_count + jnp.where(
                    neuron_idx >= 0, jnp.array(1, jnp.int32), jnp.array(0, jnp.int32)
                )
                new_spike_count = spike_count + n_spikes

                return (final_mem, neuron_idx, new_event_count, new_spike_count,
                        new_is_refrac, new_ffd, new_esf_out)

            init_b = (membrane,
                      jnp.array(0, dtype=jnp.int32),    # neuron_idx placeholder
                      jnp.array(0, dtype=jnp.int32),    # event_count
                      jnp.array(0, dtype=jnp.int32),    # spike_count
                      jnp.zeros(n_hidden, dtype=bool),  # is_refrac
                      jnp.array(False),                  # first_frame_done
                      jnp.array(0, dtype=jnp.int32))    # events_since_fire

            final_mem, _, event_count, spike_count, _, _, _ = jax.lax.while_loop(
                cond_b, forward_pass_b, init_b
            )

        # ── Common exit ──────────────────────────────────────────────────────────
        # Signal end-of-sample to rank 2 so its while-loop also exits
        send(jnp.array([-1.0, -1.0]), dest=rank + process_per_layer, tag=0, comm=comm)

        new_ns = neuron_states.replace(values=final_mem)
        # stat_a = spike_count (hidden spikes fired), stat_b = event_count (pixels received)
        return (jnp.zeros(()), new_ns, jnp.array(0, dtype=jnp.int32),
                jnp.zeros((1, 2)), spike_count, event_count)

    @jit
    def other_layers(args):
        """
        Asynchronous event-driven processing for any non-input rank.

        Used by rank 1 (first hidden, when async), ranks 2…N-2 (intermediate
        hidden layers), and rank N-1 (output accumulator).  Each hidden rank
        forwards fired spikes and -3 sentinels downstream; the output rank
        (last_layer) only accumulates membrane for final argmax.

        Parameters
        ----------
        args : tuple (NeuronStates, ignored)
            neuron_states : initial state for this sample (values zeroed)

        Returns
        -------
        6-tuple:
          (zeros, neuron_states, iteration-1, buffer, spike_count, event_count)
          spike_count : total spikes fired (hidden ranks only; 0 for last_layer)
          event_count : total pixel events received (neuron_idx >= 0)
        """
        neuron_states, _ = args

        # Use actual neuron_states.values shape (not params.layer_sizes) so that
        # the traced-but-unused rank-0 path stays shape-consistent with its dummy
        # make_empty_states(1, 1) neuron_states (shape [1]).
        n_out = neuron_states.values.shape[0]

        def cond(state):
            """Continue loop until sample-end sentinel (-1) is received."""
            _, _, neuron_idx, _, _, _, _, _ = state
            return neuron_idx != _SAMPLE_END

        @jit
        def forward_pass(state):
            """
            Process one received event through layer_computation().

            State layout (8 elements):
              [0] layer_input   : float scalar      — last received spike value
              [1] neuron_states : NeuronStates       — membrane and bookkeeping
              [2] neuron_idx    : int scalar         — last received neuron index
              [3] iteration     : int scalar         — event counter
              [4] buffer        : float (1,2)        — unused shape-filler
              [5] is_refrac     : bool (n_out,)      — refractory mask
              [6] spike_count   : int scalar         — total spikes fired this sample
              [7] event_count   : int scalar         — total pixel events received

            Returns the same 8-element tuple with updated values.
            """
            layer_input, neuron_states, _, iteration, buffer, is_refrac, spike_count, event_count = state

            # Blocking receive from previous rank
            (neuron_idx, layer_input) = recv(
                jnp.zeros((2,)), source=rank - process_per_layer, tag=0, comm=comm
            )
            neuron_idx = neuron_idx.astype(int)

            loop_iterations, activated_output, new_ns, new_is_refrac = layer_computation(
                params, key, neuron_idx, layer_input, weights,
                neuron_states, is_refrac, iteration, grad
            )

            # Accumulate per-sample stats
            new_spike_count = spike_count + loop_iterations
            new_event_count = event_count + jnp.where(
                neuron_idx >= 0, jnp.array(1, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)
            )

            @jit
            def send_spikes(_):
                """
                Send fired spikes (loop_iterations of them) to the next rank.
                Only called when layer_idx != last_layer.
                """
                def send_one(i, _):
                    """Send activated_output[i] = (neuron_idx, 1.0) downstream."""
                    send(activated_output[i],
                         dest=rank + process_per_layer, tag=0, comm=comm)
                    return None
                jax.lax.fori_loop(0, loop_iterations, send_one, None)
                return []   # match send()'s [] return type in mpi4jax 0.8.0

            # Output layer (rank 2) never sends spikes; use cond for static dispatch
            jax.lax.cond(layer_idx == last_layer, lambda _: [], send_spikes, None)

            # Forward frame-end sentinel downstream (rank 1 only) so rank 2
            # can apply its own decay.  No-op at rank 2 (last_layer).
            jax.lax.cond(
                jnp.logical_and(layer_idx != last_layer, neuron_idx == _FRAME_END),
                lambda _: send(jnp.array([-3.0, 0.0]),
                               dest=rank + process_per_layer, tag=0, comm=comm),
                lambda _: [],
                operand=None
            )

            return layer_input, new_ns, neuron_idx, iteration + 1, buffer, new_is_refrac, new_spike_count, new_event_count

        # Initial while-loop state; neuron_idx=0 is a non-terminal placeholder
        init = (jnp.zeros(()), neuron_states, jnp.array(0, dtype=jnp.int32),
                jnp.array(0, dtype=jnp.int32), jnp.zeros((1, 2)),
                jnp.zeros(n_out, dtype=bool),      # is_refrac
                jnp.array(0, dtype=jnp.int32),     # spike_count
                jnp.array(0, dtype=jnp.int32))     # event_count

        _, neuron_states, _, iteration, buffer, _, spike_count, event_count = jax.lax.while_loop(
            cond, forward_pass, init
        )

        # Signal end-of-sample to the next rank in the pipeline
        jax.lax.cond(
            layer_idx != last_layer,
            lambda _: send(jnp.array([-1.0, -1.0]),
                           dest=rank + process_per_layer, tag=0, comm=comm),
            lambda _: [],
            operand=None
        )
        # stat_a = spike_count (rank 1: hidden spikes; rank 2: 0 — no firing)
        # stat_b = event_count (events received by this rank, neuron_idx >= 0)
        return jnp.zeros(()), neuron_states, iteration - 1, buffer, spike_count, event_count

    @jit
    def loop_over_batches(_, x):
        """
        jax.lax.scan body: process one sample and return updated states + stats.

        Dispatches to the correct inner function based on layer_idx and
        sync_first_layer.  The dispatch is a Python-level if-else (not jax.lax.cond)
        because layer_idx is a static Python int — only one branch is traced per rank,
        so the branches can have different output shapes without conflicts.

        Parameters
        ----------
        _ : None
            Scan carry (unused; each sample starts from empty_neuron_states).
        x : jnp.ndarray, shape (max_events, 2)
            Event array for this sample (rank 0 only; other ranks receive zeros).

        Returns
        -------
        None, (ns_values, iterations, ns, buffer, stat_a, stat_b)
            Scan output stacked across batch dimension B.
        """
        neuron_states = empty_neuron_states
        args = (neuron_states, x)
        # layer_idx, sync_first_layer, sync_first_frame are all static →
        # only one branch is compiled per rank; no runtime cost.
        if layer_idx == 0:
            result = input_layer(args)
        elif layer_idx == 1 and (sync_first_layer or sync_first_frame):
            result = sync_lif_layer(args)
        else:
            result = other_layers(args)
        _, new_ns, iterations, buffer, stat_a, stat_b = result
        return None, (new_ns.values, iterations, new_ns, buffer, stat_a, stat_b)

    _, (all_outputs, all_iters, all_ns, buffer, all_stat_a, all_stat_b) = jax.lax.scan(
        loop_over_batches, None, batch_data
    )
    mpi4jax.barrier(comm=comm)
    # stat_a: rank 0 = input events sent, rank 1 = hidden spikes fired, ranks 2…N-1 = 0
    # stat_b: rank 0 = 0, ranks 1…N-1 = events received (neuron_idx >= 0)
    return all_outputs, all_iters, all_ns, buffer, all_stat_a, all_stat_b


# ── Data helpers ───────────────────────────────────────────────────────────────
def build_nmnist_loader(data_dir, time_window):
    """
    Build a single-sample DataLoader for the N-MNIST test set.

    Converts raw DVS events into dense frames using tonic's ToFrame transform,
    using only the first saccade to match the asynctorch training setup.

    Parameters
    ----------
    data_dir : str
        Root directory where tonic stores/downloads the N-MNIST dataset.
    time_window : float
        Duration of each frame in microseconds (e.g. 10000.0 = 10 ms).
        Must match the time_window used during training.

    Returns
    -------
    DataLoader
        Yields (frames_batch, label_batch) with batch_size=1, no shuffle,
        frames padded to equal length via tonic's PadTensors collator.
    """
    transform = transforms.ToFrame(
        sensor_size=tonic.datasets.NMNIST.sensor_size,
        time_window=time_window,
    )
    dataset = tonic.datasets.NMNIST(
        save_to=data_dir, transform=transform,
        train=False, first_saccade_only=True
    )
    collate_fn = tonic.collation.PadTensors(batch_first=False)
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


def format_frames_as_events(frames, max_events, scheduler="momentum", seed=0):
    """
    Convert a (T, C, H, W) frame array into a fixed-length event list.

    Each frame's active pixels become (pixel_index, pixel_value) pairs,
    interleaved with -3 frame-end sentinels.  The result is zero-padded or
    truncated to exactly max_events rows.

    Parameters
    ----------
    frames : np.ndarray, shape (T, C, H, W)
        Dense event frames from tonic's ToFrame transform.
        Typically T=9 frames of shape (2, 34, 34) for N-MNIST.
    max_events : int
        Fixed output length (rows).  Should be >= T * max_pixels_per_frame + T.
        Overshooting is safe; the array is zero-padded with (-2, -2).
    scheduler : str, default "momentum"
        Within-frame pixel ordering:
          "momentum" — sort by pixel value descending (highest activation first)
          "random"   — shuffle using a seeded RNG
    seed : int, default 0
        RNG seed for "random" scheduler (typically set to sample index).

    Returns
    -------
    events : np.ndarray, shape (max_events, 2), dtype float32
        Rows: [pixel_index, pixel_value] for real events,
              [-3.0, 0.0] for frame separators,
              [-2.0, -2.0] for padding.
    n_pixel_events : int
        Total number of real pixel events (excludes -3 sentinels and padding).
    """
    rng = np.random.default_rng(seed)
    events = []
    n_pixel_events = 0
    for t in range(frames.shape[0]):
        frame_flat = frames[t].flatten().astype(np.float32)
        active     = np.where(frame_flat > 0)[0]
        if len(active):
            n_pixel_events += len(active)
            if scheduler == "momentum":
                # Highest-activity pixels first — matches training momentum ordering
                order = np.argsort(frame_flat[active])[::-1]
            else:
                order = rng.permutation(len(active))
            for idx in active[order]:
                events.append([float(idx), float(frame_flat[idx])])
        events.append([-3.0, 0.0])   # frame separator

    n = len(events)
    if n > max_events:
        events = events[:max_events]
    else:
        events += [[-2.0, -2.0]] * (max_events - n)   # pad with -2 sentinel
    return np.array(events, dtype=np.float32), n_pixel_events


def make_empty_states(in_size, out_size, threshold):
    """
    Create a zero-initialized NeuronStates for one layer.

    Called once per rank before inference; the same instance is reused
    across all samples (reset inside loop_over_batches).

    Parameters
    ----------
    in_size : int
        Number of pre-synaptic neurons (input dimension of the weight matrix).
    out_size : int
        Number of neurons in this layer (output dimension).
    threshold : float
        Initial firing threshold, broadcast to all neurons.

    Returns
    -------
    NeuronStates
        All arrays zeroed; thresholds set to the given value.
    """
    return NeuronStates(
        values               = jnp.zeros(out_size),
        thresholds           = jnp.full(out_size, threshold),
        input_residuals      = jnp.zeros(in_size),
        input_order          = jnp.zeros(in_size, dtype=jnp.int32),
        input_activity       = jnp.zeros(in_size),
        layer_activity       = jnp.zeros(out_size),
        output_activity      = jnp.zeros((in_size, out_size)),
        last_sent_iteration  = jnp.array(0, dtype=jnp.int32),
        input_vector         = jnp.zeros(in_size),
        output_vector        = jnp.zeros(out_size),
        values_history       = jnp.zeros((1, out_size)),
        history_index        = jnp.array(0, dtype=jnp.int32),
    )


def load_weights(cfg, layer):
    """
    Load and transpose the weight matrix for the given layer index.

    The .npz file stores weights as (out, in); this function transposes to
    (in, out) so weights[neuron_idx] gives the outgoing weight vector for
    neuron neuron_idx directly.

    Parameters
    ----------
    cfg : dict
        Config dict with key "weight_file" pointing to the .npz file.
        Expected keys: "arr_0", "arr_1", … one per computation layer.
        e.g. [2312,128,10] → arr_0, arr_1
             [2312,64,64,64,10] → arr_0, arr_1, arr_2, arr_3
    layer : int
        Layer index = MPI rank.  rank 0 returns a dummy (1,1) zero matrix.

    Returns
    -------
    jnp.ndarray, shape (in_size, out_size), dtype float32
        Transposed weight matrix for JAX dot products.
    """
    if layer == 0:
        return jnp.zeros((1, 1))   # rank 0 is input-only; no weights needed
    npz = np.load(cfg["weight_file"])
    key = f"arr_{layer - 1}"       # layer 1 → arr_0, layer 2 → arr_1, ...
    if key not in npz:
        raise KeyError(f"Weight file '{cfg['weight_file']}' has no key '{key}' "
                       f"(expected arr_0 … arr_{len([k for k in npz if k.startswith('arr')])-1})")
    w = npz[key]
    if w.ndim == 2:
        w = w.T   # PyTorch Linear stores (out, in); we need (in, out) for index lookup
    return jnp.array(w, dtype=jnp.float32)


# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(cfg, max_samples=None, scheduler="momentum", return_stats=False,
             sync_first_layer=False, sync_first_frame=False):
    """
    Run inference on the N-MNIST test set and report accuracy + activity stats.

    Collects per-rank statistics via MPI Reduce so that rank 0 can print
    a unified summary line.

    Parameters
    ----------
    cfg : dict
        YAML config dict (see make_params for keys).
    max_samples : int or None
        Stop after this many test samples.  None = full 10k set.
    scheduler : str, default "momentum"
        Event ordering within each frame: "momentum" or "random".
    return_stats : bool, default False
        If True, return (accuracy, stats_dict) instead of None.
    sync_first_layer : bool, default False
        If True, rank 1 uses synchronous LIF for ALL frames.
    sync_first_frame : bool, default False
        If True, rank 1 uses synchronous LIF for the FIRST frame only,
        then switches to async (fire every sync_rate events) for frames 2–N.

    Returns
    -------
    (accuracy, stats_dict) if return_stats=True, else None.
    stats_dict keys (rank 0 only; empty dict on other ranks):
      "input_events"    : float — average pixel events per sample (rank 0)
      "hidden_spikes"   : float — average hidden spikes per sample (rank 1)
      "hidden_events"   : float — average events received by rank 1
      "output_events"   : float — average events received by rank 2
      "time_per_sample" : float — wall-clock seconds per sample (rank 0)
      "total_runtime"   : float — total wall-clock seconds
      "accuracy"        : float — classification accuracy (0–100)
    """
    params      = make_params(cfg)
    threshold   = params.init_thresholds
    layer_sizes = params.layer_sizes
    key         = jax.random.key(42)

    weights  = load_weights(cfg, layer_idx)

    # Build layer-specific NeuronStates.
    # rank 0 = input feeder → dummy (1,1); all other ranks own a real layer.
    # Generic formula works for any depth: rank k processes layer_sizes[k-1]→layer_sizes[k].
    if layer_idx == 0:
        empty_ns = make_empty_states(1, 1, threshold)
    else:
        empty_ns = make_empty_states(layer_sizes[layer_idx - 1],
                                     layer_sizes[layer_idx], threshold)

    # max_events: worst case is all pixels active in all 9 frames + 9 separators
    max_events = layer_sizes[0] * 9 + 20

    data_loader = None
    if rank == 0:
        data_loader = build_nmnist_loader(
            cfg.get("data_dir", "./data"),
            float(cfg.get("time_window", 10000.0))
        )
        n_test = len(data_loader.dataset)
        if max_samples:
            n_test = min(n_test, max_samples)
    else:
        n_test = 0
    # Broadcast sample count so all ranks run the same number of loop iterations
    n_test = comm.bcast(n_test, root=0)

    if rank == 0:
        if sync_first_layer:
            mode = "sync-all-frames"
        elif sync_first_frame:
            mode = "sync-frame1+async"
        else:
            mode = "async+async"
        print(f"\n[config] mode={mode} | fgs={params.sync_rate} | scheduler={scheduler} | "
              f"use_refrac={params.use_refrac} | n={n_test}")

    correct = 0
    loader_iter = iter(data_loader) if rank == 0 else None

    # Running sums across samples (divided by n_test at end for averages)
    total_input_events   = 0.0
    total_hidden_spikes  = 0.0
    total_hidden_events  = 0.0
    total_output_events  = 0.0
    total_time           = 0.0

    t_run_start = time.time()

    for sample_i in tqdm(range(n_test), disable=(rank != 0)):
        if rank == 0:
            frames_batch, label_batch = next(loader_iter)
            frames = frames_batch[:, 0].numpy()   # collapse batch dim: (T, C, H, W)
            label  = int(label_batch[0].item())
            n_pix  = int(np.sum(frames > 0))      # total pixel events (for stats)
            sample_events, _ = format_frames_as_events(frames, max_events,
                                                       scheduler=scheduler, seed=sample_i)
            batch_data = jnp.array(sample_events[None, ...])   # add batch dim: (1, E, 2)
        else:
            label      = 0
            n_pix      = 0
            batch_data = jnp.zeros((1, max_events, 2))

        # Broadcast label so all ranks compute the same sample_i
        label = comm.bcast(label, root=0)

        t_s = time.time()
        all_outputs, _, _, _, all_stat_a, all_stat_b = predict(
            params, key, weights, empty_ns, batch_data,
            sync_first_layer=sync_first_layer,
            sync_first_frame=sync_first_frame
        )
        t_sample = time.time() - t_s

        if rank == last_layer:
            pred = int(jnp.argmax(all_outputs[0]))
            correct += int(pred == label)

        # ── Collect per-sample stats across ranks via MPI Reduce ──────────────
        # Layout: [input_events, hidden_spikes, hidden_events, output_events, time_s]
        # Each rank fills only its own fields (zeros elsewhere); SUM gives rank 0
        # the complete picture without extra communication.
        local = np.zeros(5, dtype=np.float64)
        if rank == 0:
            local[0] = n_pix       # pixel events sent by rank 0
            local[4] = t_sample    # wall-clock time measured at rank 0
        elif rank == 1:
            local[1] = int(all_stat_a[0])   # hidden spikes fired
            local[2] = int(all_stat_b[0])   # events received by rank 1
        elif rank == last_layer:
            local[3] = int(all_stat_b[0])   # events received by last rank (output layer)

        global_s = np.zeros(5, dtype=np.float64)
        comm.Reduce(local, global_s, op=MPI.SUM, root=0)

        if rank == 0:
            total_input_events  += global_s[0]
            total_hidden_spikes += global_s[1]
            total_hidden_events += global_s[2]
            total_output_events += global_s[3]
            total_time          += global_s[4]

    total_runtime = time.time() - t_run_start

    # ── Collect accuracy from last rank ───────────────────────────────────────
    acc = 0.0
    if rank == last_layer:
        acc = correct / n_test * 100.0

    # Broadcast to rank 0 for printing and CSV export
    acc_buf = np.array([acc], dtype=np.float64)
    comm.Bcast(acc_buf, root=last_layer)
    acc = float(acc_buf[0])

    if rank == 0:
        n = n_test
        stats = {
            "input_events"   : total_input_events   / n,
            "hidden_spikes"  : total_hidden_spikes  / n,
            "hidden_events"  : total_hidden_events  / n,
            "output_events"  : total_output_events  / n,
            "time_per_sample": total_time           / n,
            "total_runtime"  : total_runtime,
            "accuracy"       : acc,
        }
        print(f"  Accuracy={acc:.2f}%  "
              f"in={stats['input_events']:.0f}  "
              f"hid_spk={stats['hidden_spikes']:.1f}  "
              f"hid_ev={stats['hidden_events']:.1f}  "
              f"out_ev={stats['output_events']:.1f}  "
              f"t/sample={stats['time_per_sample']*1000:.1f}ms  "
              f"total={stats['total_runtime']:.1f}s")
        if return_stats:
            return acc, stats
    else:
        if return_stats:
            return acc, {}

    return acc, {} if return_stats else None


# ── Benchmark sweep ────────────────────────────────────────────────────────────
def run_benchmark(cfg, max_samples=None, out_csv=None):
    """
    Full parameter sweep: forward_group_size × scheduler × use_refrac.

    Runs evaluate() for every combination of the three axes and writes
    results to a CSV file.  Output is also printed inline for tee capture.

    Parameters
    ----------
    cfg : dict
        Base YAML config; "forward_group_size" and "use_refrac" are overridden
        per run.
    max_samples : int or None
        Limit samples per run for faster sweeps during development.
    out_csv : str or None
        Output CSV path.  Defaults to "benchmark_nmnist.csv".

    Sweep axes
    ----------
    forward_group_size : [1, 8, 16, 32, 64, 128]
        Async fire rate: fire every N events (hidden layer only).
    scheduler          : ["momentum", "random"]
        Within-frame pixel ordering.
    use_refrac         : [True, False]
        Whether to apply the refractory mask between frames.
    """
    fw_sizes   = [1, 8, 16, 32, 64, 128]
    schedulers = ["momentum", "random"]
    refracs    = [True, False]

    results = []

    for fgs in fw_sizes:
        for sched in schedulers:
            for ur in refracs:
                cfg_run = dict(cfg)
                cfg_run["forward_group_size"] = fgs
                cfg_run["use_refrac"] = ur

                acc, stats = evaluate(cfg_run, max_samples=max_samples,
                                      scheduler=sched, return_stats=True)

                if rank == 0 and stats:
                    results.append({
                        "fgs": fgs, "scheduler": sched, "use_refrac": ur,
                        **stats,
                    })

    if rank == 0 and results:
        header = ("fgs,scheduler,use_refrac,accuracy,"
                  "input_events,hidden_spikes,hidden_events,output_events,"
                  "time_per_sample_ms,total_runtime_s")
        rows = []
        for r in results:
            rows.append(
                f"{r['fgs']},{r['scheduler']},{r['use_refrac']},"
                f"{r['accuracy']:.2f},"
                f"{r['input_events']:.1f},{r['hidden_spikes']:.1f},"
                f"{r['hidden_events']:.1f},{r['output_events']:.1f},"
                f"{r['time_per_sample']*1000:.2f},{r['total_runtime']:.1f}"
            )

        print("\n\n=== BENCHMARK RESULTS (CSV) ===")
        print(header)
        for row in rows:
            print(row)
        print("=== END CSV ===\n")

        csv_path = out_csv or "benchmark_nmnist.csv"
        with open(csv_path, "w") as f:
            f.write(header + "\n")
            for row in rows:
                f.write(row + "\n")
        print(f"[saved] {csv_path}")


# ── firing_nb sweep ────────────────────────────────────────────────────────────
def run_firing_nb_sweep(cfg, max_samples=None, out_csv=None,
                        sync_first_layer=False, sync_first_frame=False):
    """
    Sweep firing_nb at fixed best config: fgs=128, use_refrac=True, momentum.

    firing_nb caps the number of spikes emitted per firing event.  Values ≥ 64
    are lossless; there is a sharp accuracy cliff at ≤ 32.

    Parameters
    ----------
    cfg : dict
        Base YAML config; "firing_nb" is overridden per run.
    max_samples : int or None
        Limit samples per run.
    out_csv : str or None
        Output CSV path.  Defaults to "firing_nb_sweep_nmnist.csv".
    sync_first_layer : bool, default False
        If True, rank 1 uses synchronous LIF for all frames.
    sync_first_frame : bool, default False
        If True, rank 1 uses synchronous LIF only for the first frame.

    Sweep values
    ------------
    firing_nb : [-1, 128, 64, 32, 16, 8, 4, 2, 1]
        -1 = no cap (all spikes emitted); positive = hard cap per firing event.
    """
    firing_nbs = [-1, 128, 64, 32, 16, 8, 4, 2, 1]

    base = dict(cfg)
    base["forward_group_size"] = 128
    base["use_refrac"]         = True

    results = []
    for fnb in firing_nbs:
        cfg_run = dict(base)
        cfg_run["firing_nb"] = fnb

        acc, stats = evaluate(cfg_run, max_samples=max_samples,
                              scheduler="momentum", return_stats=True,
                              sync_first_layer=sync_first_layer,
                              sync_first_frame=sync_first_frame)

        if rank == 0 and stats:
            results.append({"firing_nb": fnb, **stats})

    if rank == 0 and results:
        header = ("firing_nb,accuracy,"
                  "input_events,hidden_spikes,hidden_events,output_events,"
                  "time_per_sample_ms,total_runtime_s")
        rows = []
        for r in results:
            rows.append(
                f"{r['firing_nb']},{r['accuracy']:.2f},"
                f"{r['input_events']:.1f},{r['hidden_spikes']:.1f},"
                f"{r['hidden_events']:.1f},{r['output_events']:.1f},"
                f"{r['time_per_sample']*1000:.2f},{r['total_runtime']:.1f}"
            )

        print("\n\n=== FIRING_NB SWEEP RESULTS (CSV) ===")
        print(header)
        for row in rows:
            print(row)
        print("=== END CSV ===\n")

        csv_path = out_csv or "firing_nb_sweep_nmnist.csv"
        with open(csv_path, "w") as f:
            f.write(header + "\n")
            for row in rows:
                f.write(row + "\n")
        print(f"[saved] {csv_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    """
    Entry point: parse CLI arguments, initialize MPI, and dispatch to the
    appropriate run mode (single eval, benchmark sweep, or firing_nb sweep).

    CLI arguments
    -------------
    --config            : str   — path to YAML config (default: sepi_tmlr_conf_1006.yaml)
    --max_samples       : int   — cap the test set size (useful for quick checks)
    --benchmark         : flag  — run full fgs × scheduler × use_refrac sweep
    --firing_nb_sweep   : flag  — run firing_nb sweep at best fixed config
    --scheduler         : str   — "momentum" or "random" (single-run mode only)
    --out_csv           : str   — output CSV path for benchmark/sweep modes
    --sync_first_layer  : flag  — use synchronous LIF for rank 1 (fires once per frame)

    MPI layout (set from comm size and config layer_sizes)
    ──────────────────────────────────────────────────────
    rank 0        → input feeder
    rank 1        → first hidden layer (sync-LIF or async)
    rank 2…N-2    → intermediate hidden layers (async, other_layers)
    rank N-1      → output accumulator (argmax classification)
    last_layer    = len(layer_sizes) - 1
    process_per_layer = 1  (one rank per layer; no intra-layer parallelism)
    Examples:
      layer_sizes [2312,128,10]        → mpirun -n 3, last_layer=2
      layer_sizes [2312,64,64,64,10]   → mpirun -n 5, last_layer=4
    """
    global comm, rank, size, layer_idx, process_per_layer, last_layer

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",           type=str, default="sepi_tmlr_conf_1006.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--max_samples",      type=int, default=None,
                        help="Limit number of test samples (None = full 10k)")
    parser.add_argument("--benchmark",        action="store_true",
                        help="Sweep over fgs × scheduler × use_refrac")
    parser.add_argument("--firing_nb_sweep",  action="store_true",
                        help="Sweep firing_nb at fixed best config (fgs=128, refrac=True, momentum)")
    parser.add_argument("--scheduler",        type=str, default=None,
                        help="momentum or random (single-run mode; overrides config)")
    parser.add_argument("--out_csv",           type=str, default=None,
                        help="Output CSV path for benchmark or sweep results")
    parser.add_argument("--sync_first_layer",  action="store_true",
                        help="Rank 1 uses synchronous LIF for ALL frames (fires once per frame at -3)")
    parser.add_argument("--sync_first_frame",  action="store_true",
                        help="Rank 1 uses synchronous LIF only for the FIRST frame, then async")
    parser.add_argument("--fgs",               type=int, default=None,
                        help="Override forward_group_size (async fire rate) from config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.fgs is not None:
        cfg["forward_group_size"] = args.fgs

    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    size  = comm.Get_size()

    # Map each rank to a layer: rank 0 = input, ranks 1…N-2 = hidden, rank N-1 = output
    n_computation_layers = len(cfg["layer_sizes"]) - 1
    process_per_layer    = 1
    last_layer           = n_computation_layers
    layer_idx            = rank

    if rank == 0:
        print(f"Ranks={size}, last_layer={last_layer}, layer_sizes={cfg['layer_sizes']}")

    if args.benchmark:
        run_benchmark(cfg, max_samples=args.max_samples, out_csv=args.out_csv)
    elif args.firing_nb_sweep:
        run_firing_nb_sweep(cfg, max_samples=args.max_samples, out_csv=args.out_csv,
                            sync_first_layer=args.sync_first_layer,
                            sync_first_frame=args.sync_first_frame)
    else:
        sched = args.scheduler or cfg.get("scheduler", "momentum")
        evaluate(cfg, max_samples=args.max_samples, scheduler=sched, return_stats=True,
                 sync_first_layer=args.sync_first_layer,
                 sync_first_frame=args.sync_first_frame)


if __name__ == "__main__":
    main()
