"""
Frame-based N-MNIST inference: rank 1 synchronous LIF, all other hidden ranks async.

MPI pipeline — one rank per layer:
  rank 0        — input feeder
  rank 1        — sync LIF: accumulates all frame events, fires once per frame at -3
  rank 2 … N-2  — async LIF: fires every forward_group_size events or at -3
  rank N-1      — output accumulator

  
  layer_sizes: [2312, 64, 64, 64, 10] →  mpirun -n 5
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
import pandas as pd
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

comm             = None
rank             = None
size             = None
layer_idx        = None
process_per_layer = 1
last_layer       = None

_FRAME_END  = -3   # trigger membrane decay and fire
_SAMPLE_END = -1   # exit while-loop
_PADDING    = -2   # padding rows; skipped by input_layer


def make_params(cfg):
    return Params(
        dataset          = "nmnist",
        random_seed      = 42,
        layer_sizes      = tuple(cfg["layer_sizes"]),
        init_thresholds  = float(cfg.get("threshold",          0.3)),
        num_epochs       = 0,
        learning_rate    = 0.0,
        batch_size       = 1,
        load_file        = True,
        shuffle_activations = False,
        restrict         = 0,
        firing_nb        = int(cfg.get("firing_nb",           -1)),
        sync_rate        = int(cfg.get("forward_group_size", 8)),
        max_nonzero      = 0,
        shuffle_input    = False,
        threshold_lr     = 0.0,
        sparsity_impact  = (0.0, 0.0, 0.0),
        w_reg            = 0.0,
        rerun            = "",
        top_weights      = -1,
        history_size     = 0,
        tau_m            = float(cfg.get("tau_m",        1000.0)),
        dt               = float(cfg.get("time_window", 10000.0)),
        use_refrac       = bool(cfg.get("use_refrac",   False)),
    )


@partial(jax.jit, static_argnames=["n_neurons", "firing_nb_static"])
def binary_output_events(activations, thresholds, n_neurons, firing_nb_static):
    fired_mask  = activations > thresholds
    fired_count = jnp.sum(fired_mask.astype(jnp.int32))
    sort_key   = jnp.where(fired_mask, -activations, 1e10)
    sorted_idx = jnp.argsort(sort_key)

    if firing_nb_static >= 0:
        effective_count = jnp.minimum(fired_count, firing_nb_static)
    else:
        effective_count = fired_count
    slots   = jnp.arange(n_neurons)
    out_idx = jnp.where(slots < effective_count, sorted_idx, jnp.full((), -2))
    out_val = jnp.where(out_idx >= 0, 1.0, -2.0)
    pairs   = jnp.stack([out_idx.astype(jnp.float32), out_val], axis=1)
    return pairs, effective_count


@partial(jax.jit, static_argnames=["params", "grad"])
def layer_computation(params, key, neuron_idx, layer_input, weights,
                      neuron_states, is_refrac, iteration=0, grad=False):
    filtered_weights = keep_top_k(weights[neuron_idx], params.top_weights, apply_abs=True)
    raw_activations = jax.lax.cond(
        neuron_idx < 0,
        lambda _: neuron_states.values,
        lambda _: neuron_states.values + jnp.dot(layer_input, filtered_weights),
        None
    )
    if params.use_refrac:
        activations = raw_activations * (~is_refrac)
    else:
        activations = raw_activations

    n_out = params.layer_sizes[layer_idx]

    # Output layer: no decay — decaying attenuates late-frame evidence and hurts accuracy.
    @jit
    def last_layer_case(_):
        dummy_out = jnp.zeros((n_out, 2))
        ns = neuron_states.replace(values=activations)
        return jnp.array(0), dummy_out, ns, is_refrac

    @jit
    def hidden_layer_case(_):
        # Use == _FRAME_END (not < 0) so sample-end (-1) does NOT trigger a fire.
        fire = (iteration - neuron_states.last_sent_iteration) >= params.sync_rate
        fire = jnp.logical_or(fire, neuron_idx == _FRAME_END)

        fired_mask = jnp.logical_and(activations > neuron_states.thresholds, fire)

        pairs, valid_elements = binary_output_events(
            activations, neuron_states.thresholds, n_out, params.firing_nb
        )
        pairs          = jax.lax.cond(fire, lambda _: pairs,
                                      lambda _: jnp.full_like(pairs, -2.0), None)
        valid_elements = jax.lax.cond(fire, lambda _: valid_elements,
                                      lambda _: jnp.array(0), None)

        new_vals = jnp.where(fired_mask, 0.0, activations)
        decay    = jnp.exp(-params.dt / params.tau_m)
        new_vals = jax.lax.cond(
            neuron_idx == _FRAME_END,
            lambda _: new_vals * decay,
            lambda _: new_vals,
            None
        )

        if params.use_refrac:
            new_is_refrac = jax.lax.cond(
                fire,
                lambda _: is_refrac | fired_mask,
                lambda _: is_refrac,
                None
            )
            # Clear at frame-end: no refractory state crosses frame boundaries.
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


@partial(jax.jit, static_argnames=["params", "grad"])
def predict(params, key, weights, empty_neuron_states, batch_data, grad=False):

    @jit
    def input_layer(args):
        neuron_states, x = args
        x_p = jnp.array(x)

        def not_padding(row):
            return row != _PADDING
        mask = jax.vmap(not_padding)(x_p)
        loop_iterations = (jnp.count_nonzero(mask) / 2).astype(int)

        def send_input(i, carry):
            send(x_p[i], dest=rank + process_per_layer, tag=0, comm=comm)
            return i

        jax.lax.fori_loop(0, loop_iterations, send_input, 0)
        send(jnp.array([-1.0, -1.0]), dest=rank + process_per_layer, tag=0, comm=comm)
        return jnp.zeros(()), neuron_states, 0, jnp.zeros((1, 2)), loop_iterations, jnp.array(0, dtype=jnp.int32)

    @jit
    def sync_lif_layer(args):
        neuron_states, _ = args
        membrane   = neuron_states.values
        thresholds = neuron_states.thresholds
        n_hidden   = membrane.shape[0]
        decay      = jnp.exp(-params.dt / params.tau_m)

        def fire_and_send(membrane_in, is_refrac_in):
            if params.use_refrac:
                eff = membrane_in * (~is_refrac_in)
            else:
                eff = membrane_in
            # Fire on raw accumulated membrane (pre-decay)
            fired_mask  = eff > thresholds
            fired_count = jnp.sum(fired_mask.astype(jnp.int32))
            # Fire on pre-decay membrane: decaying before firing kills the signal
            # (decay ≈ 4.5e-5 for default tau_m=1000, dt=10000).
            reset_mem   = jnp.where(fired_mask, 0.0, eff)
            carry_mem   = reset_mem * decay
            sort_key    = jnp.where(fired_mask, -eff, 1e10)
            sorted_idx  = jnp.argsort(sort_key)

            def send_spike(i, _):
                send(jnp.stack([sorted_idx[i].astype(jnp.float32), jnp.array(1.0)]),
                     dest=rank + process_per_layer, tag=0, comm=comm)
                return None

            jax.lax.fori_loop(0, fired_count, send_spike, None)
            send(jnp.array([-3.0, 0.0]), dest=rank + process_per_layer, tag=0, comm=comm)
            if params.use_refrac:
                new_is_refrac = jnp.zeros_like(fired_mask)   # clear at frame-end; no cross-frame refractory
            else:
                new_is_refrac = is_refrac_in
            return carry_mem, fired_count, new_is_refrac

        def cond_a(state):
            _, neuron_idx, _, _, _ = state
            return neuron_idx != _SAMPLE_END

        @jit
        def forward_pass_a(state):
            membrane, _, event_count, spike_count, is_refrac = state

            (neuron_idx, layer_input) = recv(
                jnp.zeros((2,)), source=rank - process_per_layer, tag=0, comm=comm
            )
            neuron_idx = neuron_idx.astype(int)

            new_membrane = jax.lax.cond(
                neuron_idx >= 0,
                lambda _: membrane + weights[neuron_idx] * layer_input,
                lambda _: membrane,
                None
            )

            @jit
            def on_frame_end(_):
                carry, fired_count, new_is_refrac = fire_and_send(new_membrane, is_refrac)
                return carry, fired_count, new_is_refrac

            @jit
            def no_frame_end(_):
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
                  jnp.array(0, dtype=jnp.int32),
                  jnp.array(0, dtype=jnp.int32),
                  jnp.array(0, dtype=jnp.int32),
                  jnp.zeros(n_hidden, dtype=bool))

        final_mem, _, event_count, spike_count, _ = jax.lax.while_loop(
            cond_a, forward_pass_a, init_a
        )

        send(jnp.array([-1.0, -1.0]), dest=rank + process_per_layer, tag=0, comm=comm)
        new_ns = neuron_states.replace(values=final_mem)
        return (jnp.zeros(()), new_ns, jnp.array(0, dtype=jnp.int32),
                jnp.zeros((1, 2)), spike_count, event_count)

    @jit
    def other_layers(args):
        neuron_states, _ = args
        # Use values.shape[0] rather than params.layer_sizes so the traced-but-unused
        # rank-0 path (dummy shape [1]) stays shape-consistent.
        n_out = neuron_states.values.shape[0]

        def cond(state):
            _, _, neuron_idx, _, _, _, _, _ = state
            return neuron_idx != _SAMPLE_END

        @jit
        def forward_pass(state):
            layer_input, neuron_states, _, iteration, buffer, is_refrac, spike_count, event_count = state

            (neuron_idx, layer_input) = recv(
                jnp.zeros((2,)), source=rank - process_per_layer, tag=0, comm=comm
            )
            neuron_idx = neuron_idx.astype(int)

            loop_iterations, activated_output, new_ns, new_is_refrac = layer_computation(
                params, key, neuron_idx, layer_input, weights,
                neuron_states, is_refrac, iteration, grad
            )

            new_spike_count = spike_count + loop_iterations
            new_event_count = event_count + jnp.where(
                neuron_idx >= 0, jnp.array(1, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)
            )

            @jit
            def send_spikes(_):
                def send_one(i, _):
                    send(activated_output[i],
                         dest=rank + process_per_layer, tag=0, comm=comm)
                    return None
                jax.lax.fori_loop(0, loop_iterations, send_one, None)
                return []

            jax.lax.cond(layer_idx == last_layer, lambda _: [], send_spikes, None)

            jax.lax.cond(
                jnp.logical_and(layer_idx != last_layer, neuron_idx == _FRAME_END),
                lambda _: send(jnp.array([-3.0, 0.0]),
                               dest=rank + process_per_layer, tag=0, comm=comm),
                lambda _: [],
                operand=None
            )

            return layer_input, new_ns, neuron_idx, iteration + 1, buffer, new_is_refrac, new_spike_count, new_event_count

        init = (jnp.zeros(()), neuron_states, jnp.array(0, dtype=jnp.int32),
                jnp.array(0, dtype=jnp.int32), jnp.zeros((1, 2)),
                jnp.zeros(n_out, dtype=bool),
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0, dtype=jnp.int32))

        _, neuron_states, _, iteration, buffer, _, spike_count, event_count = jax.lax.while_loop(
            cond, forward_pass, init
        )

        jax.lax.cond(
            layer_idx != last_layer,
            lambda _: send(jnp.array([-1.0, -1.0]),
                           dest=rank + process_per_layer, tag=0, comm=comm),
            lambda _: [],
            operand=None
        )
        return jnp.zeros(()), neuron_states, iteration - 1, buffer, spike_count, event_count

    @jit
    def loop_over_batches(_, x):
        neuron_states = empty_neuron_states
        args = (neuron_states, x)
        if layer_idx == 0:
            result = input_layer(args)
        elif layer_idx == 1:
            result = sync_lif_layer(args)
        else:
            result = other_layers(args)
        _, new_ns, iterations, buffer, stat_a, stat_b = result
        return None, (new_ns.values, iterations, new_ns, buffer, stat_a, stat_b)

    _, (all_outputs, all_iters, all_ns, buffer, all_stat_a, all_stat_b) = jax.lax.scan(
        loop_over_batches, None, batch_data
    )
    mpi4jax.barrier(comm=comm)
    return all_outputs, all_iters, all_ns, buffer, all_stat_a, all_stat_b


def build_nmnist_loader(data_dir, time_window):
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
    rng = np.random.default_rng(seed)
    events = []
    n_pixel_events = 0
    for t in range(frames.shape[0]):
        frame_flat = frames[t].flatten().astype(np.float32)
        active     = np.where(frame_flat > 0)[0]
        if len(active):
            n_pixel_events += len(active)
            if scheduler == "momentum":
                order = np.argsort(frame_flat[active])[::-1]
            else:
                order = rng.permutation(len(active))
            for idx in active[order]:
                events.append([float(idx), float(frame_flat[idx])])
        events.append([-3.0, 0.0])
    n = len(events)
    if n > max_events:
        events = events[:max_events]
    else:
        events += [[-2.0, -2.0]] * (max_events - n)
    return np.array(events, dtype=np.float32), n_pixel_events


def make_empty_states(in_size, out_size, threshold):
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
    if layer == 0:
        return jnp.zeros((1, 1))
    npz = np.load(cfg["weight_file"])
    key = f"arr_{layer - 1}"
    if key not in npz:
        raise KeyError(f"Weight file '{cfg['weight_file']}' has no key '{key}' "
                       f"(expected arr_0 … arr_{len([k for k in npz if k.startswith('arr')])-1})")
    w = npz[key]
    if w.ndim == 2:
        w = w.T   # PyTorch Linear stores (out, in); we need (in, out)
    return jnp.array(w, dtype=jnp.float32)


# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(cfg, max_samples=None, scheduler="momentum", return_stats=False, out_stats=None):
    params      = make_params(cfg)
    threshold   = params.init_thresholds
    layer_sizes = params.layer_sizes
    key         = jax.random.key(42)

    weights  = load_weights(cfg, layer_idx)

    if layer_idx == 0:
        empty_ns = make_empty_states(1, 1, threshold)
    else:
        empty_ns = make_empty_states(layer_sizes[layer_idx - 1],
                                     layer_sizes[layer_idx], threshold)

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
    n_test = comm.bcast(n_test, root=0)

    if rank == 0:
        print(f"\n[config] mode=sync-layer1+async | fgs={params.sync_rate} | "
              f"scheduler={scheduler} | use_refrac={params.use_refrac} | n={n_test}")

    correct = 0
    loader_iter = iter(data_loader) if rank == 0 else None

    total_input_events   = 0.0
    total_hidden_spikes  = 0.0
    total_time           = 0.0
    per_rank_events      = np.zeros(size, dtype=np.float64)
    sample_rows          = [] if (rank == 0 and out_stats) else None

    t_run_start = time.time()

    for sample_i in tqdm(range(n_test), disable=(rank != 0)):
        if rank == 0:
            frames_batch, label_batch = next(loader_iter)
            frames = frames_batch[:, 0].numpy()
            label  = int(label_batch[0].item())
            n_pix  = int(np.sum(frames > 0))
            sample_events, _ = format_frames_as_events(frames, max_events,
                                                       scheduler=scheduler, seed=sample_i)
            batch_data = jnp.array(sample_events[None, ...])
        else:
            label      = 0
            n_pix      = 0
            batch_data = jnp.zeros((1, max_events, 2))

        label = comm.bcast(label, root=0)

        t_s = time.time()
        all_outputs, _, _, _, all_stat_a, all_stat_b = predict(
            params, key, weights, empty_ns, batch_data
        )
        t_sample = time.time() - t_s

        pred_buf = np.array([0], dtype=np.int32)
        if rank == last_layer:
            pred_buf[0] = int(jnp.argmax(all_outputs[0]))
            correct += int(pred_buf[0] == label)
        comm.Bcast(pred_buf, root=last_layer)
        pred = int(pred_buf[0])

        local_stat_b = np.array([float(int(all_stat_b[0]))], dtype=np.float64)
        all_stat_b_gathered = None
        if rank == 0:
            all_stat_b_gathered = np.zeros(size, dtype=np.float64)
        comm.Gather(local_stat_b, all_stat_b_gathered, root=0)

        local2 = np.zeros(3, dtype=np.float64)
        if rank == 0:
            local2[0] = n_pix
            local2[2] = t_sample
        elif rank == 1:
            local2[1] = int(all_stat_a[0])

        global2 = np.zeros(3, dtype=np.float64)
        comm.Reduce(local2, global2, op=MPI.SUM, root=0)

        if rank == 0:
            total_input_events  += global2[0]
            total_hidden_spikes += global2[1]
            total_time          += global2[2]
            for r in range(size):
                per_rank_events[r] += all_stat_b_gathered[r]

            if sample_rows is not None:
                row = {
                    "sample":    sample_i,
                    "label":     label,
                    "pred":      pred,
                    "correct":   int(pred == label),
                    "input_ev":  int(global2[0]),
                    "hid_spk":   int(global2[1]),
                    "t_ms":      round(global2[2] * 1000, 2),
                }
                for r in range(1, size):
                    row[f"L{r}_ev"] = int(all_stat_b_gathered[r])
                sample_rows.append(row)

    total_runtime = time.time() - t_run_start

    acc = 0.0
    if rank == last_layer:
        acc = correct / n_test * 100.0

    acc_buf = np.array([acc], dtype=np.float64)
    comm.Bcast(acc_buf, root=last_layer)
    acc = float(acc_buf[0])

    if rank == 0:
        n = n_test
        layer_ev = [per_rank_events[r] / n for r in range(size)]
        stats = {
            "input_events"   : total_input_events  / n,
            "hidden_spikes"  : total_hidden_spikes / n,
            "layer_events"   : layer_ev,
            "time_per_sample": total_time          / n,
            "total_runtime"  : total_runtime,
            "accuracy"       : acc,
        }
        layer_str = "  ".join(f"L{r}_ev={layer_ev[r]:.1f}" for r in range(1, size))
        print(f"  Accuracy={acc:.2f}%  "
              f"in={stats['input_events']:.0f}  "
              f"hid_spk={stats['hidden_spikes']:.1f}  "
              f"{layer_str}  "
              f"t/sample={stats['time_per_sample']*1000:.1f}ms  "
              f"total={stats['total_runtime']:.1f}s")

        if out_stats and sample_rows:
            df = pd.DataFrame(sample_rows)
            base = out_stats.rsplit(".", 1)[0] if "." in out_stats else out_stats
            xlsx_path = base + ".xlsx"
            csv_path  = base + ".csv"
            df.to_excel(xlsx_path, index=False)
            df.to_csv(csv_path,   index=False)
            print(f"[saved] {xlsx_path}  ({len(df)} rows)")
            print(f"[saved] {csv_path}")

        if return_stats:
            return acc, stats
    else:
        if return_stats:
            return acc, {}

    return acc, {} if return_stats else None


# ── Benchmark sweep ────────────────────────────────────────────────────────────
def run_benchmark(cfg, max_samples=None, out_csv=None):
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
def run_firing_nb_sweep(cfg, max_samples=None, out_csv=None):
    firing_nbs = [-1, 128, 64, 32, 16, 8, 4, 2, 1]

    base = dict(cfg)
    base["forward_group_size"] = 128
    base["use_refrac"]         = True

    results = []
    for fnb in firing_nbs:
        cfg_run = dict(base)
        cfg_run["firing_nb"] = fnb

        acc, stats = evaluate(cfg_run, max_samples=max_samples,
                              scheduler="momentum", return_stats=True)

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
    global comm, rank, size, layer_idx, process_per_layer, last_layer

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          type=str,  default="sepi_tmlr_conf_1006.yaml")
    parser.add_argument("--max_samples",     type=int,  default=None)
    parser.add_argument("--benchmark",       action="store_true")
    parser.add_argument("--firing_nb_sweep", action="store_true")
    parser.add_argument("--scheduler",       type=str,  default=None)
    parser.add_argument("--out_csv",         type=str,  default=None)
    parser.add_argument("--out_stats",       type=str,  default=None,
                        help="Save per-sample activation stats to <path>.xlsx and <path>.csv")
    parser.add_argument("--fgs",             type=int,  default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.fgs is not None:
        cfg["forward_group_size"] = args.fgs

    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    size  = comm.Get_size()

    n_computation_layers = len(cfg["layer_sizes"]) - 1
    process_per_layer    = 1
    last_layer           = n_computation_layers
    layer_idx            = rank

    if rank == 0:
        print(f"Ranks={size}, last_layer={last_layer}, layer_sizes={cfg['layer_sizes']}")

    if args.benchmark:
        run_benchmark(cfg, max_samples=args.max_samples, out_csv=args.out_csv)
    elif args.firing_nb_sweep:
        run_firing_nb_sweep(cfg, max_samples=args.max_samples, out_csv=args.out_csv)
    else:
        sched = args.scheduler or cfg.get("scheduler", "momentum")
        evaluate(cfg, max_samples=args.max_samples, scheduler=sched,
                 return_stats=True, out_stats=args.out_stats)


if __name__ == "__main__":
    main()
