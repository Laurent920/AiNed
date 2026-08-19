# `sepi_tmlr_jax_1006_full.py` — Code Report

Frame-based N-MNIST inference in JAX + mpi4jax, reimplementing the asynctorch
TMLR paper (**arXiv:2408.05098**). This is the reference file to use when the
first hidden layer needs to be **synchronous** (fires once per frame) while
all remaining hidden layers stay **asynchronous** (fire every `sync_rate`
events, event-driven).

This file is JAX-side reference code and is treated as read-only in this
project — copy it before making any changes.

---

## 1. Architecture

The model runs as an MPI pipeline: **one rank per layer**. Ranks pass spike
events to each other over MPI (`mpi4jax.send`/`recv`), so the whole network
executes as N processes cooperating in lock-step through blocking receives.

```
rank 0        input feeder      reads event frames → sends (neuron_idx, value)
rank 1        first hidden      sync-LIF (if --sync_first_layer) OR async
rank 2 … N-2  hidden layers     always async (event-driven, other_layers)
rank N-1      output layer      accumulates membrane, no decay, argmax
```

`N = len(layer_sizes)`. Examples:
- `layer_sizes: [2312, 128, 10]` → `mpirun -n 3`
- `layer_sizes: [2312, 64, 64, 64, 10]` → `mpirun -n 5`

### Event encoding

Each MPI message is a `(neuron_idx, value)` pair. Three special (negative)
`neuron_idx` values act as sentinels threaded through the whole pipeline:

| Sentinel | Value | Meaning |
|---|---|---|
| `_FRAME_END` | `-3` | End of one frame — hidden layers fire + decay here |
| `_SAMPLE_END` | `-1` | End of one sample — while-loops exit |
| `_PADDING` | `-2` | Padding row in the fixed-length event array — skipped |

### Neuron model (LIF, per the TMLR paper)

1. **Integrate**: `membrane += W[neuron_idx] * spike_value`
2. **Refractory mask** (optional): `membrane *= ~is_refrac`
3. **Fire**: `membrane > threshold` AND a fire trigger is active
4. **Reset**: fired neurons hard-reset to `0` (not soft/subtractive reset)
5. **Leak/decay**: `membrane *= exp(-dt / tau_m)`, applied **once per frame**
   at the `_FRAME_END` boundary only — not per-event, so mid-frame
   integration isn't attenuated
6. **Output layer only**: skip step 5 entirely — the output layer accumulates
   total membrane across all frames with no decay, since classification uses
   `argmax` over the accumulated signal as a proxy for spike count. Decaying
   it would wipe out early-frame evidence and collapse accuracy (~55%
   instead of ~96%).

Firing is **momentum-ordered**: among neurons that crossed threshold, the one
with the highest pre-spike membrane is sent first (`binary_output_events`),
matching the ordering convention asynctorch used during training.

---

## 2. How the two "first hidden layer" modes work

The file supports three effective modes for rank 1, selected by CLI flags at
trace time (Python-level `if`, so JAX compiles only one branch per rank — no
runtime branching cost):

| Flag | Rank-1 behavior |
|---|---|
| *(none)* | **async+async** — rank 1 behaves exactly like ranks 2…N-2: fires every `sync_rate` events, or at `_FRAME_END` |
| `--sync_first_layer` | **sync-all-frames** — rank 1 accumulates the *entire* frame's events first, fires once at each `_FRAME_END`, then decays the reset residual for the next frame |
| `--sync_first_frame` | **sync-frame1+async** — rank 1 is sync-only for frame 1, then automatically switches to the async `sync_rate` rule for frames 2..N |

Rule you asked for — **first hidden layer sync, rest async** — is
`--sync_first_layer`.

Firing order inside the sync path matters and is a known trap: **fire on the
raw accumulated membrane first, then decay the reset residual** for carry to
the next frame. Decaying before checking the threshold kills the signal,
since `exp(-dt/tau_m) = exp(-10000/1000) ≈ 4.5e-5` with the default
`tau_m=1000`, `dt=10000`.

---

## 3. Function-by-function reference

### Setup / config

- **`make_params(cfg)`** — converts a YAML config dict into a frozen `Params`
  dataclass (from `other_helpers/helpers.py`). Frozen because JAX
  recompiles per unique static config.
- **`load_weights(cfg, layer)`** — loads `weight_file` (`.npz`), key
  `arr_{layer-1}`. Transposes every 2D array (PyTorch `Linear` stores
  `(out, in)`; JAX indexing here needs `(in, out)`). Rank 0 gets a dummy
  `(1,1)` zero matrix (it has no weights).
- **`make_empty_states(in_size, out_size, threshold)`** — builds a
  zero-initialized `NeuronStates` for one layer/rank; reused every sample,
  reset each call inside `loop_over_batches`.

### Data pipeline

- **`build_nmnist_loader(data_dir, time_window)`** — wraps `tonic`'s N-MNIST
  test set with `ToFrame` (first saccade only, matching training setup);
  returns a `DataLoader` (batch size 1, no shuffle).
- **`format_frames_as_events(frames, max_events, scheduler, seed)`** —
  converts a `(T, C, H, W)` dense frame tensor into a fixed-length
  `(max_events, 2)` event array: real `[pixel_idx, value]` rows per active
  pixel, a `[-3, 0]` separator after each frame, zero-padded with `[-2, -2]`.
  `scheduler="momentum"` sorts pixels by activation descending;
  `"random"` shuffles with a seeded RNG (seed = sample index).

### Core compute

- **`binary_output_events(activations, thresholds, n_neurons, firing_nb_static)`**
  — turns membrane potentials into a padded `(n_neurons, 2)` spike array,
  momentum-ordered (highest membrane first), optionally capped at
  `firing_nb_static` spikes (values `< 32` sharply hurt accuracy per the
  paper's sweep).
- **`layer_computation(...)`** — the one-event LIF update used by every
  **async** rank (both hidden and output). Dispatches internally on
  `layer_idx == last_layer`:
  - `last_layer_case` — accumulate only, no decay (see §1).
  - `hidden_layer_case` — fire condition = `sync_rate` events since last fire
    OR `_FRAME_END`; hard reset; decay only at `_FRAME_END`; refractory
    mask update (optional, gated on `params.use_refrac`).
- **`predict(...)`** — the JIT-compiled per-batch driver, one function body
  traced differently per rank. Contains four inner closures:
  - **`input_layer`** (rank 0) — walks the fixed event array, sends every
    non-padding row downstream, then sends the `_SAMPLE_END` sentinel.
  - **`sync_lif_layer`** (rank 1, only when `sync_first_layer` or
    `sync_first_frame`) — implements the two sync sub-modes described in §2
    via a `while_loop` (`forward_pass_a` for all-frames-sync,
    `forward_pass_b` for first-frame-only-sync). Shared helper
    `fire_and_send` does fire→reset→decay→momentum-sort→send for mode A.
  - **`other_layers`** (every non-input rank not using sync mode) — generic
    async event loop: `recv` one event, run `layer_computation`, forward any
    resulting spikes + a forwarded `_FRAME_END` sentinel downstream (skipped
    at the output rank).
  - **`loop_over_batches`** — the `jax.lax.scan` body; a Python-level
    `if/elif/else` picks exactly one of the three closures above per rank
    (static dispatch, no wasted compute).

### Orchestration

- **`evaluate(cfg, max_samples, scheduler, return_stats, sync_first_layer, sync_first_frame)`**
  — the main test loop: builds params/weights/states once, then per sample
  loads a frame, calls `predict`, takes `argmax` at the output rank for the
  prediction, and does an `MPI.Reduce` to gather per-rank stats (input
  events, hidden spikes, hidden/output events received, timing) onto rank 0.
  Prints the summary line and optionally returns `(accuracy, stats_dict)`.
- **`run_benchmark(cfg, max_samples, out_csv)`** — full sweep over
  `forward_group_size × scheduler × use_refrac`, writes a CSV
  (`benchmark_nmnist.csv` by default).
- **`run_firing_nb_sweep(cfg, max_samples, out_csv, sync_first_layer, sync_first_frame)`**
  — sweeps `firing_nb` at a fixed best config (`fgs=128, use_refrac=True,
  momentum`), writes `firing_nb_sweep_nmnist.csv` by default.
- **`main()`** — argument parsing + MPI rank/layer mapping, then dispatches
  to `evaluate` / `run_benchmark` / `run_firing_nb_sweep`.

---

## 4. Config file

Example: `sepi_tmlr_conf_1006.yaml` (2-layer network `[2312, 128, 10]`):

```yaml
layer_sizes: [2312, 128, 10]
mode: inference

weight_file: "weights_nmnist/weights_nmnist_1006_1.npz"

tau_m: 1000.0
threshold: 0.3
firing_nb: -1
forward_group_size: 128
use_refrac: true

time_window: 10000.0
data_dir: "./data"
debug: true
```

| Key | Meaning |
|---|---|
| `layer_sizes` | Neuron counts per layer, input → …hidden… → output. Also determines `mpirun -n`. |
| `weight_file` | `.npz` with keys `arr_0 … arr_{L-1}`, one per computation layer (PyTorch `(out,in)`, auto-transposed by `load_weights`). |
| `tau_m` | Membrane time constant (µs). |
| `threshold` | LIF firing threshold. |
| `firing_nb` | Max spikes emitted per firing event; `-1` = uncapped. |
| `forward_group_size` | `sync_rate` — async hidden layers fire every N events received. |
| `use_refrac` | Apply refractory masking (`(V+I)*~is_refrac`), cleared each frame-end. |
| `time_window` | Frame duration in µs — this is `dt` used in the decay `exp(-dt/tau_m)`. |
| `data_dir` | Where `tonic` stores/downloads the N-MNIST dataset. |
| `scheduler` | *(optional key, not present here)* `momentum` or `random`; defaults to `momentum` if absent — CLI `--scheduler` overrides it. |

Other configs available in the repo for different network sizes / datasets:
`sepi_tmlr_conf_3hidden_1006.yaml` (`[2312,64,64,64,10]`, N-MNIST),
`sepi_tmlr_conf_3hidden_nmnist_momentum.yaml` / `_random.yaml`,
`sepi_tmlr_conf_3hidden_shd.yaml` / `_shd_random.yaml` (SHD dataset).

---

## 5. Run command

Environment: use the project venv, which has `mpi4py`, `mpi4jax`, `jax`,
`tonic`, `torch` installed (`/home/sepi/.acync-cpu/bin/python3`; the system
`python3` has no packages).

**First hidden layer sync, remaining hidden layers async** (what you asked for):

```bash
JAX_PLATFORMS=cpu mpirun -n 3 /home/sepi/.acync-cpu/bin/python3 \
    sepi_tmlr_jax_1006_full.py \
    --config sepi_tmlr_conf_1006.yaml \
    --sync_first_layer \
    --max_samples 200
```

- `-n 3` must equal `len(layer_sizes)` (here `[2312,128,10]` → 3 ranks).
- Drop `--max_samples 200` to run the full 10,000-sample N-MNIST test set.
- Add `--scheduler random` to use random (instead of momentum) within-frame
  pixel ordering.
- Other run modes on the same file: `--benchmark` (full fgs × scheduler ×
  use_refrac sweep) and `--firing_nb_sweep` (firing_nb cap sweep).

---

## 6. Actual output of running it

All three commands below were executed against this exact file, this config,
and the venv above, on 2026-08-18.

### `--sync_first_layer`, 30 samples (smoke test)

```
Ranks=3, last_layer=2, layer_sizes=[2312, 128, 10]

[config] mode=sync-all-frames | fgs=128 | scheduler=momentum | use_refrac=True | n=30
Accuracy=100.00%  in=1038  hid_spk=263.2  hid_ev=1038.3  out_ev=263.2  t/sample=11.6ms  total=0.7s
```

### `--sync_first_layer`, 200 samples

```
Ranks=3, last_layer=2, layer_sizes=[2312, 128, 10]

[config] mode=sync-all-frames | fgs=128 | scheduler=momentum | use_refrac=True | n=200
Accuracy=98.50%  in=1035  hid_spk=262.6  hid_ev=1034.7  out_ev=262.6  t/sample=5.4ms  total=2.5s
```

### Default (`async+async`, no flag), 200 samples — for comparison

```
Ranks=3, last_layer=2, layer_sizes=[2312, 128, 10]

[config] mode=async+async | fgs=128 | scheduler=momentum | use_refrac=True | n=200
Accuracy=98.00%  in=1035  hid_spk=469.4  hid_ev=1034.7  out_ev=469.4  t/sample=20.5ms  total=2.5s
```

**Output field key:**
- `in` — avg. input pixel events sent per sample (rank 0)
- `hid_spk` — avg. spikes fired by the first hidden layer (rank 1) per sample
- `hid_ev` — avg. events received by rank 1
- `out_ev` — avg. events received by the output layer (rank 2)
- `t/sample` — wall-clock time per sample
- `total` — total wall-clock runtime for the run

**Observation:** sync-first-layer fires roughly **half** as many hidden
spikes as full async (262.6 vs 469.4 avg per sample) at the same accuracy
ballpark and same input, and runs noticeably faster per sample here (5.4ms
vs 20.5ms) — fewer MPI round trips because rank 1 only fires at frame
boundaries instead of continuously. These are small-sample (200/10,000)
numbers meant to confirm the run is wired correctly, not final accuracy —
run without `--max_samples` for the full 10k-sample number before trusting
any accuracy comparison.
