import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
from pathlib import Path

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri', 'DejaVu Sans', 'Arial', 'Helvetica']



plot_framework_performance  = False
plot_example_graph = True

if plot_framework_performance:
    # Data from the table
    layers = np.array([3, 4, 5, 6, 7])

    # Different configurations: (Input activations, AA)
    mnist_avg_150 = np.array([119, 125, 140, 142, 140])  # MNIST avg, AA=150
    input_300_aa_150 = np.array([162, 147, 163, 162, 158])  # Input=300, AA=150
    input_150_aa_300 = np.array([162, 206, 216, 223, 225])  # Input=150, AA=300
    input_300_aa_300 = np.array([184, 207, 224, 231, 230])  # Input=300, AA=300
    input_600_aa_600 = np.array([317, 372, 386, 393, 399])  # Input=600, AA=600

    # Create the plot
    plt.figure(figsize=(12, 7))

    plt.plot(layers, mnist_avg_150, marker='o', linewidth=2, markersize=8, label='Input: 150, AA: 150')
    plt.plot(layers, input_300_aa_150, marker='s', linewidth=2, markersize=8, label='Input: 300, AA: 150')
    plt.plot(layers, input_150_aa_300, marker='^', linewidth=2, markersize=8, label='Input: 150, AA: 300')
    plt.plot(layers, input_300_aa_300, marker='D', linewidth=2, markersize=8, label='Input: 300, AA: 300')
    plt.plot(layers, input_600_aa_600, marker='*', linewidth=2, markersize=12, label='Input: 600, AA: 600')

    plt.xlabel('Number of Layers', fontsize=12, fontweight='bold')
    plt.ylabel('Time per Epoch (s)', fontsize=12, fontweight='bold')
    plt.title('Framework Performance vs Number of Layers', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(layers)

    plt.tight_layout()
    plt.savefig('framework_performance_time_per_epoch.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Also create the data as a dictionary for easy access
    data_dict = {
        'layers': layers,
        'MNIST_avg_150': mnist_avg_150,
        '300_150': input_300_aa_150,
        '150_300': input_150_aa_300,
        '300_300': input_300_aa_300,
        '600_600': input_600_aa_600
    }

    # Print the arrays for verification
    print("Data arrays:")
    print(f"Layers: {layers}")
    print(f"MNIST avg, AA=150: {mnist_avg_150}")
    print(f"Input=300, AA=150: {input_300_aa_150}")
    print(f"Input=150, AA=300: {input_150_aa_300}")
    print(f"Input=300, AA=300: {input_300_aa_300}")
    print(f"Input=600, AA=600: {input_600_aa_600}")


    # Stack all configurations by layer
    data = np.stack([mnist_avg_150, input_300_aa_300, input_600_aa_600], axis=1)  # shape (5 layers, 3 configs)
    print(data)
    x = [150, 300, 600]

    plt.figure(figsize=(8, 6))
    # for i, layer in enumerate(layers):
    #     plt.plot(x, data[i], marker='o', label=f'Layer {layer}')
    plt.plot(x, data[0], marker='o', linewidth=2, markersize=8, label='3 layers')
    plt.plot(x, data[1], marker='s', linewidth=2, markersize=8, label='4 layers')
    plt.plot(x, data[2], marker='^', linewidth=2, markersize=8, label='5 layers')
    plt.plot(x, data[3], marker='D', linewidth=2, markersize=8, label='6 layers')
    plt.plot(x, data[4], marker='*', linewidth=2, markersize=12, label='7 layers')

    plt.xlabel('Number of Activations (AA)')
    plt.ylabel('Time')
    plt.title('Relation between Number of Activations and Time per Layer')
    plt.legend(title='Layers')
    plt.grid(True)

    plt.xticks(x)

    plt.tight_layout()
    plt.savefig('framework_performance_act_vs_time.png', dpi=300, bbox_inches='tight')
    plt.show()

# if plot_example_graph:
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

def load_trace_spikes(trace_path: Path, display_names: dict):
    """Return {layer_name: [(time_s, abs_value), ...]} sorted by time."""
    if not trace_path.exists():
        return None

    with trace_path.open() as f:
        trace = json.load(f)

    layer_events = {}
    all_valid_times_us = []

    for rank_info in trace["ranks"].values():
        layer_idx = rank_info["layer_idx"]
        if layer_idx not in display_names:
            continue
        events = [
            (row[3], abs(row[2]))
            for row in rank_info["buffer"]
            if row[1] >= 0 and row[2] >= 0
        ]
        if not events:
            continue

        layer_events[layer_idx] = events
        all_valid_times_us.extend(t for t, _ in events)

    if not all_valid_times_us:
        return None

    start_time_us = min(all_valid_times_us)
    return {
        display_names[layer_idx]: sorted(
            [((t - start_time_us) / 1e6, v) for t, v in evts]
        )
        for layer_idx, evts in sorted(layer_events.items())
    }


def build_raster_plot(data, layers, t_end, save_path,
                      spike_height_min=0.10, spike_height_max=0.30,
                      spike_width=0.5, spike_color="#2c4f8a", ref_line=None):
    """ref_line: optional x-value (in seconds) to draw a vertical reference
    line spanning all subplots.  Spike heights are proportional to activation
    values, scaled between spike_height_min and spike_height_max."""
    y_pos = {name: len(layers) - i for i, name in enumerate(layers)}

    # Compute global min/max activation values across all data
    all_vals = []
    for spikes in data.values():
        for events in spikes.values():
            all_vals.extend(v for _, v in events)
    v_min, v_max = min(all_vals), max(all_vals)
    v_range = v_max - v_min if v_max > v_min else 1.0

    fig, axes = plt.subplots(len(data), 1, figsize=(8, 2.3 * len(data) + 0.5),
                             sharex=True)
    if len(data) == 1:
        axes = [axes]

    for ax, (title, spikes) in zip(axes, data.items()):
        for name, events in spikes.items():
            if name not in y_pos:
                continue
            y = y_pos[name]
            for t, v in events:
                h = spike_height_min + (v - v_min) / v_range * (spike_height_max - spike_height_min)
                ax.vlines(t, y - spike_height_max, y - spike_height_max + 2 * h,
                          lw=spike_width, color=spike_color)

        ax.set_xlim(0.0, t_end)
        ax.set_ylim(0.5, len(layers) + 0.5)
        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels(layers, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=6)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        ax.set_xticks([])

        # if ref_line is not None:
        #     ax.axvline(ref_line, color="red", ls="--", lw=0.6, zorder=5)

        ax.plot([0, 1], [0, 0], transform=ax.transAxes,
                color="black", lw=1.2, clip_on=False)
        ax.annotate("",
            xy=(1.03, 0), xycoords=("axes fraction", "axes fraction"),
            xytext=(0.99, 0), textcoords=("axes fraction", "axes fraction"),
            arrowprops=dict(arrowstyle="-|>", color="black",
                            lw=1.2, mutation_scale=14),
            annotation_clip=False)

    axes[0].set_xlabel("Time", fontsize=11, labelpad=8, loc="right")
    axes[-1].set_xlabel("Time", fontsize=11, labelpad=8, loc="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ── Two-hidden-layer raster plot ─────────────────────────────────────────────
TRACE_DIR = Path("network_results/mnist/inference/MLP")

TWO_HIDDEN_NAMES = {1: "Input Layer", 2: "Hidden Layer 1", 3: "Hidden Layer 2"}
TWO_HIDDEN_LAYERS = ["Input Layer", "Hidden Layer 1", "Hidden Layer 2"]

data_2h = {
    "Synchronous": load_trace_spikes(
        TRACE_DIR / "42_ep20_batch36_784_128_128_10_acc0.976_adam__event_timing_trace.json",
        TWO_HIDDEN_NAMES,
    ),
    # "Asynchronous": load_trace_spikes(
    #     TRACE_DIR / "42_ep20_batch36_784_128_128_10_acc0.974_adam__event_timing_trace.json",
    #     TWO_HIDDEN_NAMES,
    # ),
    "Asynchronous": load_trace_spikes(
        TRACE_DIR / "42_ep20_batch36_784_128_128_10_acc0.976_adam__sparse_event_timing_trace.json",
        TWO_HIDDEN_NAMES,
    ),
}

sparse_ref_2h = max(t for t, v in data_2h["Asynchronous"]["Hidden Layer 2"])
build_raster_plot(data_2h, TWO_HIDDEN_LAYERS, t_end=0.04,
                  save_path="raster_plots.png", ref_line=sparse_ref_2h)

# ── Single-hidden-layer raster plot ──────────────────────────────────────────
ONE_HIDDEN_NAMES = {0: "Input Layer", 1: "Hidden Layer", 2: "Output Layer"}
ONE_HIDDEN_LAYERS = ["Input Layer", "Hidden Layer", "Output Layer"]

data_1h = {
    "Synchronous": load_trace_spikes(
        TRACE_DIR / "42_ep20_b36_784_256_10_acc0.975_adam__event_timing_trace.json",
        ONE_HIDDEN_NAMES,
    ),
    "Asynchronous": load_trace_spikes(
        TRACE_DIR / "42_ep20_batch36_784_128_10_acc0.975_adam__event_timing_trace.json",
        ONE_HIDDEN_NAMES,
    ),
    "Asynchronous (sparse)": load_trace_spikes(
        TRACE_DIR / "42_ep50_b36_784_256_10_acc0.979_adam__event_timing_trace.json",
        ONE_HIDDEN_NAMES,
    ),
}

sparse_ref_1h = max(t for t, v in data_1h["Asynchronous (sparse)"]["Output Layer"])
build_raster_plot(data_1h, ONE_HIDDEN_LAYERS, t_end=0.05,
                  save_path="raster_plots_1hidden.png", ref_line=sparse_ref_1h)
