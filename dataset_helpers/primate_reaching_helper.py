import os
import math
import sys
import hashlib
import urllib.request
from urllib.error import URLError

import numpy as np
import h5py
from scipy.signal import convolve2d
import jax.numpy as jnp

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional


# Sampling rate of spike data (4ms interval) used in the raw recordings.
SAMPLING_RATE = 4e-3

# Zenodo URL hosting the public primate reaching dataset files.
ZENODO_URL = "https://zenodo.org/record/583331/files/"

# MD5 checksums for integrity verification (matches the source repository).
MD5_CHECKSUMS = {
    "indy_20170131_02.mat": "2790b1c869564afaa7772dbf9e42d784",
    "indy_20160630_01.mat": "197413a5339630ea926cbd22b8b43338",
    "indy_20160622_01.mat": "c33d5fff31320d709d23fe445561fb6e",
    "loco_20170301_05.mat": "47342da09f9c950050c9213c3df38ea3",
    "loco_20170215_02.mat": "739b70762d838f3a1f358733c426bb02",
    "loco_20170210_03.mat": "4cae63b58c4cb9c8abd44929216c703b",
}

# Custom user agent for polite dataset downloads.
USER_AGENT = "snn-training"


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute an MD5 checksum for integrity verification."""
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(fpath: str, md5: str = None) -> bool:
    """Return True if a file exists and (optionally) matches a given MD5."""
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return md5 == calculate_md5(fpath)


def _save_response_content(content, destination, length):
    """Stream-download a file while showing a progress bar."""
    try:
        from torch.utils.model_zoo import tqdm
    except ImportError:
        from tqdm import tqdm

    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            if not chunk:
                continue
            fh.write(chunk)
            pbar.update(len(chunk))


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024 * 32):
    """Download a URL with a custom user agent and chunked reads."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        _save_response_content(
            iter(lambda: response.read(chunk_size), b""),
            filename,
            length=response.length,
        )


def _get_redirect_url(url: str, max_hops: int = 3) -> str:
    """Resolve redirects (some Zenodo links redirect once or twice)."""
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

    for _ in range(max_hops + 1):
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request) as response:
            if response.url == url or response.url is None:
                return url
            url = response.url

    raise RecursionError(
        f"Request to {initial_url} exceeded {max_hops} redirects."
    )


def download_url(url: str, file_path: str, md5: str = None) -> None:
    """Download a file and verify it against the expected MD5 if provided."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if check_integrity(file_path, md5):
        print(f"Using downloaded and verified file: {file_path}")
        return

    url = _get_redirect_url(url)

    try:
        print(f"Downloading {url} to {file_path}")
        _urlretrieve(url, file_path)
    except (URLError, OSError) as e:
        if url.startswith("https"):
            url = url.replace("https:", "http:")
            print(f"HTTPS failed, trying HTTP: {url}")
            _urlretrieve(url, file_path)
        else:
            raise e

    if md5 and not check_integrity(file_path, md5):
        raise RuntimeError(f"Downloaded file {file_path} failed MD5 verification")


def ensure_dataset(data_dir: str, filename: str) -> str:
    """Ensure a dataset file exists locally (download if missing or corrupted)."""
    file_path = os.path.join(data_dir, filename)

    if os.path.isfile(file_path):
        md5 = MD5_CHECKSUMS.get(filename)
        if md5 and not check_integrity(file_path, md5):
            print(f"File {file_path} exists but failed MD5 check, re-downloading...")
        else:
            return file_path

    if filename not in MD5_CHECKSUMS:
        raise ValueError(
            f"Unknown dataset file: {filename}. "
            f"Available files: {list(MD5_CHECKSUMS.keys())}"
        )

    url = ZENODO_URL + filename
    md5 = MD5_CHECKSUMS[filename]
    download_url(url, file_path, md5)

    return file_path


def list_available_datasets() -> list:
    """Return all available dataset filenames."""
    return list(MD5_CHECKSUMS.keys())


def _expected_channel_count(filename: str, fallback: Optional[int] = None) -> int:
    if "indy" in filename:
        return 96
    if "loco" in filename:
        return 192
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not infer channel count from filename: {filename}")


def _normalize_spike_layout(spikes: np.ndarray, expected_channels: int) -> np.ndarray:
    """
    Normalize spike references to shape (channels, unit_slots).

    The Zenodo metadata describes the layout as (channels, units), but in practice the
    HDF5 arrays often appear as (unit_slots, channels) when loaded with h5py.
    """
    if spikes.ndim != 2:
        raise ValueError(f"Expected a 2D spikes array, got shape {spikes.shape}")

    if spikes.shape[0] == expected_channels:
        return spikes
    if spikes.shape[1] == expected_channels:
        return spikes.T

    raise ValueError(
        f"Could not normalize spikes layout with expected_channels={expected_channels}. "
        f"Observed shape: {spikes.shape}"
    )


def _load_spike_times(dataset: h5py.File, ref) -> np.ndarray:
    if isinstance(ref, np.ndarray):
        spike_times = ref
    else:
        spike_times = dataset[ref][()]
    return np.asarray(spike_times, dtype=np.float32).reshape(-1)


def _feature_index(channel_idx: int, unit_idx: int, unit_slots: int, collapse_units: bool) -> int:
    if collapse_units:
        return channel_idx
    return channel_idx * unit_slots + unit_idx


def _build_binned_spike_matrix(
    dataset: h5py.File,
    spike_refs: np.ndarray,
    time_bins: np.ndarray,
    collapse_units: bool,
) -> np.ndarray:
    num_channels, unit_slots = spike_refs.shape
    input_size = num_channels if collapse_units else num_channels * unit_slots
    spike_train = np.zeros((input_size, len(time_bins)), dtype=np.int16)

    for channel_idx in range(num_channels):
        for unit_idx in range(unit_slots):
            spike_times = _load_spike_times(dataset, spike_refs[channel_idx, unit_idx])
            if spike_times.size == 0:
                continue

            bins, _ = np.histogram(spike_times, bins=time_bins.squeeze())
            active_idx = np.nonzero(bins)[0] + 1
            if active_idx.size == 0:
                continue

            feature_idx = _feature_index(channel_idx, unit_idx, unit_slots, collapse_units)
            spike_train[feature_idx, active_idx] = 1

    return spike_train


def _build_raw_event_stream(
    dataset: h5py.File,
    spike_refs: np.ndarray,
    collapse_units: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a globally time-sorted raw event stream from the original spike times.

    Each event is represented by:
      - event_times[k]: spike timestamp
      - event_features[k]: input feature index (channel or channel-unit slot)
    """
    num_channels, unit_slots = spike_refs.shape
    all_times = []
    all_features = []

    for channel_idx in range(num_channels):
        for unit_idx in range(unit_slots):
            spike_times = _load_spike_times(dataset, spike_refs[channel_idx, unit_idx])
            if spike_times.size == 0:
                continue

            feature_idx = _feature_index(channel_idx, unit_idx, unit_slots, collapse_units)
            all_times.append(spike_times)
            all_features.append(np.full(spike_times.shape, feature_idx, dtype=np.int32))

    if not all_times:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    event_times = np.concatenate(all_times).astype(np.float32)
    event_features = np.concatenate(all_features).astype(np.int32)

    # Stable sort preserves the per-feature append order for tied timestamps.
    order = np.argsort(event_times, kind="stable")
    return event_times[order], event_features[order]


def _slice_dense_window(samples: np.ndarray, actual_idx: int, window: int, ratio: int) -> np.ndarray:
    start_idx = max(actual_idx - (window - 1) * ratio, 0)
    mask = np.arange(start_idx, actual_idx + 1, ratio, dtype=np.int64)
    x = samples[:, mask]

    if x.shape[1] < window:
        pad = np.zeros((samples.shape[0], window - x.shape[1]), dtype=samples.dtype)
        x = np.concatenate([pad, x], axis=1)

    return x


def _slice_label_window(labels: np.ndarray, actual_idx: int, window: int, ratio: int) -> np.ndarray:
    start_idx = max(actual_idx - (window - 1) * ratio, 0)
    mask = np.arange(start_idx, actual_idx + 1, ratio, dtype=np.int64)
    y = labels[:, mask]

    if y.shape[1] < window:
        pad = np.zeros((labels.shape[0], window - y.shape[1]), dtype=labels.dtype)
        y = np.concatenate([pad, y], axis=1)

    return y


def _select_label_output(label_window: np.ndarray, label_mode: str):
    if label_mode == "last":
        return label_window[:, -1]
    if label_mode == "mean":
        return np.mean(label_window, axis=1)
    if label_mode == "window":
        return label_window
    if label_mode == "last_x":
        return label_window[0, -1]
    if label_mode == "last_y":
        return label_window[1, -1]
    raise ValueError("label_mode must be one of: last, mean, window, last_x, last_y")


def _compute_exact_max_nonzero(
    event_times: np.ndarray,
    sample_times: np.ndarray,
    indices: np.ndarray,
    window: int,
    ratio: int,
) -> int:
    if indices.size == 0 or event_times.size == 0:
        return 0

    start_indices = np.maximum(indices - (window - 1) * ratio, 0)
    start_times = sample_times[start_indices]
    end_times = sample_times[indices]

    left = np.searchsorted(event_times, start_times, side="left")
    right = np.searchsorted(event_times, end_times, side="right")
    return int(np.max(right - left))


def _compute_binned_max_nonzero(
    samples: np.ndarray,
    indices: np.ndarray,
    window: int,
    ratio: int,
) -> int:
    """
    Compute the true maximum number of emitted events in any dense sample window.

    In binned mode, `_window_to_events` emits one event for each nonzero
    feature/time entry in the sampled window. This helper mirrors that logic
    without materializing every window explicitly.
    """
    if indices.size == 0 or samples.size == 0:
        return 0

    # One emitted event corresponds to one nonzero entry in the dense window.
    active_per_timestep = np.count_nonzero(samples, axis=0).astype(np.int64)

    if ratio <= 1:
        prefix = np.concatenate(([0], np.cumsum(active_per_timestep, dtype=np.int64)))
        start_indices = np.maximum(indices - (window - 1), 0)
        counts = prefix[indices + 1] - prefix[start_indices]
        return int(np.max(counts))

    # When ratio > 1, sampled windows step through every `ratio`-th column.
    # We compute prefix sums separately for each modulo class to match the
    # exact columns selected by `_slice_dense_window`.
    counts = np.empty(indices.shape[0], dtype=np.int64)
    for phase in range(ratio):
        phase_mask = (indices % ratio) == phase
        if not np.any(phase_mask):
            continue

        phase_indices = indices[phase_mask]
        phase_series = active_per_timestep[phase::ratio]
        phase_prefix = np.concatenate(([0], np.cumsum(phase_series, dtype=np.int64)))

        phase_positions = phase_indices // ratio
        start_positions = np.maximum(phase_positions - (window - 1), 0)
        counts[phase_mask] = (
            phase_prefix[phase_positions + 1] - phase_prefix[start_positions]
        )

    return int(np.max(counts))


def load_primate_reaching_data(
    data_dir: str,
    filename: str,
    bin_width: float = 0.004,
    stride: float = 0.004,
    train_ratio: float = 0.5,
    split_num: int = 1,
    download: bool = True,
    collapse_units: bool = True,
    preserve_exact_times: bool = False,
) -> dict:
    """
    Load a primate reaching session from .mat and convert to spike trains + velocity labels.

    Returns:
        samples: (features, timesteps) float32 spike counts or None in exact-time mode
        labels:  (2, timesteps) float32 velocity (dx, dy)
        indices: train/val/test time indices for sampling windows
    """
    # Resolve data path and optionally download.
    if download:
        file_path = ensure_dataset(data_dir, filename)
    else:
        file_path = os.path.join(data_dir, filename)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}. "
                f"Set download=True to fetch it from Zenodo."
            )

    print(f"Loading {filename}")

    # Load the HDF5/MAT file (MATLAB v7.3).
    dataset = h5py.File(file_path, "r")

    # Raw arrays:
    # spikes: spike-time references laid out as either (channels, unit_slots) or
    #         (unit_slots, channels), depending on how h5py exposes the MATLAB data.
    # cursor_pos: x/y cursor position
    # target_pos: target location used to segment reaches
    spikes = dataset["spikes"][()]
    cursor_pos = dataset["cursor_pos"][()]   # shape: (2, timesteps)
    target_pos = dataset["target_pos"][()]   # shape: (2, timesteps)
    t = np.squeeze(dataset["t"][()]).astype(np.float32)

    expected_channels = _expected_channel_count(filename, fallback=cursor_pos.shape[0])
    spike_refs = _normalize_spike_layout(spikes, expected_channels)
    num_channels, unit_slots = spike_refs.shape

    # Build a uniform time grid aligned to the recording.
    new_t = np.arange(t[0] - bin_width, t[-1], SAMPLING_RATE)
    ratio = int(np.round(bin_width / SAMPLING_RATE))

    if preserve_exact_times:
        samples = None
        event_times, event_features = _build_raw_event_stream(
            dataset=dataset,
            spike_refs=spike_refs,
            collapse_units=collapse_units,
        )
    else:
        spike_train = _build_binned_spike_matrix(
            dataset=dataset,
            spike_refs=spike_refs,
            time_bins=new_t,
            collapse_units=collapse_units,
        )
        if ratio != 1:
            spike_train = convolve2d(spike_train, np.ones((1, ratio)), mode="valid")
        samples = spike_train.astype(np.float32)
        event_times = None
        event_features = None

    labels = np.gradient(cursor_pos.astype(np.float32), axis=1).astype(np.float32)  # (2, timesteps)

    # Segment the recording into reach segments using target changes.
    target_diff = np.diff(target_pos, axis=1, append=target_pos[:, -1:])
    segment_boundaries = np.nonzero(np.sum(np.abs(target_diff), axis=0))[0]

    segment_boundaries = np.insert(segment_boundaries, 0, 0)
    segment_boundaries = np.append(segment_boundaries, target_pos.shape[1])
    time_segments = np.column_stack([segment_boundaries[:-1], segment_boundaries[1:]])

    # Split segments into train/val/test according to the requested ratios.
    total_segments = time_segments.shape[0]
    sub_length = int(total_segments / split_num)
    stride_steps = int(stride / SAMPLING_RATE)

    train_len = math.floor(train_ratio * sub_length)
    val_len = math.floor((sub_length - train_len) / 2)

    ind_train, ind_val, ind_test = [], [], []
    seg_train, seg_val, seg_test = [], [], []  # per-segment index lists

    for split_no in range(split_num):
        for i in range(sub_length):
            seg_idx = split_no * sub_length + i
            if seg_idx >= total_segments:
                break

            seg_start = time_segments[seg_idx, 0]
            seg_end = time_segments[seg_idx, 1]
            seg_indices = list(np.arange(seg_start, seg_end, stride_steps))

            if i < train_len:
                ind_train += seg_indices
                seg_train.append(seg_indices)
            elif train_len <= i < train_len + val_len:
                ind_val += seg_indices
                seg_val.append(seg_indices)
            else:
                ind_test += seg_indices
                seg_test.append(seg_indices)

    dataset.close()

    input_size = num_channels if collapse_units else num_channels * unit_slots

    if preserve_exact_times:
        print(
            f"Data loaded: {num_channels} channels, {unit_slots} unit slots, "
            f"{event_times.shape[0]} raw events"
        )
    else:
        print(
            f"Data loaded: {samples.shape[0]} features, {samples.shape[1]} timesteps "
            f"(channels={num_channels}, unit_slots={unit_slots})"
        )
    print(f"Split: train={len(ind_train)}, val={len(ind_val)}, test={len(ind_test)}")

    return {
        "samples": samples,
        "event_times": event_times,
        "event_features": event_features,
        "labels": labels,
        "ind_train": ind_train,
        "ind_val": ind_val,
        "ind_test": ind_test,
        "seg_train": seg_train,
        "seg_val": seg_val,
        "seg_test": seg_test,
        "ratio": ratio,
        "input_feature_size": input_size,
        "sample_times": t,
        "num_channels": num_channels,
        "unit_slots": unit_slots,
        "collapse_units": collapse_units,
        "preserve_exact_times": preserve_exact_times,
    }


class PrimateReachingDataset(Dataset):
    def __init__(
        self,
        samples: np.ndarray,
        labels: np.ndarray,
        indices: list,
        window: int,
        ratio: int,
        label_mode: str = "last",
    ):
        self.samples = samples
        self.labels = labels
        self.indices = indices
        self.window = window
        self.ratio = ratio
        self.label_mode = label_mode

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
        """Return a window of spikes and its corresponding velocity label(s)."""
        actual_idx = self.indices[idx]
        x = _slice_dense_window(self.samples, actual_idx, self.window, self.ratio)
        y = _slice_label_window(self.labels, actual_idx, self.window, self.ratio)
        return x, _select_label_output(y, self.label_mode)


class PrimateReachingExactEventsDataset(Dataset):
    def __init__(
        self,
        event_times: np.ndarray,
        event_features: np.ndarray,
        labels: np.ndarray,
        sample_times: np.ndarray,
        indices: list,
        window: int,
        ratio: int,
        label_mode: str = "last",
    ):
        self.event_times = event_times
        self.event_features = event_features
        self.labels = labels
        self.sample_times = sample_times
        self.indices = indices
        self.window = window
        self.ratio = ratio
        self.label_mode = label_mode

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
        actual_idx = self.indices[idx]
        start_idx = max(actual_idx - (self.window - 1) * self.ratio, 0)
        start_time = self.sample_times[start_idx]
        end_time = self.sample_times[actual_idx]

        left = np.searchsorted(self.event_times, start_time, side="left")
        right = np.searchsorted(self.event_times, end_time, side="right")

        if right > left:
            events = np.column_stack([
                self.event_features[left:right].astype(np.float32),
                np.ones((right - left,), dtype=np.float32),
            ])
        else:
            events = np.empty((0, 2), dtype=np.float32)

        y = _slice_label_window(self.labels, actual_idx, self.window, self.ratio)
        return events, _select_label_output(y, self.label_mode)


def _window_to_events(window_data: np.ndarray) -> np.ndarray:
    """Convert a dense (channels, window) spike window to event pairs (index, value)."""
    events = []
    channels, window = window_data.shape

    # Iterate timesteps in order to preserve temporal structure in events.
    for t in range(window):
        col = window_data[:, t]
        nonzero_idx = np.nonzero(col)[0]
        if nonzero_idx.size == 0:
            continue
        vals = col[nonzero_idx]
        for c, v in zip(nonzero_idx, vals):
            events.append([c, float(v)])

    if not events:
        return np.empty((0, 2), dtype=np.float32)

    return np.array(events, dtype=np.float32)


def primate_event_collate(batch, max_len: int, truncate: bool = False, input_is_event_list: bool = False):
    """Collate dense windows or raw event lists into padded (index, value) tensors."""
    data, labels = zip(*batch)
    padded_data = []

    for sample in data:
        events = sample if input_is_event_list else _window_to_events(sample)
        num_events = len(events)

        # Guard against extremely dense windows.
        if num_events > max_len:
            if truncate:
                events = events[:max_len]
                num_events = max_len
            else:
                raise NotImplementedError(
                    f"data size exceeds max_len: {num_events} > {max_len}"
                )

        # Pad with sentinel values (-2) to match existing async loaders.
        d_padded = np.full((max_len, 2), -2, dtype=np.float32)
        if num_events > 0:
            d_padded[:num_events, 0] = events[:, 0]
            d_padded[:num_events, 1] = events[:, 1]

        padded_data.append(d_padded)

    batch_array = jnp.array(padded_data, dtype=jnp.float32)
    label_array = jnp.array(labels, dtype=jnp.float32)

    return batch_array, label_array


def torch_primate_reaching_loader(
    batch_size: int,
    shuffle: bool = False,
    downsample: bool = False,
    CNN_preprocess: bool = False,
    data_dir: str = "",
    filename: str = "indy_20160622_01.mat",
    window: int = 50,
    bin_width: float = 0.004,
    stride: float = 0.004,
    train_ratio: float = 0.5,
    split_num: int = 1,
    download: bool = True,
    label_mode: str = "last",
    max_nonzero: Optional[int] = None,
    truncate: bool = False,
    collapse_units: bool = True,
    preserve_exact_times: bool = False,
):
    """
    Create train/val/test DataLoaders in the same format as other dataset helpers.
    Returns: (loader, n_batches) triplet + max_nonzero.
    """
    del downsample, CNN_preprocess

    # Resolve data directory layout to align with other datasets.
    if data_dir:
        base_dir = os.path.join(data_dir, "data", "primate_reaching")
    else:
        base_dir = os.path.join(".", "data", "primate_reaching")

    os.makedirs(base_dir, exist_ok=True)

    # Load and preprocess the raw primate reaching data.
    data = load_primate_reaching_data(
        data_dir=base_dir,
        filename=filename,
        bin_width=bin_width,
        stride=stride,
        train_ratio=train_ratio,
        split_num=split_num,
        download=download,
        collapse_units=collapse_units,
        preserve_exact_times=preserve_exact_times,
    )

    input_size = data["input_feature_size"]
    if max_nonzero is None:
        all_indices = np.asarray(
            data["ind_train"] + data["ind_val"] + data["ind_test"],
            dtype=np.int64,
        )
        if preserve_exact_times:
            max_nonzero = _compute_exact_max_nonzero(
                data["event_times"],
                data["sample_times"],
                all_indices,
                window,
                data["ratio"],
            )
        else:
            max_nonzero = _compute_binned_max_nonzero(
                data["samples"],
                all_indices,
                window,
                data["ratio"],
            )

    if preserve_exact_times:
        train_dataset = PrimateReachingExactEventsDataset(
            data["event_times"], data["event_features"], data["labels"], data["sample_times"],
            data["ind_train"], window, data["ratio"], label_mode=label_mode
        )
        val_dataset = PrimateReachingExactEventsDataset(
            data["event_times"], data["event_features"], data["labels"], data["sample_times"],
            data["ind_val"], window, data["ratio"], label_mode=label_mode
        )
        test_dataset = PrimateReachingExactEventsDataset(
            data["event_times"], data["event_features"], data["labels"], data["sample_times"],
            data["ind_test"], window, data["ratio"], label_mode=label_mode
        )
    else:
        train_dataset = PrimateReachingDataset(
            data["samples"], data["labels"], data["ind_train"],
            window, data["ratio"], label_mode=label_mode
        )
        val_dataset = PrimateReachingDataset(
            data["samples"], data["labels"], data["ind_val"],
            window, data["ratio"], label_mode=label_mode
        )
        test_dataset = PrimateReachingDataset(
            data["samples"], data["labels"], data["ind_test"],
            window, data["ratio"], label_mode=label_mode
        )

    collate_fn = lambda batch: primate_event_collate(
        batch,
        max_nonzero,
        truncate=truncate,
        input_is_event_list=preserve_exact_times,
    )

    trainloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    valloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    total_train_batches = len(trainloader)
    total_val_batches = len(valloader)
    total_test_batches = len(testloader)

    return (trainloader, total_train_batches), (valloader, total_val_batches), (testloader, total_test_batches), max_nonzero


def torch_primate_reaching_stateful_loader(
    data_dir: str = "",
    filename: str = "indy_20160622_01.mat",
    bin_width: float = 0.004,
    stride: float = 0.004,
    train_ratio: float = 0.5,
    split_num: int = 1,
    download: bool = True,
    collapse_units: bool = True,
    preserve_exact_times: bool = False,
    max_nonzero: Optional[int] = None,
):
    """
    Stateful loader for primate reaching: returns trials as ordered sequences of
    single-bin samples. Each element of a split is a list of (events, label) pairs
    representing one trial in time order.

    Returns:
        train_trials, val_trials, test_trials: list of trials, each trial is a list
            of (events_array, label_array) where events_array has shape (max_nonzero, 2)
            and label_array has shape (2,).
        max_nonzero: max events per single bin across all splits.
    """
    if data_dir:
        base_dir = os.path.join(data_dir, "data", "primate_reaching")
    else:
        base_dir = os.path.join(".", "data", "primate_reaching")
    os.makedirs(base_dir, exist_ok=True)

    data = load_primate_reaching_data(
        data_dir=base_dir,
        filename=filename,
        bin_width=bin_width,
        stride=stride,
        train_ratio=train_ratio,
        split_num=split_num,
        download=download,
        collapse_units=collapse_units,
        preserve_exact_times=preserve_exact_times,
    )

    ratio = data["ratio"]
    sample_times = data["sample_times"]

    # Build single-bin datasets (window=1) for each split.
    def _make_split_dataset(indices):
        if preserve_exact_times:
            return PrimateReachingExactEventsDataset(
                data["event_times"], data["event_features"], data["labels"],
                sample_times, indices, window=1, ratio=ratio, label_mode="last",
            )
        else:
            return PrimateReachingDataset(
                data["samples"], data["labels"], indices,
                window=1, ratio=ratio, label_mode="last",
            )

    train_ds = _make_split_dataset(data["ind_train"])
    val_ds   = _make_split_dataset(data["ind_val"])
    test_ds  = _make_split_dataset(data["ind_test"])

    # Compute max_nonzero across all splits if not provided.
    if max_nonzero is None:
        all_indices = np.asarray(
            data["ind_train"] + data["ind_val"] + data["ind_test"], dtype=np.int64
        )
        if preserve_exact_times:
            max_nonzero = _compute_exact_max_nonzero(
                data["event_times"], sample_times, all_indices, 1, ratio,
            )
        else:
            max_nonzero = _compute_binned_max_nonzero(
                data["samples"], all_indices, 1, ratio,
            )
        max_nonzero = max(max_nonzero, 1)

    def _build_trials_from_segments(seg_index_lists):
        """Build trials from pre-grouped segment index lists.
        Each segment is one trial; returns list of trials, each a list of (events, label)."""
        # Build a flat dataset and a flat->segment position mapping
        flat_indices = [idx for seg in seg_index_lists for idx in seg]
        if preserve_exact_times:
            flat_ds = PrimateReachingExactEventsDataset(
                data["event_times"], data["event_features"], data["labels"],
                sample_times, flat_indices, window=1, ratio=ratio, label_mode="last",
            )
        else:
            flat_ds = PrimateReachingDataset(
                data["samples"], data["labels"], flat_indices,
                window=1, ratio=ratio, label_mode="last",
            )

        trials = []
        pos = 0
        for seg in seg_index_lists:
            trial = []
            for _ in seg:
                x, y = flat_ds[pos]
                if not preserve_exact_times:
                    x = _window_to_events(x)
                num_events = len(x)
                padded = np.full((max_nonzero, 2), -2, dtype=np.float32)
                if num_events > 0:
                    use = min(num_events, max_nonzero)
                    padded[:use] = x[:use]
                trial.append((padded, np.asarray(y, dtype=np.float32)))
                pos += 1
            trials.append(trial)
        return trials

    train_trials = _build_trials_from_segments(data["seg_train"])
    val_trials   = _build_trials_from_segments(data["seg_val"])
    test_trials  = _build_trials_from_segments(data["seg_test"])

    return train_trials, val_trials, test_trials, max_nonzero


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone primate reaching loader test")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the test loaders")
    parser.add_argument("--data_dir", type=str, default="", help="Base directory for data (default: ./data/primate_reaching)")
    parser.add_argument("--filename", type=str, default="indy_20160622_01.mat", help="Session filename (.mat)")
    parser.add_argument("--window", type=int, default=50, help="Window length in timesteps")
    parser.add_argument("--bin_width", type=float, default=0.004, help="Spike bin width (seconds)")
    parser.add_argument("--stride", type=float, default=0.004, help="Stride between samples (seconds)")
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Train ratio per split")
    parser.add_argument("--split_num", type=int, default=1, help="Number of split chunks")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing")
    parser.add_argument("--label_mode", type=str, default="last", help="Label mode: last, mean, window, last_x, last_y")
    parser.add_argument("--max_nonzero", type=int, default=None, help="Override max_nonzero (events per sample)")
    parser.add_argument("--truncate", action="store_true", help="Truncate events if they exceed max_nonzero")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle train loader")
    parser.add_argument("--separate_units", action="store_true", help="Keep unit slots separated instead of collapsing them per channel")
    parser.add_argument("--preserve_exact_times", action="store_true", help="Preserve raw spike-time order instead of converting through time bins")

    args = parser.parse_args()

    # Build loaders and print a quick sanity check.
    train, val, test, max_nonzero = torch_primate_reaching_loader(
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        data_dir=args.data_dir,
        filename=args.filename,
        window=args.window,
        bin_width=args.bin_width,
        stride=args.stride,
        train_ratio=args.train_ratio,
        split_num=args.split_num,
        download=args.download,
        label_mode=args.label_mode,
        max_nonzero=args.max_nonzero,
        truncate=args.truncate,
        collapse_units=not args.separate_units,
        preserve_exact_times=args.preserve_exact_times,
    )

    trainloader, n_train = train
    valloader, n_val = val
    testloader, n_test = test

    print(
        f"Primate reaching loader ok. Batches -> train: {n_train}, val: {n_val}, test: {n_test}, "
        f"max_nonzero={max_nonzero}"
    )

    sample_x, sample_y = next(iter(trainloader))
    print(f"Sample batch x shape: {sample_x.shape}, y shape: {sample_y.shape}")
    print(f"Sample batch x(first): {sample_x[:1]}, labels (first 1): {sample_y[:1]}")
