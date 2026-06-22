import os
from pathlib import Path
import shutil
import argparse
import subprocess

import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split


NCARS_SENSOR_SIZE = (120, 100, 2)  # x, y, polarity
NCARS_VAL_SPLIT = 0.1
NCARS_ARCHIVE_FILENAME = "Prophesee_Dataset_n_cars.7z"
NCARS_ARCHIVE_SPLIT_NAMES = {
    "train": ("train", "n-cars_train"),
    "test": ("test", "n-cars_test"),
}
NCARS_EXPECTED_SPLIT_COUNTS = {
    "train": {"cars": 7940, "background": 7482},
    "test": {"cars": 4396, "background": 4211},
}


PROPHESSEE_EVENT_DTYPE = np.dtype(
    [("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.bool_)]
)


def _class_to_label(folder_name: str) -> int:
    lower = folder_name.lower()
    if lower == "cars":
        return 0
    if lower == "background":
        return 1
    raise ValueError(f"Unknown NCARS class folder: {folder_name}")


def _skip_optional_dat_header(raw: bytes) -> int:
    offset = 0
    saw_ascii_header = False

    while offset < len(raw) and raw[offset:offset + 1] == b"%":
        eol = raw.find(b"\n", offset)
        if eol < 0:
            return len(raw)
        saw_ascii_header = True
        offset = eol + 1

    # DAT streams can include 2 bytes: event_type + event_size
    if saw_ascii_header and offset + 2 <= len(raw):
        event_size = raw[offset + 1]
        if event_size == 8:
            offset += 2

    return offset


def _read_prophesee_file(path: Path) -> np.ndarray:
    """
    Parse Prophesee NCARS events encoded as:
      uint32 timestamp + uint32 packed_data
    packed_data layout:
      bits 0..13  -> x
      bits 14..27 -> y
      bit 28      -> polarity
    """
    raw = path.read_bytes()
    offset = _skip_optional_dat_header(raw)
    payload = raw[offset:]

    if len(payload) < 8:
        return np.empty((0,), dtype=PROPHESSEE_EVENT_DTYPE)

    size = len(payload) - (len(payload) % 8)
    if size <= 0:
        return np.empty((0,), dtype=PROPHESSEE_EVENT_DTYPE)

    words = np.frombuffer(payload[:size], dtype="<u4")
    ts = words[0::2].astype(np.int64)
    packed = words[1::2]

    events = np.empty((ts.shape[0],), dtype=PROPHESSEE_EVENT_DTYPE)
    events["x"] = (packed & 0x3FFF).astype(np.int16)
    events["y"] = ((packed >> 14) & 0x3FFF).astype(np.int16)
    events["t"] = ts
    events["p"] = ((packed >> 28) & 1).astype(np.bool_)
    return events


def _downsample_events_2x(events: np.ndarray) -> np.ndarray:
    if events.size == 0:
        return events

    out = events.copy()
    out["x"] = (out["x"].astype(np.int32) // 2).astype(np.int16)
    out["y"] = (out["y"].astype(np.int32) // 2).astype(np.int16)
    return out


def _dedup_events(events: np.ndarray) -> np.ndarray:
    if events.size == 0:
        return events
    keys = np.stack([events["x"].astype(np.int32), events["y"].astype(np.int32), events["p"].astype(np.int32)], axis=1)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    return events[np.sort(first_indices)]


def _augment_events(events: np.ndarray, x_size: int, y_size: int) -> np.ndarray:
    """Random horizontal flip + random translation (±4 px), drops out-of-bounds events."""
    if events.size == 0:
        return events
    out = events.copy()
    if np.random.rand() > 0.5:
        out["x"] = (x_size - 1 - out["x"].astype(np.int32)).astype(np.int16)
    dx = np.random.randint(-4, 5)
    dy = np.random.randint(-4, 5)
    x = out["x"].astype(np.int32) + dx
    y = out["y"].astype(np.int32) + dy
    valid = (x >= 0) & (x < x_size) & (y >= 0) & (y < y_size)
    out = out[valid]
    out["x"] = x[valid].astype(np.int16)
    out["y"] = y[valid].astype(np.int16)
    return out


def _resolve_ncars_root(data_dir: str) -> Path:
    base = Path(data_dir) if data_dir else Path(".")
    target_root = base / "data" / "ncars"
    target_root.mkdir(parents=True, exist_ok=True)

    _prepare_ncars_from_local_archive(target_root)
    _validate_ncars_split_counts(target_root)
    return target_root


def _count_dat_files(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".dat")


def _collect_ncars_split_counts(root: Path):
    counts = {}
    for split_name, class_targets in NCARS_EXPECTED_SPLIT_COUNTS.items():
        split_dir = root / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing NCARS split directory: {split_dir}")
        counts[split_name] = {}
        for class_name in class_targets:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing NCARS class directory: {class_dir}")
            counts[split_name][class_name] = _count_dat_files(class_dir)
    return counts


def _validate_ncars_split_counts(root: Path):
    counts = _collect_ncars_split_counts(root)
    mismatches = []
    for split_name, class_targets in NCARS_EXPECTED_SPLIT_COUNTS.items():
        for class_name, expected_count in class_targets.items():
            actual_count = counts[split_name][class_name]
            if actual_count != expected_count:
                mismatches.append(
                    f"{split_name}/{class_name}: expected {expected_count}, found {actual_count}"
                )

    if mismatches:
        raise RuntimeError(
            "NCARS sample count mismatch:\n"
            + "\n".join(mismatches)
            + "\nExpected counts are train(cars=7940, background=7482) and "
            "test(cars=4396, background=4211)."
        )
    return counts


def _prepare_ncars_from_local_archive(target_root: Path) -> None:
    archive_path = target_root / NCARS_ARCHIVE_FILENAME

    try:
        _validate_ncars_split_counts(target_root)
        return
    except (FileNotFoundError, RuntimeError):
        pass

    if _has_train_test_folders(target_root):
        _normalize_local_split_dirs(target_root)
        return

    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Could not find NCARS archive at {archive_path}. "
            f"Place {NCARS_ARCHIVE_FILENAME} in data/ncars."
        )

    tmp_dir = target_root.parent / "_ncars_tmp_extract"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Extracting NCARS archive: {archive_path}")
        _extract_archive(archive_path, tmp_dir)
        found = _find_root_with_train_test(tmp_dir)
        if found is None:
            raise RuntimeError(
                "Archive extracted but train/test folders were not found inside it."
            )

        for split_name in ("train", "test"):
            src = _resolve_archive_split_dir(found, split_name)
            dst = target_root / split_name
            if src is None:
                expected_names = ", ".join(NCARS_ARCHIVE_SPLIT_NAMES[split_name])
                raise RuntimeError(
                    f"Missing '{split_name}' split in extracted archive root {found}. "
                    f"Expected one of: {expected_names}"
                )
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _scan_split(split_dir: Path, allow_empty: bool = False):
    split_name = split_dir.name.lower()
    if split_name not in NCARS_EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unexpected NCARS split folder: {split_dir}")

    samples = []
    for class_name in NCARS_EXPECTED_SPLIT_COUNTS[split_name]:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            if allow_empty:
                continue
            raise FileNotFoundError(f"Missing NCARS class directory: {class_dir}")

        label = _class_to_label(class_name)
        for file_path in sorted(class_dir.rglob("*")):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            if file_path.suffix.lower() != ".dat":
                continue
            samples.append((file_path, label))

    if not samples and not allow_empty:
        raise FileNotFoundError(f"No NCARS samples found under {split_dir}")
    return samples


class NCARSPropheseeDataset(Dataset):
    def __init__(self, root: Path, train: bool = True, downsample: bool = False, dedup: bool = False, allow_empty: bool = False):
        split_name = "train" if train else "test"
        split_dir = _resolve_split_dir(root, split_name)

        self.samples = _scan_split(split_dir, allow_empty=allow_empty)
        self.downsample = downsample
        self.dedup = dedup

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        events = _read_prophesee_file(file_path)
        if self.downsample:
            events = _downsample_events_2x(events)
        if self.dedup:
            events = _dedup_events(events)
        return events, label


def _has_train_test_folders(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    return (
        _resolve_archive_split_dir(root, "train") is not None
        and _resolve_archive_split_dir(root, "test") is not None
    )


def _resolve_archive_split_dir(root: Path, split_name: str) -> Path | None:
    for candidate_name in NCARS_ARCHIVE_SPLIT_NAMES[split_name]:
        candidate = root / candidate_name
        if candidate.is_dir():
            return candidate
    return None


def _normalize_local_split_dirs(root: Path) -> None:
    for split_name in ("train", "test"):
        src = _resolve_archive_split_dir(root, split_name)
        if src is None:
            expected_names = ", ".join(NCARS_ARCHIVE_SPLIT_NAMES[split_name])
            raise FileNotFoundError(
                f"Missing '{split_name}' split under {root}. Expected one of: {expected_names}"
            )
        dst = root / split_name
        if src.resolve() == dst.resolve():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))


def _resolve_split_dir(root: Path, split: str) -> Path:
    split_name = split.lower()
    split_dir = root / split_name
    if split_dir.exists() and split_dir.is_dir():
        return split_dir
    raise FileNotFoundError(
        f"Could not find NCARS '{split_name}' split at {split_dir}. "
        f"Expected {root / 'train'} and {root / 'test'}."
    )


def _find_root_with_train_test(search_root: Path) -> Path | None:
    if _has_train_test_folders(search_root):
        return search_root
    for child in search_root.rglob("*"):
        if child.is_dir() and _has_train_test_folders(child):
            return child
    return None


def _extract_archive(archive_path: Path, dst_dir: Path) -> None:
    if archive_path.suffix.lower() != ".7z":
        raise RuntimeError(
            f"Expected a .7z NCARS archive, got: {archive_path}"
        )

    try:
        import py7zr  # type: ignore

        with py7zr.SevenZipFile(archive_path, mode="r") as zf:
            zf.extractall(path=dst_dir)
        return
    except ModuleNotFoundError:
        seven_zip_bin = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
        if seven_zip_bin is None:
            raise RuntimeError(
                f"Cannot extract {archive_path.name}: this is a .7z archive and neither "
                "'py7zr' nor a 7zip binary (7z/7za/7zr) is available. "
                "Install py7zr in your venv or install a 7zip binary."
            )
        subprocess.run(
            [seven_zip_bin, "x", str(archive_path), f"-o{dst_dir}", "-y"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


def _dummy_preprocess_sample(events: np.ndarray, cnn_preprocess: bool, x_size: int, y_size: int) -> np.ndarray:
    """
    Dummy preprocessing pass used only for one-time max-length computation.
    """
    num_events = len(events)
    if cnn_preprocess:
        sample = np.empty((num_events, 4), dtype=np.int32)
        sample[:, 0] = events["p"].astype(np.int32)
        sample[:, 1] = events["x"].astype(np.int32)
        sample[:, 2] = events["y"].astype(np.int32)
        sample[:, 3] = 1
        return sample

    p = events["p"].astype(np.int32)
    x = events["x"].astype(np.int32)
    y = events["y"].astype(np.int32)
    neuron_idx = p * (x_size * y_size) + x * y_size + y
    sample = np.empty((num_events, 2), dtype=np.int32)
    sample[:, 0] = neuron_idx
    sample[:, 1] = 1
    return sample


def _compute_max_data_length_once(trainset, testset, cache_dir: str, cnn_preprocess: bool, x_size: int, y_size: int, dedup: bool = False) -> int:
    mode = "cnn" if cnn_preprocess else "mlp"
    dedup_suffix = "_dedup" if dedup else ""
    cache_file = Path(cache_dir) / f"max_data_length_{mode}_{x_size}x{y_size}{dedup_suffix}.txt"
    if cache_file.exists():
        try:
            cached = int(cache_file.read_text().strip())
            if cached > 0:
                return cached
        except ValueError:
            pass

    max_len = 0
    for dataset in (trainset, testset):
        for i in range(len(dataset)):
            events, _ = dataset[i]
            preprocessed = _dummy_preprocess_sample(events, cnn_preprocess, x_size, y_size)
            max_len = max(max_len, preprocessed.shape[0])

    if max_len <= 0:
        raise RuntimeError("Failed to compute NCARS max data length.")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(str(max_len))
    return max_len


def basic_event_collate(batch):
    events, labels = zip(*batch)
    return list(events), np.array(labels)


def custom_event_pad_collate(batch, max_len):
    """
    Returns data in shape (B, max_len, 4), each event as [p, x, y, 1].
    """
    data, labels = zip(*batch)
    padded_data = []

    for d in data:
        num_events = len(d)
        if num_events > max_len:
            raise ValueError(f"NCARS sample length {num_events} exceeds max_len {max_len}")

        d_padded = np.full((max_len, 4), -2, dtype=np.int32)
        d_padded[:num_events, 0] = d["p"].astype(np.int32)
        d_padded[:num_events, 1] = d["x"].astype(np.int32)
        d_padded[:num_events, 2] = d["y"].astype(np.int32)
        d_padded[:num_events, 3] = 1
        padded_data.append(d_padded)

    batch_array = jnp.array(padded_data, dtype=jnp.int32)
    label_array = jnp.array(labels, dtype=jnp.int32)
    return batch_array, label_array


def augmenting_event_pad_collate(batch, max_len, x_size, y_size):
    """
    Like custom_event_pad_collate but applies random flip + translation per sample.
    Only used for the train loader.
    """
    data, labels = zip(*batch)
    padded_data = []

    for d in data:
        d = _augment_events(d, x_size, y_size)
        num_events = len(d)

        d_padded = np.full((max_len, 4), -2, dtype=np.int32)
        d_padded[:num_events, 0] = d["p"].astype(np.int32)
        d_padded[:num_events, 1] = d["x"].astype(np.int32)
        d_padded[:num_events, 2] = d["y"].astype(np.int32)
        d_padded[:num_events, 3] = 1
        padded_data.append(d_padded)

    batch_array = jnp.array(padded_data, dtype=jnp.int32)
    label_array = jnp.array(labels, dtype=jnp.int32)
    return batch_array, label_array


def custom_preprocess_event_pad_collate(batch, max_len, x_size, y_size):
    """
    Returns data in shape (B, max_len, 2), each event as [neuron_idx, 1].
    """
    data, labels = zip(*batch)
    padded_data = []

    for d in data:
        num_events = len(d)
        if num_events > max_len:
            raise ValueError(f"NCARS sample length {num_events} exceeds max_len {max_len}")

        d_padded = np.full((max_len, 2), -2, dtype=np.int32)
        p = d["p"].astype(np.int32)
        x = d["x"].astype(np.int32)
        y = d["y"].astype(np.int32)
        neuron_idx = p * (x_size * y_size) + x * y_size + y

        d_padded[:num_events, 0] = neuron_idx
        d_padded[:num_events, 1] = 1
        padded_data.append(d_padded)

    batch_array = jnp.array(padded_data, dtype=jnp.int32)
    label_array = jnp.array(labels, dtype=jnp.int32)
    return batch_array, label_array


def augmenting_preprocess_event_pad_collate(batch, max_len, x_size, y_size):
    """
    Like custom_preprocess_event_pad_collate but applies random flip + translation
    per sample. Only used for the train loader.
    """
    data, labels = zip(*batch)
    padded_data = []

    for d in data:
        d = _augment_events(d, x_size, y_size)
        num_events = len(d)

        d_padded = np.full((max_len, 2), -2, dtype=np.int32)
        p = d["p"].astype(np.int32)
        x = d["x"].astype(np.int32)
        y = d["y"].astype(np.int32)
        neuron_idx = p * (x_size * y_size) + x * y_size + y

        d_padded[:num_events, 0] = neuron_idx
        d_padded[:num_events, 1] = 1
        padded_data.append(d_padded)

    batch_array = jnp.array(padded_data, dtype=jnp.int32)
    label_array = jnp.array(labels, dtype=jnp.int32)
    return batch_array, label_array


def _events_to_dense_matrix(events, x_size, y_size):
    """
    Aggregate all events from one sample into a dense (C, H, W) matrix.
    Channel 0 and 1 correspond to the two polarities.
    """
    frame = np.zeros((NCARS_SENSOR_SIZE[2], x_size, y_size), dtype=np.float32)
    if len(events) == 0:
        return frame

    p = events["p"].astype(np.intp)
    x = events["x"].astype(np.intp)
    y = events["y"].astype(np.intp)

    valid = (
        (p >= 0)
        & (p < NCARS_SENSOR_SIZE[2])
        & (x >= 0)
        & (x < x_size)
        & (y >= 0)
        & (y < y_size)
    )
    if np.any(valid):
        np.add.at(frame, (p[valid], x[valid], y[valid]), 1.0)
    return frame


def dense_matrix_collate(batch, x_size, y_size):
    """
    Returns a dense batch in shape (B, C, H, W) for regular CNN training.
    Each matrix entry stores the event count accumulated at that location.
    """
    data, labels = zip(*batch)
    batch_array = np.stack(
        [_events_to_dense_matrix(events, x_size, y_size) for events in data],
        axis=0,
    )
    label_array = np.asarray(labels, dtype=np.int32)
    return batch_array, label_array

# region ncars loader
def torch_NCARS_loader(
    batch_size,
    CNN_preprocess=False,
    shuffle=False,
    downsample=False,
    dedup=False,
    augment=False,
    data_dir="",
    full_matrix=False,
):
    """
    NCARS dataloader.

    By default it returns the existing event-based formats used by the async
    models. Set ``full_matrix=True`` to aggregate each sample into a dense
    ``(C, H, W)`` matrix for regular CNN training.
    """
    if data_dir:
        base_cache_dir = os.path.join(data_dir, "cache/NCARS")
    else:
        base_cache_dir = "./cache/NCARS"

    os.makedirs(base_cache_dir, exist_ok=True)

    root = _resolve_ncars_root(data_dir)
    print(f"Loading NCARS dataset from: {root}")
    split_counts = _validate_ncars_split_counts(root)
    print(
        "NCARS verified counts -> "
        f"train: cars={split_counts['train']['cars']}, background={split_counts['train']['background']}; "
        f"test: cars={split_counts['test']['cars']}, background={split_counts['test']['background']}"
    )

    trainset = NCARSPropheseeDataset(root, train=True, downsample=downsample, dedup=dedup, allow_empty=False)
    testset = NCARSPropheseeDataset(root, train=False, downsample=downsample, dedup=dedup, allow_empty=False)

    n_train_raw = len(trainset)
    n_test_raw = len(testset)
    print(f"NCARS split sizes on disk -> train: {n_train_raw}, test: {n_test_raw}")
    expected_train_total = sum(NCARS_EXPECTED_SPLIT_COUNTS["train"].values())
    expected_test_total = sum(NCARS_EXPECTED_SPLIT_COUNTS["test"].values())
    if n_train_raw != expected_train_total or n_test_raw != expected_test_total:
        raise RuntimeError(
            "NCARS loaded sample totals do not match expected values: "
            f"train expected {expected_train_total}, loaded {n_train_raw}; "
            f"test expected {expected_test_total}, loaded {n_test_raw}."
        )

    train_len = int(len(trainset) * (1 - NCARS_VAL_SPLIT))
    val_len = len(trainset) - train_len
    train_subset, val_subset = random_split(trainset, [train_len, val_len])

    x_size = NCARS_SENSOR_SIZE[0] // 2 if downsample else NCARS_SENSOR_SIZE[0]
    y_size = NCARS_SENSOR_SIZE[1] // 2 if downsample else NCARS_SENSOR_SIZE[1]
    if full_matrix:
        max_data_length = x_size * y_size * NCARS_SENSOR_SIZE[2]
        collate_fn = lambda batch: dense_matrix_collate(batch, x_size, y_size)
        train_collate_fn = collate_fn
    else:
        max_data_length = _compute_max_data_length_once(
            trainset,
            testset,
            base_cache_dir,
            bool(CNN_preprocess),
            x_size,
            y_size,
            dedup=dedup,
        )
        if CNN_preprocess:
            collate_fn = lambda batch: custom_event_pad_collate(batch, max_data_length)
            train_collate_fn = (
                lambda batch: augmenting_event_pad_collate(batch, max_data_length, x_size, y_size)
                if augment else collate_fn
            )
        elif CNN_preprocess is None:
            collate_fn = basic_event_collate
            train_collate_fn = collate_fn
        else:
            collate_fn = lambda batch: custom_preprocess_event_pad_collate(
                batch, max_data_length, x_size, y_size
            )
            if augment:
                train_collate_fn = lambda batch: augmenting_preprocess_event_pad_collate(
                    batch, max_data_length, x_size, y_size
                )
            else:
                train_collate_fn = collate_fn

    trainloader = DataLoader(train_subset, batch_size=batch_size, collate_fn=train_collate_fn, shuffle=shuffle)
    valloader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    total_train_batches = len(trainloader)
    total_val_batches = len(valloader)
    total_test_batches = len(testloader)

    return (
        (trainloader, total_train_batches),
        (valloader, total_val_batches),
        (testloader, total_test_batches),
        max_data_length,
    )


def _run_standalone_test():
    parser = argparse.ArgumentParser(description="Standalone NCARS loader test")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the test loaders")
    parser.add_argument("--data_dir", type=str, default="", help="Project root containing data/ and cache/")
    parser.add_argument("--downsample", action="store_true", help="Use 60x50 downsampled events")
    parser.add_argument(
        "--cnn_preprocess",
        action="store_true",
        help="Use CNN event format [p,x,y,1] instead of flattened indices",
    )
    parser.add_argument(
        "--full_matrix",
        action="store_true",
        help="Return dense (C, H, W) matrices for regular CNN training",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle train loader")
    args = parser.parse_args()

    print(
        "NCARS test config:",
        {
            "batch_size": args.batch_size,
            "data_dir": args.data_dir,
            "downsample": args.downsample,
            "cnn_preprocess": args.cnn_preprocess,
            "full_matrix": args.full_matrix,
            "shuffle": args.shuffle,
            "archive": NCARS_ARCHIVE_FILENAME,
            "expected_counts": NCARS_EXPECTED_SPLIT_COUNTS,
        },
    )

    try:
        train, val, test, max_data_length = torch_NCARS_loader(
            batch_size=args.batch_size,
            CNN_preprocess=args.cnn_preprocess,
            shuffle=args.shuffle,
            downsample=args.downsample,
            data_dir=args.data_dir,
            full_matrix=args.full_matrix,
        )
        trainloader, n_train = train
        valloader, n_val = val
        testloader, n_test = test

        print(
            f"NCARS loader ok. Batches -> train: {n_train}, val: {n_val}, test: {n_test}, "
            f"max_data_length: {max_data_length}"
        )

        sample_x, sample_y = next(iter(trainloader))
        print(f"First train batch shapes: x={tuple(sample_x.shape)}, y={tuple(sample_y.shape)}")
        print(f"First train batch labels (first 16): {np.array(sample_y)[:16].tolist()}")
        print(f"len(trainloader)={len(trainloader)}, len(valloader)={len(valloader)}, len(testloader)={len(testloader)}")
    except Exception as e:
        print("\nNCARS standalone test failed.")
        print(f"{type(e).__name__}: {e}")
        print(
            f"\nExpected archive path: data/ncars/{NCARS_ARCHIVE_FILENAME}\n"
            "If extraction fails, install py7zr in your venv or install a 7zip binary (7z/7za/7zr)."
        )
        raise


if __name__ == "__main__":
    _run_standalone_test()
