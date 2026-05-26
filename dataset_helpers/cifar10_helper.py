import numpy as np
import os
try:
    import dataset_helpers.network_helper as network_helper
except ModuleNotFoundError:
    import network_helper

# CIFAR-10 per-channel stats (standard values, computed on training set)
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

IMG_SIZE = 32
NUM_CHANNELS = 3
NUM_PIXELS = IMG_SIZE * IMG_SIZE  # 1024 spatial positions
NUM_FEATURES = NUM_PIXELS * NUM_CHANNELS  # 3072 values total


def download_cifar10(dataset_folder):
    """
    Download CIFAR-10 using torchvision and convert to a simple numpy format.
    Stored as (N, 32, 32, 3) uint8 arrays with labels.
    """
    try:
        from torchvision import datasets
    except ImportError:
        raise ImportError(
            "PyTorch / torchvision is required for automatic CIFAR-10 download. "
            "Install with: pip install torch torchvision"
        )

    os.makedirs(dataset_folder, exist_ok=True)

    train_npz = os.path.join(dataset_folder, 'cifar10_train.npz')
    test_npz = os.path.join(dataset_folder, 'cifar10_test.npz')

    if os.path.exists(train_npz) and os.path.exists(test_npz):
        print("CIFAR-10 numpy files already exist")
        return

    print("Downloading CIFAR-10 dataset using torchvision...")
    torch_data_dir = os.path.join(dataset_folder, 'torch_tmp')

    train_dataset = datasets.CIFAR10(root=torch_data_dir, train=True, download=True)
    test_dataset = datasets.CIFAR10(root=torch_data_dir, train=False, download=True)

    train_images = np.array(train_dataset.data, dtype=np.uint8)  # (50000, 32, 32, 3)
    train_labels = np.array(train_dataset.targets, dtype=np.int64)
    test_images = np.array(test_dataset.data, dtype=np.uint8)    # (10000, 32, 32, 3)
    test_labels = np.array(test_dataset.targets, dtype=np.int64)

    np.savez_compressed(train_npz, x=train_images, y=train_labels)
    np.savez_compressed(test_npz, x=test_images, y=test_labels)

    import shutil
    if os.path.exists(torch_data_dir):
        shutil.rmtree(torch_data_dir)

    print("CIFAR-10 dataset downloaded and saved successfully!")


def flatten_row_major(images):
    """
    Turn (N, 32, 32, 3) uint8 images into (N, 32*32*3) float32 values in [0,1],
    ordered row by row, column by column, channel by channel at each pixel.

    Emission order for each image: for row in 0..31, for col in 0..31, for ch in (R,G,B).
    This means the network sees the whole image pixel by pixel, row by row.
    """
    images = images.astype(np.float32) / 255.0
    # images shape is already (N, H, W, C) with row-major H then W then C,
    # so a plain reshape(N, -1) gives row by row, col by col, RGB per pixel.
    return images.reshape(images.shape[0], -1)


def normalize_per_channel(flat_images):
    """
    Apply (x - mean) / std per channel on a flat (N, H*W*C) array that still has
    RGB interleaved at each pixel (i.e. stride-3 channel dimension).
    """
    # reshape to (N, H*W, 3) to broadcast per-channel mean/std
    n = flat_images.shape[0]
    reshaped = flat_images.reshape(n, NUM_PIXELS, NUM_CHANNELS)
    reshaped = (reshaped - CIFAR10_MEAN) / CIFAR10_STD
    return reshaped.reshape(n, NUM_FEATURES)


def preprocess_dataset_sequential(dataset_x):
    """
    Apply sequential preprocessing to the whole dataset.

    dataset_x: shape (N, NUM_FEATURES), already normalized.
    Returns: shape (N, NUM_FEATURES, 2), with column 0 holding the input-neuron
    index (0..NUM_FEATURES-1 in row-major, RGB-interleaved order) and column 1
    the normalized value — analogous to MNIST's non-sequential (idx, value)
    format but with every pixel/channel emitted, so the network sees the whole
    image row by row.
    """
    N = dataset_x.shape[0]
    out = np.zeros((N, NUM_FEATURES, 2), dtype=np.float32)
    out[:, :, 0] = np.arange(NUM_FEATURES, dtype=np.float32)
    out[:, :, 1] = dataset_x
    return out


def preprocess_dataset_CNN(dataset_x):
    """
    Apply CNN preprocessing to the whole dataset.

    dataset_x: shape (N, NUM_FEATURES), already normalized, RGB interleaved per
    pixel in row-major order (same layout as `preprocess_dataset_sequential`).
    Returns: shape (N, NUM_FEATURES, 4), each event formatted as (channel, row,
    col, value) — matching the CNN input format used by [async_CNN.py] and
    mirroring `preprocess_dataset_CNN` in [dataset_helpers/mnist_helper.py].

    Emission order per image: for row r in 0..31, for col c in 0..31, for
    channel ch in (R=0, G=1, B=2): event (ch, r, c, v). So the network sees
    every pixel's RGB triple together, row by row.
    """
    N = dataset_x.shape[0]
    # Build the (channel, row, col) index table once.
    rows = np.repeat(np.arange(IMG_SIZE, dtype=np.float32), IMG_SIZE * NUM_CHANNELS)
    cols = np.tile(np.repeat(np.arange(IMG_SIZE, dtype=np.float32), NUM_CHANNELS), IMG_SIZE)
    chans = np.tile(np.arange(NUM_CHANNELS, dtype=np.float32), NUM_PIXELS)

    out = np.zeros((N, NUM_FEATURES, 4), dtype=np.float32)
    out[:, :, 0] = chans
    out[:, :, 1] = rows
    out[:, :, 2] = cols
    out[:, :, 3] = dataset_x
    return out


def cifar10_loader_manual(batch_size,
                          shuffle=False,
                          preprocess=True,
                          CNN_preprocess=False,
                          normalize=True,
                          data_dir="",
                          cache_dir='./cache/cifar10',
                          **_ignored):
    """
    CIFAR-10 loader that serves every pixel in order, row by row.

    For each image, features are ordered: row 0 col 0 R, row 0 col 0 G, row 0 col 0 B,
    row 0 col 1 R, ..., row 31 col 31 B. Total of 32*32*3 = 3072 values per image.

    With `preprocess=True`:
      - `CNN_preprocess=False` (MLP): each sample is shaped (3072, 2) as
        (input_neuron_index, value) — analogous to MNIST's non-sequential format
        but with every pixel/channel emitted, so the network sees the whole
        image row by row with 3072 input neurons.
      - `CNN_preprocess=True` (CNN): each sample is shaped (3072, 4) as
        (channel, row, col, value) — matches the CNN input format expected by
        [async_CNN.py] / mirroring [dataset_helpers/mnist_helper.py].
    """
    dataset_folder = os.path.join(data_dir, "data/cifar10/")
    cache_dir = os.path.join(data_dir, cache_dir)

    download_cifar10(dataset_folder)

    if preprocess:
        suffix = "/async_CNN" if CNN_preprocess else "/async_MLP_sequential"
        cache_dir += suffix if normalize else suffix + "_no_norm"

    os.makedirs(cache_dir, exist_ok=True)
    train_cache_path = os.path.join(cache_dir, 'train.npz')
    test_cache_path = os.path.join(cache_dir, 'test.npz')

    if preprocess and os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
        print("Loading cached CIFAR-10 dataset")
        data = np.load(train_cache_path)
        cifar_x, cifar_y = data['x'], data['y']
        data = np.load(test_cache_path)
        cifar_x_test, cifar_y_test = data['x'], data['y']
    else:
        train_raw = np.load(os.path.join(dataset_folder, 'cifar10_train.npz'))
        test_raw = np.load(os.path.join(dataset_folder, 'cifar10_test.npz'))

        cifar_x = flatten_row_major(train_raw['x'])
        cifar_y = train_raw['y']
        cifar_x_test = flatten_row_major(test_raw['x'])
        cifar_y_test = test_raw['y']

        if normalize:
            cifar_x = normalize_per_channel(cifar_x)
            cifar_x_test = normalize_per_channel(cifar_x_test)

        if preprocess:
            if CNN_preprocess:
                print('Preprocessing CIFAR-10 dataset (CNN, row-by-row, RGB per pixel)')
                cifar_x = preprocess_dataset_CNN(cifar_x)
                cifar_x_test = preprocess_dataset_CNN(cifar_x_test)
            else:
                print('Preprocessing CIFAR-10 dataset (sequential, row-by-row)')
                cifar_x = preprocess_dataset_sequential(cifar_x)
                cifar_x_test = preprocess_dataset_sequential(cifar_x_test)

            np.savez_compressed(train_cache_path, x=cifar_x, y=cifar_y)
            np.savez_compressed(test_cache_path, x=cifar_x_test, y=cifar_y_test)

    train_indices, val_indices = network_helper.train_validate_split(cifar_y, val_ratio=0.2, shuffle=shuffle)
    train_dataloader = network_helper.DataLoader(cifar_x, cifar_y, batch_size, train_indices, shuffle=shuffle)
    val_dataloader = network_helper.DataLoader(cifar_x, cifar_y, batch_size, val_indices, shuffle=shuffle)

    test_indices, _ = network_helper.train_validate_split(cifar_y_test, val_ratio=0, shuffle=shuffle)
    test_dataloader = network_helper.DataLoader(cifar_x_test, cifar_y_test, batch_size, test_indices, shuffle=False)

    total_train_batches = network_helper.get_total_batches(batch_size, train_indices)
    total_val_batches = network_helper.get_total_batches(batch_size, val_indices)
    total_test_batches = network_helper.get_total_batches(batch_size, test_indices)

    # In sequential mode every feature is emitted, so max_nonzero == NUM_FEATURES.
    max_nonzero = NUM_FEATURES

    return ((train_dataloader, total_train_batches),
            (val_dataloader, total_val_batches),
            (test_dataloader, total_test_batches),
            max_nonzero)


if __name__ == "__main__":
    batch_size = 64

    for cnn in (False, True):
        label = "CNN (c,r,c,v)" if cnn else "MLP (idx,v)"
        print(f"\n--- CIFAR-10 loader smoke test: {label} ---")
        ((train_gen, n_train),
         (val_gen, n_val),
         (test_gen, n_test),
         max_nonzero) = cifar10_loader_manual(batch_size,
                                              shuffle=False,
                                              preprocess=True,
                                              CNN_preprocess=cnn)

        print(f"Train batches: {n_train}, Val batches: {n_val}, Test batches: {n_test}")
        print(f"Features per sample: {max_nonzero}")

        for batch_x, batch_y in train_gen:
            arr = np.array(batch_x)
            print(f"Batch x shape: {arr.shape}, y shape: {np.array(batch_y).shape}")
            print(f"First 6 entries of sample 0: {arr[0, :6]}")
            break
