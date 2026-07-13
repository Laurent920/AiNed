import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import pad

import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import os

N_CHANNELS = 700  # SHD cochlea channels (tonotopic, frequency-ordered)

def torch_SHD_loader(batch_size, shuffle=False, downsample=False, CNN_preprocess=False, data_dir="", augment=False):
    """
    Load SHD (Spiking Heidelberg Digits) dataset.

    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        downsample: Downsampling option (currently unused for SHD)
        data_dir: Root directory for data storage. If empty, uses current directory.

    Returns:
        Tuple of (train_data, val_data, test_data, max_nonzero)
    """
    if data_dir:
        save_dir = os.path.join(data_dir, 'data')
        base_cache_dir = os.path.join(data_dir, "cache/SHD")
    else:
        save_dir = './data'
        base_cache_dir = "./cache/SHD"

    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading SHD dataset from: {save_dir}")

    trainset = tonic.datasets.SHD(save_to=save_dir, train=True)
    testset = tonic.datasets.SHD(save_to=save_dir, train=False)

    cached_trainset = DiskCachedDataset(trainset, cache_path=os.path.join(base_cache_dir, 'train'))
    cached_testset = DiskCachedDataset(testset, cache_path=os.path.join(base_cache_dir, 'test'))

    val_split = 0
    train_len = int(len(cached_trainset) * (1 - val_split))
    val_len = len(cached_trainset) - train_len
    train_subset, val_subset = random_split(cached_trainset, [train_len, val_len])

    max_data_length = 16257
    collate_fn = lambda batch: custom_event_pad_collate(batch, max_data_length)
    if augment:
        train_collate_fn = lambda batch: augmenting_event_pad_collate(batch, max_data_length)
    else:
        train_collate_fn = collate_fn

    trainloader = DataLoader(train_subset, batch_size=batch_size, collate_fn=train_collate_fn, shuffle=shuffle)
    valloader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    total_train_batches = len(trainloader)
    total_val_batches = len(valloader)
    total_test_batches = len(testloader)

    return (trainloader, total_train_batches), (valloader, total_val_batches), (testloader, total_test_batches), max_data_length

def basic_event_collate(batch):
    events, labels = zip(*batch)  # unzip list of tuples
    return list(events), np.array(labels)

def custom_event_pad_collate(batch, max_len):
    data, labels = zip(*batch)  # each d is a np structured array with dtype [('t'), ('x'), ('p=1')]
    padded_data = []

    for d in data:
        num_events = len(d)
        if num_events <= max_len:
            pad_len = max_len - num_events

            # Pre-allocate the output array directly
            d_padded_2d = np.full((max_len, 2), -2, dtype=np.int32)
            
            # Fill in the actual data (no need to create intermediate structured array)
            d_padded_2d[:num_events, 0] = d['x'].astype(np.int32)  # p values
            d_padded_2d[:num_events, 1] = 1  # ones for actual events
            # Padding (-2) is already filled by np.full
        else:
            print(f"data size exceeds the max len: {num_events} {max_len}")
            raise NotImplementedError

        padded_data.append(d_padded_2d)

    # Convert directly to JAX array
    batch_array = jnp.array(padded_data, dtype=jnp.int32)  # shape: (B, max_len, 4)
    label_array = jnp.array(labels, dtype=jnp.int32)

    return batch_array, label_array

def _augment_shd_events(events, channel_shift=16, channel_jitter=3.0, drop_p=0.1):
    """Train-only SHD augmentation, applied per sample on the raw event array.

    The collate discards timestamps, so only the channel index ('x'), event order
    and count reach the network — augmentation therefore acts in channel space:
      - channel_shift:  rigid per-sample shift of all channels (speaker/pitch shift)
      - channel_jitter: per-event Gaussian channel jitter (frequency blur)
      - drop_p:         randomly delete a fraction of events (regularization)
    Events shifted/jittered out of [0, N_CHANNELS) are dropped.
    """
    if events.size == 0:
        return events
    out = events.copy()
    x = out['x'].astype(np.int32)
    if channel_shift:
        x += np.random.randint(-channel_shift, channel_shift + 1)
    if channel_jitter:
        x = x + np.round(np.random.normal(0.0, channel_jitter, size=x.shape)).astype(np.int32)
    keep = (x >= 0) & (x < N_CHANNELS)
    if drop_p:
        keep &= np.random.rand(x.shape[0]) >= drop_p
    out = out[keep]
    out['x'] = x[keep]
    return out

def augmenting_event_pad_collate(batch, max_len):
    """Train-only collate: augment each sample in channel space, then pad like
    custom_event_pad_collate. Augmentation only drops events, so num_events never
    exceeds max_len."""
    data, labels = zip(*batch)
    padded_data = []

    for d in data:
        d = _augment_shd_events(d)
        num_events = len(d)
        d_padded_2d = np.full((max_len, 2), -2, dtype=np.int32)
        d_padded_2d[:num_events, 0] = d['x'].astype(np.int32)
        d_padded_2d[:num_events, 1] = 1
        padded_data.append(d_padded_2d)

    batch_array = jnp.array(padded_data, dtype=jnp.int32)
    label_array = jnp.array(labels, dtype=jnp.int32)

    return batch_array, label_array

if __name__ == '__main__':
    batch_size = 128
    (trainloader, total_train_batches), (valloader, total_val_batches), (testloader, total_test_batches), max_nonzero = torch_SHD_loader(batch_size)
    batch_iterator = iter(trainloader)
    max_length = 0
    for loader in [trainloader, valloader, testloader]:
        for batch in tqdm(iter(loader)):
            data, labels = batch
            # print(data.shape, (data[0][0]), (labels))
        
            for d in data:
                length = d.shape[0]
                if length > max_length:
                    print('max length:', max_length)
                    max_length = length
    print(max_length)    
    # CLEAR CACHE: rm -r ./cache/SHD