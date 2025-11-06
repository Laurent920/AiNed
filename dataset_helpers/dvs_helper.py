import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import pad
from tqdm import tqdm

import os
import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import numpy as np
import jax.numpy as jnp

def torch_DVSGesture_loader(batch_size, CNN_preproces=False, shuffle=False):
    trainset = tonic.datasets.DVSGesture(save_to='./data', train=True)#, transform=transforms.NumpyAsType(float))
    testset = tonic.datasets.DVSGesture(save_to='./data', train=False)#, transform=transforms.NumpyAsType(float))
    
    data, label = trainset[0]
    # print("Type of data:", type(data))
    # print("Label:", label)
    # print(data.shape, data[0:200], (data[200:]))
    cache_dir = "./cache/DVSGesture"

    if CNN_preproces:
        cache_dir += "/CNN"
        os.makedirs(cache_dir, exist_ok=True)

        cached_trainset = DiskCachedDataset(trainset, cache_path=cache_dir + '/train') 
        cached_testset = DiskCachedDataset(testset, cache_path=cache_dir + '/test')
    else:
        cache_dir += "/MLP"
        os.makedirs(cache_dir, exist_ok=True)

        cached_trainset = DiskCachedDataset(trainset, cache_path=cache_dir + '/train') 
        cached_testset = DiskCachedDataset(testset, cache_path=cache_dir + '/test')
    

    # Train - validation - test split
    val_split = 0.2
    train_len = int(len(cached_trainset) * (1 - val_split))
    val_len = len(cached_trainset) - train_len
    train_subset, val_subset = random_split(cached_trainset, [train_len, val_len])

    max_data_length = 1594557
    # Create DataLoaders
    # collate_fn = lambda batch: basic_event_collate(batch)
    if CNN_preproces:
        collate_fn = lambda batch: custom_event_pad_collate(batch, max_data_length) 
    else:
        collate_fn = lambda batch: custom_preprocess_event_pad_collate(batch, max_data_length) 

    trainloader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
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
    '''
    Returns the data in format (Batch, max_len, 4) where output is [p, x, y, 1]
    with p as 0 or 1 (converted from boolean)
    '''
    data, labels = zip(*batch)
    padded_data = []
    
    for d in data:
        num_events = len(d)
        
        if num_events <= max_len:
            pad_len = max_len - num_events
            
            # Pre-allocate the output array directly
            d_padded_2d = np.full((max_len, 4), -2, dtype=np.int32)
            
            # Fill in the actual data (no need to create intermediate structured array)
            d_padded_2d[:num_events, 0] = d['p'].astype(np.int32)  # p values
            d_padded_2d[:num_events, 1] = d['x'].astype(np.int32)  # x values
            d_padded_2d[:num_events, 2] = d['y'].astype(np.int32)  # y values
            d_padded_2d[:num_events, 3] = 1  # ones for actual events
            # Padding (-2) is already filled by np.full
            
        else:
            print(f"data size exceeds the max len: {num_events} {max_len}")
            raise NotImplementedError
        
        padded_data.append(d_padded_2d)
    
    # Convert directly to JAX array
    batch_array = jnp.array(padded_data, dtype=jnp.int32)  # shape: (B, max_len, 4)
    label_array = jnp.array(labels, dtype=jnp.int32)
    
    return batch_array, label_array

def custom_preprocess_event_pad_collate(batch, max_len):
    '''
    Returns data in shape (B, T, 2), where each event is (neuron_idx, 1).
    neuron_idx = 128*128*p + x*128 + y
    '''
    data, labels = zip(*batch)
    padded_data = []
    
    for d in data:
        num_events = len(d)
        
        # Pre-allocate output array
        d_padded_2d = np.full((max_len, 2), -2, dtype=np.int32)

        # Compute neuron_idx only for actual events
        x = d["x"].astype(np.int32)
        y = d["y"].astype(np.int32)
        p = d["p"].astype(np.int32)
        neuron_idx = x * 128 + y + 128 * 128 * p
        
        # Fill in the actual data
        d_padded_2d[:num_events, 0] = neuron_idx
        d_padded_2d[:num_events, 1] = 1
        # Padding (-2) already filled
        
        padded_data.append(d_padded_2d)
    
    # Convert directly to JAX
    batch_array = jnp.array(padded_data, dtype=jnp.int32)
    label_array = jnp.array(labels, dtype=jnp.int32)
    
    return batch_array, label_array


if __name__ == '__main__':
    batch_size = 128
    (trainloader, total_train_batches), (valloader, total_val_batches), (testloader, total_test_batches), max_nonzero = torch_DVSGesture_loader(batch_size, CNN_preproces=True)
    print(total_train_batches, total_val_batches, total_test_batches)

    max_length = 0
    for loader in [trainloader, valloader, testloader]:
        for batch in tqdm(iter(loader)):
            batch_data, batch_labels = batch
            # print(len(batch_data), (batch_data[0].shape), (batch_data[0][:5]), batch_data.dtype)
        
            for d in batch_data:
                length = d.shape[0]
                # print(d[:100])
                if length > max_length:
                    print('max length:, new_length', max_length, length)
                    max_length = length            
    print('final max length', max_length)
    # CLEAR CACHE: rm -r ./cache/DVSGesture