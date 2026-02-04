import torch
from torch.utils.data import DataLoader, random_split
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import tonic
from tonic import DiskCachedDataset
import os

def torch_smnist_loader(batch_size, CNN_preprocess=True, shuffle=False):
    '''
    Sequential MNIST loader using Tonic's SMNIST dataset.
    Each MNIST image is presented as a sequence of events over time.
    
    If CNN_preprocess=True: Returns data in format (B, max_len, 4) where each event is [p, x, y, 1]
    If CNN_preprocess=False: Returns data in format (B, max_len, 2) where each event is [neuron_index, 1]
                             neuron_index = p * H * W + x * W + y
    '''
    
    # Load Tonic's Sequential MNIST dataset
    trainset = tonic.datasets.SMNIST(save_to='./data', train=True, duplicate=False)
    testset = tonic.datasets.SMNIST(save_to='./data', train=False, duplicate=False)
    
    # Get sensor size for SMNIST (99 neurons)
    sensor_size = 99
    print(f"PSMNIST sensor size: {sensor_size}")
    
    # Setup cache directories
    cache_dir = "./cache/SMNIST"
    if CNN_preprocess:
        cache_dir += "/CNN"
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
    
    max_data_length = 5044  # Approximate max events for SMNIST
    
    # Create DataLoaders
    if CNN_preprocess:
        collate_fn = lambda batch: custom_event_pad_collate(batch, max_data_length)
    elif CNN_preprocess is None: # Use the basic collate function to get the max data length
        collate_fn = lambda batch: basic_event_collate(batch)
    else:
        collate_fn = lambda batch: custom_event_flatten_collate(batch, max_data_length, sensor_size)
    

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
    """
    Collate function for raw event data with CNN preprocessing.
    Returns the data in format (B, max_len, 4) where output is [p, x, y, 1]
    """
    data, labels = zip(*batch)
    padded_data = []
    
    for d in data:
        num_events = len(d)
        print(len(d), d, d[10:20])        
        if num_events <= max_len:
            # Pre-allocate the output array
            d_padded_2d = np.full((max_len, 4), -2, dtype=np.int32)
            
            # Fill in the actual data
            d_padded_2d[:num_events, 0] = d['p'].astype(np.int32)  # p values
            d_padded_2d[:num_events, 1] = d['x'].astype(np.int32)  # x values
            d_padded_2d[:num_events, 2] = d['y'].astype(np.int32)  # y values
            d_padded_2d[:num_events, 3] = 1  # ones for actual events
        else:
            print(f"data size exceeds the max len: {num_events} {max_len}")
            raise NotImplementedError
        
        padded_data.append(d_padded_2d)
    
    # Convert to JAX array
    batch_array = jnp.array(padded_data, dtype=jnp.int32)  # shape: (B, max_len, 4)
    label_array = jnp.array(labels, dtype=jnp.int32)
    
    return batch_array, label_array

def custom_event_flatten_collate(batch, max_len):
    """
    Collate function for raw event data when CNN_preprocess=False.
    Converts events from (x, y, t, p) to (neuron_index, 1) format.
    
    neuron_index is computed as: p * H * W + x * W + y
    where H, W are sensor dimensions and p is polarity (0 or 1).
    
    Returns: (B, max_len, 2) array where each event is [neuron_index, 1]
    """
    data, labels = zip(*batch)
    padded_data = []
    
    for d in data:
        num_events = len(d)
        # print(len(d), d, d[10:20])        
        
        if num_events <= max_len:
            # Pre-allocate the output array
            d_padded_2d = np.full((max_len, 2), -2, dtype=np.int32)
            
            p = d['p'].astype(np.int32)
            neuron_indices = d['x'].astype(np.int32)
            # print(p)
            
            # Fill in the actual data
            d_padded_2d[:num_events, 0] = neuron_indices  # neuron index
            d_padded_2d[:num_events, 1] = p  # ones for actual events
        else:
            print(f"data size exceeds the max len: {num_events} {max_len}")
            raise NotImplementedError
        
        padded_data.append(d_padded_2d)
    
    # Convert to JAX array
    batch_array = jnp.array(padded_data, dtype=jnp.int32)  # shape: (B, max_len, 2)
    label_array = jnp.array(labels, dtype=jnp.int32)
    
    return batch_array, label_array

if __name__ == "__main__":
    train, val, test, max_data_length = torch_smnist_loader(
        128, 
        CNN_preprocess=False,
        shuffle=False
    )
    (trainloader, total_train_batches) = train
    (valloader, total_val_batches) = val
    (testloader, total_test_batches) = test
    
    print(f"Total train batches: {total_train_batches}, Total val batches: {total_val_batches}, Total test batches: {total_test_batches}, max data length: {max_data_length}")
    
    # Find actual max length
    max_len = 0
    x_max = 0
    for loader in [trainloader, valloader, testloader]:
        for batch in tqdm(iter(loader)):
            data, labels = batch
            for i, x in enumerate(data):
                x_max = max(np.max(x[:,0]), x_max)
                # print(x)
                # print(x[:, 0])
                new_max = x.shape[0]
                if new_max > max_len:
                    max_len = new_max
                    print(f"New max: {new_max}")
    print(f"Final max: {max_len}")
    print(f"max x index {x_max}")

# CLEAR CACHE: rm -r ./cache/SMNIST