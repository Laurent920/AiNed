import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import pad

import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import numpy as np

def torch_nmnist_loader(batch_size, shuffle=False, augmentation=False, binned=False, aggregate_time=True):
    '''
    If not binned it returns the raw dataset in forms of tuples of 4 which correspond to (x, y, time, polarity)=>(polarity, x, y, 1),
    x and y are comprised in [0,34] and polarity is 1 for positive spike and 0 for negative spike
    
    If binned and aggregate_time=True, returns (B, C, H, W) by summing across time.
    If binned and aggregate_time=False, returns (B, T, C, H, W) with time preserved.
    '''
    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_transform = transforms.Denoise(filter_time=10000) # Raw data
    if binned:
        # Denoise removes isolated, one-off events time_window
        frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                            transforms.ToFrame(sensor_size=sensor_size,
                                                                time_window=1000)
                                            ])
    trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

    # for x, y in testset:
    #     print(f"data shape: {np.array(x).shape}, data[0]: {x[0]}, label: {y}")
    # return

    # tonic.utils.plot_event_grid(events)

    transform = tonic.transforms.Compose([torch.from_numpy,
                                        torchvision.transforms.RandomRotation([-10,10])])
    
    if binned:
        train_cache_path = './cache/nmnist/binned/train'
        test_cache_path = './cache/nmnist/binned/test'
    else:
        train_cache_path = './cache/nmnist/raw/train'
        test_cache_path = './cache/nmnist/raw/test'
        
    if augmentation:
        cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path=train_cache_path)
    else:
        cached_trainset = DiskCachedDataset(trainset, cache_path=train_cache_path) 

    # no augmentations for the testset
    cached_testset = DiskCachedDataset(testset, cache_path=test_cache_path)

    # Train - validation - test split
    val_split = 0.2
    train_len = int(len(cached_trainset) * (1 - val_split))
    val_len = len(cached_trainset) - train_len
    train_subset, val_subset = random_split(cached_trainset, [train_len, val_len])

    max_data_length = 7793 # For raw data
    maximum_time_steps = 314 # For binned data

    # Create DataLoaders
    # collate_fn = lambda batch: basic_event_collate(batch)
    if binned: # For binned dataloader
        if aggregate_time:
            collate_fn = lambda batch: custom_pad_collate_aggregated(batch)
        else:
            collate_fn = lambda batch: custom_pad_collate(batch, maximum_time_steps)
    else:
        collate_fn = lambda batch: custom_event_pad_collate(batch, max_data_length) # For raw dataloader

    trainloader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    valloader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    total_train_batches = len(trainloader)
    total_val_batches = len(valloader)
    total_test_batches = len(testloader)

    # maximum_time_steps = get_max_timesteps([valloader])
    max_nonzero = maximum_time_steps if binned else max_data_length
    return (trainloader, total_train_batches), (valloader, total_val_batches), (testloader, total_test_batches), max_nonzero

def get_max_timesteps(loaders_list):
    max = 0
    for loaders in loaders_list:
        data = iter(loaders)
        for i in data:
            event_tensor, target = (i)
            print(event_tensor.shape)
            non_zeros = torch.count_nonzero(event_tensor[0][-1])
            # if non_zeros != 0:
            #     print(f"-1: {non_zeros}")
            non_zeros = torch.count_nonzero(event_tensor[0][-10])
            # if non_zeros != 0:
            #     print(f"-2: {non_zeros}")
            # else:
            #     for i in range(8):
            #         non_zeros = torch.count_nonzero(event_tensor[0][-9+i])
            #         if non_zeros != 0:
            #             print(f"others after have non zeros-{-9+i}: {non_zeros}")

            timesteps = event_tensor.shape[1] # A mini-batch has the dimensions (batch size, time steps, channels, height, width)
            if timesteps > max:
                max = timesteps
            tensor = event_tensor[100]
            print(timesteps, torch.max(tensor), torch.min(tensor), (tensor == 2).sum(), (tensor == 1).sum(), (tensor == 0).sum(), (tensor == -1).sum())
            # if timesteps == 314:
            #     print(event_tensor.shape, target.shape) 
            # print(f"Total train batches: {total_train_batches}, Total val batches: {total_val_batches}, Total test batches: {total_test_batches}, timesteps: {timesteps}")
    print(f"Maximum timesteps accros train, val and test: {max}")
    return max

def custom_pad_collate_aggregated(batch):
    '''
    Collate function for binned data with time aggregation.
    Returns: (B, C, H, W) by summing all time steps.
    '''
    batch_data, batch_targets = zip(*batch)

    aggregated_data = []
    for data in batch_data:
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        
        # Sum across time dimension: (T, C, H, W) -> (C, H, W)
        aggregated_frame = torch.sum(data, dim=0)
        aggregated_data.append(aggregated_frame)

    batch_tensor = torch.stack(aggregated_data)  # (B, C, H, W)
    batch_targets = torch.tensor(batch_targets)

    return batch_tensor, batch_targets

def custom_pad_collate(batch, max_timesteps):
    '''
    Collate function for binned data
    '''
    batch_data, batch_targets = zip(*batch)  # list of (T, C, H, W) samples

    padded_data = []
    for data in batch_data:
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)  # Fix for numpy input
        T, C, H, W = data.shape
        pad_T = max_timesteps - T
        padded = pad(data, (0, 0, 0, 0, 0, 0, 0, pad_T))  # pad T dimension
        padded_data.append(padded)

    batch_tensor = torch.stack(padded_data)  # (B, T, C, H, W)
    batch_targets = torch.tensor(batch_targets)

    return batch_tensor, batch_targets

def basic_event_collate(batch):
    events, labels = zip(*batch)  # unzip list of tuples
    print("events", events, events.shape)
    return list(events), np.array(labels)

def custom_event_pad_collate(batch, max_len):
    """
    Collate function for raw event data.
    Reshapes events from (x, y, t, p) to (p, x, y, 1) format.
    """
    data, labels = zip(*batch)  # each d is a np structured array with dtype [('x', '<i8'), ('y', '<i8'), ('t', '<i8'), ('p', '<i8')]
    
    # print("data shape:", len(data), "first sample events:", len(data[0]))
    
    padded_data = []
    for d in data:
        num_events = len(d)
        example_dtype = d.dtype

        # Pad or truncate to max_len
        if num_events < max_len:
            pad_len = max_len - num_events
            pad = np.zeros(pad_len, dtype=example_dtype)
            for name in example_dtype.names:
                pad[name] = -2  # sentinel padding value
            d_padded = np.concatenate([d, pad], axis=0)
        else:
            d_padded = d[:max_len]

        # Extract fields: (max_len,) arrays
        p = d_padded['p']  # polarity
        x = d_padded['x']  # x coordinate
        y = d_padded['y']  # y coordinate
        # t = d_padded['t']  # timestamp (not used in new format)
        
        # Reshape to (max_len, 4) where columns are [p, x, y, 1]
        ones = np.concatenate([np.ones(num_events, dtype=np.int64), np.full(max_len-num_events, -2, dtype=np.int64)])
        d_reshaped = np.stack([p, x, y, ones], axis=1).astype(np.int64)
        
        padded_data.append(torch.from_numpy(d_reshaped))

    batch_tensor = torch.stack(padded_data)  # shape: (batch_size, max_len, 4)
    labels_tensor = torch.tensor(labels)
    
    # print(f"Batch tensor shape: {batch_tensor.shape}")
    # print(f"First event in batch: {batch_tensor[0, 0]}")  # Should be [p, x, y, 1]

    return batch_tensor, labels_tensor

if __name__ == "__main__":
    (trainloader, total_train_batches), (valloader, total_val_batches), (testloader, total_test_batches), maximum_time_steps = torch_nmnist_loader(128, shuffle=False, augmentation=False)
    print(f"Total train batches: {total_train_batches}, Total val batches: {total_val_batches}, Total test batches: {total_test_batches}, maximum time steps: {maximum_time_steps}")
    
    max = 0
    for loader in [testloader]:
        for data, y in loader:
            for i, x in enumerate(data):
                new_max = x.shape[0] 
                if new_max > max:
                    max = new_max
                    # print(x.shape)
                    print(new_max)
                # if i == 127:
                #     print(i)
            # print(f"data shape: {(x.shape)}, \ndata[0]: {(x)}, label: {(y.shape)}")
    print("final max", max)

# CLEAR CACHE: rm -r ./cache/nmnist
