import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import pad

import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import numpy as np
import jax.numpy as jnp

def torch_SHD_loader(batch_size, shuffle=False):
    trainset = tonic.datasets.SHD(save_to='./data', train=True)#, transform=transforms.NumpyAsType(float))
    testset = tonic.datasets.SHD(save_to='./data', train=False)#, transform=transforms.NumpyAsType(float))
    
    # data, label = trainset[0]

    # print("Type of data:", type(data))
    # print("Label:", label)
    # print(data.shape, data[0:200])
    
    cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/SHD/train') 
    cached_testset = DiskCachedDataset(testset, cache_path='./cache/SHD/test')
    
    # Train - validation - test split
    val_split = 0.2
    train_len = int(len(cached_trainset) * (1 - val_split))
    val_len = len(cached_trainset) - train_len
    train_subset, val_subset = random_split(cached_trainset, [train_len, val_len])

    max_data_length = 14917
    # Create DataLoaders
    collate_fn = lambda batch: basic_event_collate(batch)
    collate_fn = lambda batch: custom_event_pad_collate(batch, max_data_length) 

    trainloader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    valloader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    total_train_batches = len(trainloader)
    total_val_batches = len(valloader)
    total_test_batches = len(testloader)

    # maximum_time_steps = get_max_timesteps([valloader])
    return (trainloader, total_train_batches), (valloader, total_val_batches), (testloader, total_test_batches), max_data_length

def basic_event_collate(batch):
    events, labels = zip(*batch)  # unzip list of tuples
    return list(events), np.array(labels)

def custom_event_pad_collate(batch, max_len):
    data, labels = zip(*batch)  # each d is a np structured array with dtype [('t'), ('index'), ('p=1')]
    # print(type(data[0]), data[0])
    padded_data = []
    for d in data:
        num_events = len(d)
        example_dtype = d.dtype

        if num_events < max_len:
            pad_len = max_len - num_events
            pad = np.zeros(pad_len, dtype=example_dtype)
            for name in example_dtype.names:
                pad[name] = -2  # sentinel padding value, for example -2
            d_padded = np.concatenate([d, pad], axis=0)
        else:
            d_padded = d[:max_len]

        # Convert structured array to (max_len, 4) int64 numpy array
        d_padded_2d = np.stack([d_padded[name] for name in example_dtype.names], axis=1).astype(np.int64)

        padded_data.append(torch.from_numpy(d_padded_2d))

    # batch_tensor = torch.stack(padded_data)  # shape: (batch_size, max_len, 4)
    # labels_tensor = torch.tensor(labels)
    batch_array = jnp.stack([jnp.array(x) for x in padded_data])  # shape: (B, T, 3)
    # print(batch_array[0])
    label_array = jnp.array(labels, dtype=jnp.int32)
    # print("type and shape", type(batch_array), batch_array.shape)

    return batch_array[:,:,-2:], label_array

if __name__ == '__main__':
    batch_size = 128
    (trainloader, total_train_batches), (valloader, total_val_batches), (testloader, total_test_batches), max_nonzero = torch_SHD_loader(batch_size)
    batch_iterator = iter(trainloader)
    max_length = 0
    for batch in batch_iterator:
        data, labels = batch
        print(data.shape, (data[0][0]), (labels))
    
        for d in data:
            # last_el = d[:, -2:]
            # print(last_el)
            length = d.shape[0]
            if length > max_length:
                print('max length:', max_length)
                max_length = length
    # print(max_length)
    
    # CLEAR CACHE: rm -r ./cache/SHD