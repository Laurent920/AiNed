import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import pad

import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms

def torch_nmnist_loader(batch_size, shuffle=False, augmentation=False):
    maximum_time_steps = 314
    
    sensor_size = tonic.datasets.NMNIST.sensor_size

    # Denoise removes isolated, one-off events time_window
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                        transforms.ToFrame(sensor_size=sensor_size,
                                                            time_window=1000)
                                        ])

    trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

    # tonic.utils.plot_event_grid(events)

    transform = tonic.transforms.Compose([torch.from_numpy,
                                        torchvision.transforms.RandomRotation([-10,10])])
    if augmentation:
        cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')
    else:
        cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train') 

    # no augmentations for the testset
    cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')

    # Train - validation - test split
    val_split = 0.2
    train_len = int(len(cached_trainset) * (1 - val_split))
    val_len = len(cached_trainset) - train_len
    train_subset, val_subset = random_split(cached_trainset, [train_len, val_len])

    # Create DataLoaders
    # collate_fn = tonic.collation.PadTensors(batch_first=True)
    collate_fn = lambda batch: custom_pad_collate(batch, maximum_time_steps)

    trainloader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    valloader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    total_train_batches = len(trainloader)
    total_val_batches = len(valloader)
    total_test_batches = len(testloader)

    # maximum_time_steps = get_max_timesteps([valloader])
    return (trainloader, total_train_batches), (valloader, total_val_batches), (testloader, total_test_batches), maximum_time_steps

def get_max_timesteps(loaders_list):
    max = 0
    for loaders in loaders_list:
        data = iter(loaders)
        for i in data:
            event_tensor, target = (i)
            print(target)
            non_zeros = torch.count_nonzero(event_tensor[0][-1])
            # if non_zeros != 0:
            #     print(f"-1: {non_zeros}")
            non_zeros = torch.count_nonzero(event_tensor[0][-10])
            if non_zeros != 0:
                print(f"-2: {non_zeros}")
            else:
                for i in range(8):
                    non_zeros = torch.count_nonzero(event_tensor[0][-9+i])
                    if non_zeros != 0:
                        print(f"others after have non zeros-{-9+i}: {non_zeros}")

            timesteps = event_tensor.shape[1] # A mini-batch has the dimensions (batch size, time steps, channels, height, width)
            if timesteps > max:
                max = timesteps
            print(timesteps)
            # if timesteps == 314:
            #     print(event_tensor.shape, target.shape) 
            # print(f"Total train batches: {total_train_batches}, Total val batches: {total_val_batches}, Total test batches: {total_test_batches}, timesteps: {timesteps}")
    print(f"Maximum timesteps accros train, val and test: {max}")
    return max

def custom_pad_collate(batch, max_timesteps):
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


torch_nmnist_loader(128, shuffle=False, augmentation=False)
