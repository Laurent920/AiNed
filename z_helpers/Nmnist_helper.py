import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt

folder = "datasets/Nmnist/"
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# extract_zip(f"{folder}Train.zip", f"{folder}")
# extract_zip(f"{folder}Test.zip", f"{folder}")

def read_nmnist_bin_file(filename):
    with open(filename, 'rb') as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
        
    # The data is stored in 5 bytes per event
    raw_data = raw_data.reshape(-1, 5)
    
    # Pixel coordinates range from 0 to 33
    x = raw_data[:, 0].astype(np.int16)
    y = raw_data[:, 1].astype(np.int16)
    pol = ((raw_data[:, 2] >> 7) & 1).astype(np.int8)
    timestamp = (
        ((raw_data[:, 2] & 0x7F).astype(np.uint32) << 16) |
        (raw_data[:, 3].astype(np.uint32) << 8) |
        raw_data[:, 4].astype(np.uint32)
    )

    return x, y, pol, timestamp

x, y, p, t = read_nmnist_bin_file(f"{folder}/Train/0/00022.bin")
print(x.shape, y.shape, t.shape, p.shape)
print((x[:10]), (y[:10]), t[:], p[:10], max(x), max(y), min(x), min(y))

def plot_events(x, y, pol, title="Event Frame"):
    plt.figure(figsize=(4, 4))
    plt.scatter(x[pol==1], 33 - y[pol==1], color='red', s=0.5, label='ON')
    plt.scatter(x[pol==0], 33 - y[pol==0], color='blue', s=0.5, label='OFF')
    plt.legend()
    plt.title(title)
    plt.xlim(-1, 34)
    plt.ylim(-1, 34)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"{folder}event_plot.png")
    plt.close()

plot_events(x, y, p, title="Event Frame")

def events_to_frame(x, y, pol, frame_size=(34, 34), polarity_mode='both'):
    """
    Convert event data to a 2D image frame.

    Args:
        x (np.ndarray): x-coordinates of events
        y (np.ndarray): y-coordinates of events
        pol (np.ndarray): polarity of events (1 = ON, 0 = OFF)
        frame_size (tuple): image resolution (default: 34×34 for N-MNIST)
        polarity_mode (str): 'on', 'off', or 'both'

    Returns:
        np.ndarray: 2D or 3D frame with accumulated event counts
    """
    h, w = frame_size
    if polarity_mode == 'both':
        frame = np.zeros((2, h, w), dtype=np.uint8)
        np.add.at(frame[0], (y[pol == 1], x[pol == 1]), 1)  # ON events
        np.add.at(frame[1], (y[pol == 0], x[pol == 0]), 1)  # OFF events
    else:
        frame = np.zeros((h, w), dtype=np.uint8)
        mask = pol == (1 if polarity_mode == 'on' else 0)
        np.add.at(frame, (y[mask], x[mask]), 1)
    return frame

frame = events_to_frame(x, y, p, polarity_mode='both')

# Plot ON and OFF channels
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("ON events")
plt.imshow(frame[0], cmap='hot')
plt.subplot(1, 2, 2)
plt.title("OFF events")
plt.imshow(frame[1], cmap='hot')
plt.savefig(f"{folder}event_plot_frame.png")
plt.close()