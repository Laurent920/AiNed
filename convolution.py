from async_CNN import Params, Neuron_states, Conv_Neuron, init_params

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import tree_math
import dataclasses
import sys
import numpy as np

from data_helpers.nmnist_helper import torch_nmnist_loader

@tree_math.struct
@dataclasses.dataclass(frozen=True)
class Network:
    '''
    A class representing the neural network architecture.
    
    It initializes the layers depending on the layer_sizes input and stores either a convolution layer or a fully connected layer.
    
    It then initializes the weights/filters for each layer.
    '''
    def __init__(self, key, params:Params, layer_sizes: tuple[tuple[int, ...]], debug=False):
        '''
        Parameters:
            key: random key 
            params: Params object containing the parameters for the network
            layer_sizes (tuple of tuples): 
                        - for a fully connected layer each tuple contains a single integer representing the number of neurons in that layer
                        - for a convolutional layer each tuple contains (out_chan, kernel, padding, stride) representing the output channel, kernel size, padding and stride.
        
        '''
        self.params = params
        self.layers = []        
        previous_layer = 0  # Initialize previous layer as zero
        for i, layer in enumerate(layer_sizes):
            if len(layer) == 1:
                if i == 0:
                    previous_layer = 1 # Use the smallest array because residuals are not needed for input layer
                else:
                    if isinstance(previous_layer, int):
                        pass
                    if isinstance(previous_layer, jax.Array):
                        flat_previous_layer =  previous_layer.flatten().size
                        # if rank == 0 and debug:
                        #     print(f"rank {rank}, Previous layer: {previous_layer.shape}, flattened: {flat_previous_layer}")
                        previous_layer = flat_previous_layer
                        
                key, subkey = jax.random.split(key) 
                thresholds = jax.random.normal(subkey, layer) * params.init_thresholds

                empty_neuron_states = Neuron_states(
                                        values=jnp.zeros(layer), 
                                        thresholds=thresholds, 
                                        input_residuals=jnp.zeros((previous_layer,)),
                                        weight_residuals={"input order": jnp.full((previous_layer,), -1, dtype=int), 
                                                        "input activity": jnp.full((previous_layer,), 0, dtype=int), 
                                                        "layer activity": jnp.zeros((layer[0],), dtype=int), 
                                                        "output activity": jnp.zeros((previous_layer, layer[0]))},
                                        last_sent_iteration=0,
                                        weights_shape=(previous_layer, layer[0])
                                        )
                self.layers.append(empty_neuron_states)
                previous_layer = layer[0]
            else:
                if i == 0:
                    previous_layer = jnp.zeros(1)
                    values = jnp.zeros(layer)
                    out_chan, kernel, padding, stride = 1, (0,0), (0,0), (0,0) # Values used as placeholders for the input layer
                else:
                    out_chan, kernel, padding, stride = layer
                    in_shape = previous_layer.shape
                    h_out = (in_shape[1] + 2 * padding[0] - kernel[0]) // stride[0] + 1
                    w_out = (in_shape[2] + 2 * padding[1] - kernel[1]) // stride[1] + 1
                    # if rank == 0 and debug:
                    #     print(f"rank {rank}, previous layer shape: {previous_layer.shape}, out shape: {(out_chan, h_out, w_out)}, kernel: {kernel}, padding: {padding}, stride: {stride}")
                    values = jnp.zeros((out_chan, h_out, w_out))  # Initialize values for convolutional layer
                
                in_chan = previous_layer.shape[0]
                key, subkey = jax.random.split(key) 
                thresholds = jax.random.normal(subkey, values.shape) * params.init_thresholds
                empty_conv_neuron = Conv_Neuron(
                                    neuron_state=Neuron_states(
                                                    values=values, 
                                                    thresholds=thresholds, 
                                                    input_residuals=jnp.zeros(previous_layer.shape),
                                                    weight_residuals={"input order": jnp.full(previous_layer.shape, -1, dtype=int), 
                                                                    "input activity": jnp.full(previous_layer.shape, 0, dtype=int), 
                                                                    "layer activity": jnp.zeros((layer[0],), dtype=int), 
                                                                    "output activity": previous_layer},
                                                    last_sent_iteration=0,
                                                    weights_shape=(out_chan, in_chan, kernel[0], kernel[1])),  
                                    kernel=kernel,
                                    padding=padding,
                                    stride=stride,
                                    previous_layer=previous_layer
                                    )   
                
                self.layers.append(empty_conv_neuron)
                previous_layer = values
        self.key = key
        self.layers = tuple(self.layers) # Convert to tuple to allow jit

    def init_weights(self, scale=1e-2):
        '''
        Initialize the weights for each layer based on the layer sizes.
        
        Returns the weights correponding to the MPI split_rank.
        ''' 
        # weights = init_params(self.key, self.layers, self.params.load_file)
        # print(f"Rank {split_rank} initialized weights: {weights.shape}")
        weights = []
        keys = jax.random.split(key, len(self.layers))
        for i, layer in enumerate(self.layers):
            if type(layer) is Neuron_states:
                weights_shape = layer.weights_shape
            else:
                weights_shape = layer.neuron_state.weights_shape
            l_weights = scale * jax.random.normal(keys[i], weights_shape)
            weights.append(l_weights)    
            print(i, l_weights.shape)
        return weights

    
def im2col(image, ksize, stride=1):
    """
    Rearranges the input's image into columns based on the receptive field define by ksize and stride

    :param image: A numpy's ndarray of shape [batch, in_channel, height, width] or [in_channel, height, width]
    :param ksize: A python's int32 list containing the shape of the kernel: [out_channel, in_channel, k_height, k_width]
    :param stride: (optional, default 1) The stride, this value will be used for both vertical and horizontal  stride
    :return: - img_cols: A numpy's ndarray with shape [img_channel * k_height * k_width, batch * out_h * out_w], where
                out_h and out_w depend of the provided ksize and stride
             - (out_h, out_w): The shape of the output feature map.
    """

    batch, img_channel, img_height, img_width = image.shape if image.ndim == 4 else [1, *image.shape]
    out_channel, _, k_height, k_width = ksize

    out_h = (img_height - k_height) // stride + 1
    out_w = (img_width - k_width) // stride + 1

    if image.ndim == 4:
        shape = (img_channel, k_height, k_width, batch, out_h, out_w)
        strides = (img_height * img_width, img_width, 1, img_channel * img_height * img_width, stride * img_width, stride)
    else:
        shape = (img_channel, k_height, k_width, out_h, out_w)
        strides = (img_height * img_width, img_width, 1, stride * img_width, stride)

    strides = image.itemsize * np.array(strides)
    img_stride = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    img_cols = np.ascontiguousarray(img_stride)
    img_cols.shape = (img_channel * k_height * k_width, batch * out_h * out_w)

    return img_cols, [out_h, out_w]

def conv2d(image, kernel, bias=None, padding='VALID', stride=1):
    """
    Computes the convolution of the kernel on the provided image.

    :param image: The image as a [batch, in_channel, height, width] (or [in_channel, height, width]) numpy ndarray
    :param kernel: The kernel to be convolved as a [out_channel, in_channel, k_height, k_width] ndarray
    :param bias: (optional) If provided, the bias will be added after the convolution operation.
    :param padding: (optional) A string representing the padding to be applied. 'VALID' is the only one currently
        supported.
    :param stride: (optional, default 1) The stride, this value will be used for both vertical and horizontal  stride.
    :return: The convolved image as a numpy's ndarray of shape [batch, out_channel, out_h, out_w]
    """

    image = np.ascontiguousarray(image)

    batch, in_channel, in_h, in_w = image.shape if image.ndim == 4 else [1, *image.shape]
    out_channel, _, k_h, k_w = kernel.shape

    if padding == 'SAME':
        out_h = int(np.ceil(in_h / stride))
        out_w = int(np.ceil(in_w / stride))

        pad_h = max((out_h - 1) * stride + k_h - in_h, 0)
        pad_w = max((out_w - 1) * stride + k_w - in_w, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if image.ndim == 4:
            image = np.pad(
                image,
                ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant')
        else:
            image = np.pad(
                image,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant')
            
    img_cols, (out_h, out_w) = im2col(image, kernel.shape, stride)
    kernel_rows = kernel.reshape(out_channel, -1)

    conv = kernel_rows.dot(img_cols)
    if bias is not None:
        bias_rows = bias.reshape(out_channel, 1)
        conv = conv + bias_rows

    if image.ndim == 4:
        conv = conv.reshape(out_channel, batch, out_h, out_w).transpose(1, 0, 2, 3)
    else:
        conv = conv.reshape(out_channel, out_h, out_w)

    return conv


def int_max(a, b):
    return a if a >= b else b

def int_min(a, b):
    return a if a <= b else b

def im2col_event(image, event_y, event_x, k_height, k_width, stride, chan_as_cols=0):
    num_events = event_y.shape[0]
    in_channel, in_height, in_width = image.shape
    out_height = (in_height - k_height) // stride + 1
    out_width = (in_width - k_width) // stride + 1

    out_covered = np.zeros((out_height, out_width), dtype=np.int32)
    max_cols = out_height * out_width

    if chan_as_cols:
        out_cols = np.empty((k_height * k_width, in_channel * max_cols), dtype=np.float32, order='F')
    else:
        out_cols = np.empty((in_channel * k_height * k_width, max_cols), dtype=np.float32, order='F')

    out_cols_y = np.empty(max_cols, dtype=np.int32)
    out_cols_x = np.empty(max_cols, dtype=np.int32)
    next_out_idx = 0

    for i in range(num_events):
        y = event_y[i]
        x = event_x[i]

        if stride == 1:
            y_min_rf = int_max(0, y - (k_height - 1))
            y_max_rf = int_min(in_height, y + (k_height - 1) + 1)
            x_min_rf = int_max(0, x - (k_width - 1))
            x_max_rf = int_min(in_width, x + (k_width - 1) + 1)
        elif stride == k_width and stride == k_height:
            y_min_rf = (y // stride) * k_height
            y_max_rf = y_min_rf + k_height
            x_min_rf = (x // stride) * k_width
            x_max_rf = x_min_rf + k_width
        else:
            raise NotImplementedError("Only stride=1 or stride equal to kernel dimensions is supported.")

        num_height = (y_max_rf - y_min_rf - k_height) // stride + 1
        num_width = (x_max_rf - x_min_rf - k_width) // stride + 1

        for offset_y in range(0, num_height * stride, stride):
            for offset_x in range(0, num_width * stride, stride):
                top_y = y_min_rf + offset_y
                left_x = x_min_rf + offset_x
                out_y_rf = top_y // stride
                out_x_rf = left_x // stride

                if out_covered[out_y_rf, out_x_rf] == 0:
                    out_covered[out_y_rf, out_x_rf] = 1
                    out_cols_y[next_out_idx] = out_y_rf
                    out_cols_x[next_out_idx] = out_x_rf

                    for rf_chn in range(in_channel):
                        for rf_offset_y in range(k_height):
                            rf_y = top_y + rf_offset_y
                            for rf_offset_x in range(k_width):
                                rf_x = left_x + rf_offset_x
                                if chan_as_cols:
                                    row_idx = rf_offset_y * k_width + rf_offset_x
                                    col_idx = in_channel * next_out_idx + rf_chn
                                else:
                                    row_idx = rf_chn * (k_height * k_width) + rf_offset_y * k_width + rf_offset_x
                                    col_idx = next_out_idx
                                out_cols[row_idx, col_idx] = image[rf_chn, rf_y, rf_x]

                    next_out_idx += 1

    out_len = next_out_idx * in_channel if chan_as_cols else next_out_idx
    return out_cols[:, :out_len], (out_cols_y[:next_out_idx], out_cols_x[:next_out_idx])


def conv2d_event(image, events, kernel, bias=None, padding='VALID', stride=1):
    """
    Computes the convolution of the receptive fields around the provided events.

    :param image: The image as a [batch, in_channel, height, width] numpy ndarray
    :param events: The events around which the convolution must be applied.
    :param kernel: The kernel to be convolved as a [out_channel, in_channel, k_height, k_width] ndarray
    :param bias: (optional) If provided, the bias will be added after the convolution operation.
    :param padding: (optional) A string representing the padding to be applied. 'VALID' is the only one currently
        supported.
    :param stride: (optional, default 1) The stride, this value will be used for both vertical and horizontal  stride.
    :return: - out_conv: The result of the convolution around the events as a flat array of shape [out_channels, num_rf]
                Each column is the result of the convolution
                over one of the receptive fields affected by one of the events. Its location is specified by the other
                return value of this method.
             - (out_y, out_x): The coordinates where each value in the 'conv' output has to be placed. You can place
                the obtained values in a featuremap with: featuremap[:, out_y, out_x] = out_conv
    """

    image = np.ascontiguousarray(image)

    out_channel, _, k_height, k_width = kernel.shape
    y_events, x_events = events
    # Makes sure that all the types are correct
    y_events = y_events.astype(np.int32)
    x_events = x_events.astype(np.int32)
    image = image.astype(np.float32)

    img_cols, out_events = im2col_event(image, y_events, x_events, k_height, k_width, stride)
    kernel_rows = kernel.reshape(out_channel, -1)

    conv = kernel_rows.dot(img_cols)
    if bias is not None:
        bias_rows = bias.reshape(out_channel, 1)
        conv = conv + bias_rows
    conv = conv.reshape(out_channel, -1)

    return conv, out_events


from data_helpers.nmnist_helper import torch_nmnist_loader
batch_size = 36
shuffle = False

(training_generator, total_train_batches), (validation_generator, total_val_batches), (test_generator, total_test_batches), max_nonzero = torch_nmnist_loader(batch_size, shuffle=shuffle)

import torch
import torch.nn.functional as F

#__________________________________________________________________________________Test conv2d_____________________________________________________________________________
# # Create dummy data
# batch_size = 2
# in_channels = 4
# height = width = 8
# out_channels = 6
# kernel_size = (5, 5)
# stride = 1

# # Generate random input, weights, and biases
# x = np.random.randn(batch_size, in_channels, height, width).astype(np.float32)
# w = np.random.randn(out_channels, in_channels, *kernel_size).astype(np.float32)

# # NumPy conv
# out_numpy = conv2d(x, w, None, padding='SAME', stride=1)

# # PyTorch conv
# x_torch = torch.tensor(x)
# w_torch = torch.tensor(w)
# out_torch = F.conv2d(x_torch, w_torch, None, padding='same')

# # Compare
# print(f"numpy output shape: {out_numpy.shape}, pytorch output shape: {out_torch.shape}")
# print(np.allclose(out_numpy, out_torch.numpy(), atol=1e-5))
#__________________________________________________________________________________Test conv2d_event_____________________________________________________________________________
# import numpy as np

# # --------------------
# # Dummy input data
# # --------------------
# image = np.arange(3 * 5 * 5).reshape(1, 3, 5, 5).astype(np.float32)
# # [batch=1, channels=3, height=5, width=5]

# kernel = np.ones((2, 3, 3, 3), dtype=np.float32)  # [out_channels=2, in_channels=3, 3x3 kernel]
# bias = np.array([1.0, -1.0], dtype=np.float32)

# # Events at center-ish positions
# event_y = np.array([1, 2, 3])
# event_x = np.array([1, 2, 3])

# # --------------------
# # Full convolution
# # --------------------
# def conv2d(image, kernel, bias=None, padding='VALID', stride=1):
#     batch, in_ch, in_h, in_w = image.shape
#     out_ch, _, k_h, k_w = kernel.shape
#     out_h = (in_h - k_h) // stride + 1
#     out_w = (in_w - k_w) // stride + 1

#     output = np.zeros((batch, out_ch, out_h, out_w), dtype=np.float32)
#     for b in range(batch):
#         for oc in range(out_ch):
#             for ic in range(in_ch):
#                 for i in range(out_h):
#                     for j in range(out_w):
#                         y = i * stride
#                         x = j * stride
#                         region = image[b, ic, y:y+k_h, x:x+k_w]
#                         output[b, oc, i, j] += np.sum(region * kernel[oc, ic])
#             if bias is not None:
#                 output[b, oc] += bias[oc]
#     return output

# # --------------------
# # Your event-based convolution
# # --------------------
# # Use your previous implementation of conv2d_event (assumed available in scope)
# # If image is [1, C, H, W], convert to [C, H, W]
# event_output, (out_y, out_x) = conv2d_event(image[0], (event_y, event_x), kernel, bias)

# # --------------------
# # Compare
# # --------------------
# # Compute full output as reference
# full_output = conv2d(image, kernel, bias)

# # Compare values at event positions
# for i in range(len(out_y)):
#     y = out_y[i]
#     x = out_x[i]
#     full_val = full_output[0, :, y, x]
#     event_val = event_output[:, i]
#     print(f"At (y={y}, x={x}):")
#     print(f"  Full conv:   {full_val}")
#     print(f"  Event conv:  {event_val}")
#     print(f"  Match:       {np.allclose(full_val, event_val)}")
#     print("-----")


#__________________________________________________________________________________Main_____________________________________________________________________________

def numpy_forward_pass(params, weights, empty_neuron_states, x):
    """
    Perform a forward pass using custom numpy conv2d and matmul for dense layers.
    """
    x = x # Handle PyTorch tensors ___ Shape: (T, C, H, W)

    neuron_states = empty_neuron_states
    for timestep in range(params.max_nonzero):
        # print(timestep)
        current_frame = x[timestep, :]
        for idx, layer in enumerate(layer_sizes):
            if idx == 0:
                continue
            w = weights[idx]

            if len(layer) == 4:  # Conv layer
                out_ch, kernel_size, padding, stride = layer

                # Extract kernel and bias
                current_frame = conv2d(current_frame, w, bias=None, padding='SAME', stride=stride[0])
                # print(f"after convolution layer {idx} shape: {current_frame.shape}")
            elif len(layer) == 1:  # Dense layer
                
                current_frame = current_frame.reshape(1, np.prod(current_frame.shape[0:]))  # Flatten
                
                current_frame = np.dot(current_frame, w)
                # print(f"after fc layer {idx} shape: {current_frame.shape}")

            neuron_states[idx].values += current_frame
            current_frame = np.maximum(current_frame, 0)  # ReLU or spike function
            neuron_states[idx].values -= current_frame
            
            current_frame = neuron_states[idx].values
    
    return neuron_states

    
def numpy_forward_pass_event(params, weights, empty_neuron_states, x):
    """
    Perform a forward pass using custom conv2d_event for sparse event-based data.
    """
    neuron_states = empty_neuron_states

    for timestep in range(params.max_nonzero):
        current_frame = x[timestep, :]  # Shape: (C, H, W)
        
        # Extract event coordinates: assume channel 0 has event data
        event_mask = current_frame != 0
        valid_events = np.nonzero(event_mask)
        event_y = valid_events[:, 1].numpy()
        event_x = valid_events[:, 2].numpy()
        # print("Valid events:", len(valid_events), event_x, event_y)

        for idx, layer in enumerate(layer_sizes):
            if idx == 0:
                continue
            w = weights[idx]

            if len(layer) == 4:  # Conv layer
                out_ch, kernel_size, padding, stride = layer
                kernel = w  # assuming shape: (out_ch, in_ch, k_h, k_w)
                conv_out, (out_y, out_x) = conv2d_event(
                    current_frame,
                    (event_y, event_x),
                    kernel,
                    bias=None,
                    padding='VALID',  # 'SAME' can be added if supported later
                    stride=stride[0]
                )

                # Prepare next input as zero image and place conv result
                out_h = int(np.ceil(current_frame.shape[1] / stride[0]))
                out_w = int(np.ceil(current_frame.shape[2] / stride[0]))
                current_frame = np.zeros((out_ch, out_h, out_w), dtype=np.float32)
                current_frame[:, out_y, out_x] = conv_out

            elif len(layer) == 1:  # Dense layer
                current_frame = current_frame.reshape(1, -1)
                current_frame = np.dot(current_frame, w)
            # print(f'layer {idx}: current frame shape: {current_frame.shape}')
            neuron_states[idx].values += current_frame
            current_frame = np.maximum(current_frame, 0)
            neuron_states[idx].values -= current_frame
            current_frame = neuron_states[idx].values

    return neuron_states


def convolution(params, network, weights, training_generator, total_train_batches):
    empty_neuron_states = [] 
    for i in range(len(network.layers)):
        empty_neuron_states.append(network.layers[i])

    batch_iterator = iter(training_generator)
            
    for i in range(total_train_batches):
        all_batch_x, all_batch_y = next(batch_iterator)
        print(f"batch x shape: {all_batch_x.shape}, batch y shape: {all_batch_y.shape}")
        all_predicted = []
        for x in all_batch_x:
            neuron_states = numpy_forward_pass_event(params, weights, empty_neuron_states, x)
            outputs = neuron_states[-1].values
            all_predicted.append(np.argmax(outputs, axis=1))
            
        pred_np = np.array([int(p[0]) for p in all_predicted])  # assuming p is Array([digit])
        true_np = all_batch_y.numpy() if isinstance(all_batch_y, torch.Tensor) else np.array(all_batch_y)
        correct = (pred_np == true_np).sum()

        print(f"Batch {i} output: {pred_np}, {outputs} correct: {correct}/{params.batch_size}")

    return

if __name__ == "__main__":
    random_seed = 42
    key = jax.random.key(random_seed)
    
    layer_sizes = ((2, 34, 34), # (channel, height, width)
                   (6, (3,3), (1,1), (1,1)), # (out_channel, kernel_size, padding, stride)
                   (12, (5,5), (2,2), (1,1)), 
                   (64,), # Fully connected layer
                   (10,))
    load_file = False
    batch_size = 36
    shuffle = False
    init_thresholds = 0.0 #float(jnp.sqrt(2))

    # Load dataset
    (training_generator, total_train_batches), (validation_generator, total_val_batches), (test_generator, total_test_batches), max_nonzero = torch_nmnist_loader(batch_size, shuffle=shuffle)

    params = Params(
        random_seed=random_seed,
        layer_sizes=layer_sizes, 
        init_thresholds=init_thresholds, 
        num_epochs=20, 
        learning_rate=0.0001, 
        batch_size=batch_size,
        load_file=load_file,
        shuffle=shuffle,
        restrict=2,
        firing_nb=128,
        sync_rate=1,
        max_nonzero=max_nonzero,
        shuffle_input=False,
        threshold_lr=0.01, 
        threshold_impact=0.0, # Beta sparse
        rerun="",
        async_layer=-1,
        flat_layer_sizes=()
    )
    
    key, subkey = jax.random.split(key) 
    network = Network(key, params, layer_sizes=layer_sizes)
    weights = network.init_weights()
    # print(weights.shape)

    flat_layer_sizes = []
    for layer in network.layers:
        flat_layer_sizes.append(layer.values.shape)

    print(flat_layer_sizes)
    params = dataclasses.replace(params, flat_layer_sizes=tuple(flat_layer_sizes))

    # convolution(params, network, weights, test_generator, total_train_batches)

    input = jnp.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    kernel = jnp.array([
        [1, 0],
        [0, -1]
    ])

    # Pad input manually to achieve 'same' padding
    padded_input = jnp.pad(input, ((0, 1), (0, 1)))  # bottom and right

    # Shape of output
    out_h = input.shape[0]
    out_w = input.shape[1]

    output = jnp.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = padded_input[i:i+2, j:j+2]
            output = output.at[i, j].set(jnp.sum(patch * kernel))

    print(output)

    