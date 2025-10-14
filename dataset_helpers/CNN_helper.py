import numpy as np

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

def event_conv(event, kernels, padding, stride):
    '''
    Computes the convolution in an event-driven manner by only multiplying the event and the kernels 
    and giving the output events in a list of the format (c, x, y, value) 
    '''
    
    return 