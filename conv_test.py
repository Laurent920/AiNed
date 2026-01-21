import jax.numpy as jnp
import jax


# ============================================================================
# Test 1: Basic 2D Convolution Backprop (No Stride, No Padding)
# ============================================================================
print("=" * 60)
print("Test 1: Basic 2D Convolution (stride=1, padding=0)")
print("=" * 60)

# Input: 2 batches, 2 channels, 4x4
input_multi = jnp.stack([
    jnp.arange(1., 17.).reshape(4, 4),
    jnp.arange(17., 33.).reshape(4, 4),
    jnp.arange(34., 50.).reshape(4, 4),
    jnp.arange(51., 67.).reshape(4, 4),
    jnp.arange(68., 84.).reshape(4, 4),
    jnp.arange(85., 101.).reshape(4, 4)
]).reshape(3, 2, 4, 4) # (Batch, Channel, H, W) == (N, C, H, W)

# Filters: 2 output channels, 1 input channels, 2x2 kernel
filter_multi = jnp.array([
    # Output channel 0
    [[1., 0.], [0., 1.]],   # From input channel 0
    [[0., 1.], [1., 0.]]    # From input channel 1
]).reshape(2, 1, 2, 2) # (Out_Channel, In_Channel, k_H, k_W) == (O, I, kH, kW)

print("Input shape: ", input_multi.shape)
print("Filter shape: ", filter_multi.shape)

pad_x, pad_y = 1, 1
k_H, k_W = filter_multi.shape[2], filter_multi.shape[3]
strides = (1, 1) 

lhs = input_multi
rhs = filter_multi.transpose(1, 0, 2, 3)

pad_h = k_H - 1 - pad_x
pad_w = k_W - 1 - pad_y
manual_padding = ((pad_h, pad_h), (pad_w, pad_w))
print("Manual padding for backprop: ", manual_padding)
grad_x = jax.lax.conv_general_dilated(
    lhs=lhs,                 # gradient from next layer
    rhs=rhs,                # flipped weights
    window_strides=(1,1),
    padding=manual_padding,        
    dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
)

print("Grad input shape: ", grad_x.shape)
print(input_multi)
print(filter_multi)
print(grad_x)

"""

# ============================================================================
# Test 2: Convolution with Stride
# ============================================================================
print("\n" + "=" * 60)
print("Test 2: Convolution with Stride (stride=2, padding=0)")
print("=" * 60)

input_stride = jnp.arange(1., 26.).reshape(5, 5)
filter_stride = jnp.array([[1., 2.], [3., 4.]])

output_stride = forward_conv(input_stride, filter_stride, stride=2, padding=0)
print(f"Input shape: {input_stride.shape}")
print(f"Output shape: {output_stride.shape}")

grad_output_stride = jnp.ones_like(output_stride)
grad_input_stride = conv_backprop(grad_output_stride, filter_stride, input_stride.shape, stride=2, padding=0)

print(f"\nGrad input:\n{grad_input_stride}")

def loss_fn_stride(x):
    return jnp.sum(forward_conv(x, filter_stride, stride=2, padding=0))

grad_autodiff_stride = jax.grad(loss_fn_stride)(input_stride)
print(f"Grad from autodiff:\n{grad_autodiff_stride}")
print(f"Match: {jnp.allclose(grad_input_stride, grad_autodiff_stride)}")


# ============================================================================
# Test 3: Multi-Channel Convolution
# ============================================================================
print("\n" + "=" * 60)
print("Test 3: Multi-Channel Convolution (2 in, 3 out)")
print("=" * 60)

# Input: 2 channels, 4x4
input_multi = jnp.stack([
    jnp.arange(1., 17.).reshape(4, 4),
    jnp.arange(17., 33.).reshape(4, 4)
])

# Filters: 3 output channels, 2 input channels, 2x2 kernel
filter_multi = jnp.array([
    # Output channel 0
    [[1., 0.], [0., 1.]],   # From input channel 0
    [[0., 1.], [1., 0.]]    # From input channel 1
]).reshape(1, 2, 2, 2)

filter_multi = jnp.concatenate([
    filter_multi,
    filter_multi * 0.5,  # Output channel 1
    filter_multi * 2.0   # Output channel 2
], axis=0)

print(f"Input shape: {input_multi.shape}")  # (2, 4, 4)
print(f"Filter shape: {filter_multi.shape}")  # (3, 2, 2, 2)

output_multi = forward_conv(input_multi, filter_multi, stride=1, padding=0)
print(f"Output shape: {output_multi.shape}")  # (3, 3, 3)

grad_output_multi = jnp.ones_like(output_multi)
grad_input_multi = conv_backprop(grad_output_multi, filter_multi, input_multi.shape, stride=1, padding=0)

print(f"\nGrad input shape: {grad_input_multi.shape}")

def loss_fn_multi(x):
    return jnp.sum(forward_conv(x, filter_multi, stride=1, padding=0))

grad_autodiff_multi = jax.grad(loss_fn_multi)(input_multi)
print(f"\nGrad from custom backprop:\n{grad_input_multi}")
print(f"\nGrad from autodiff:\n{grad_autodiff_multi}")
print(f"\nMatch: {jnp.allclose(grad_input_multi, grad_autodiff_multi, atol=1e-5)}")


# ============================================================================
# Test 4: Multi-Channel with Stride
# ============================================================================
print("\n" + "=" * 60)
print("Test 4: Multi-Channel with Stride (stride=2)")
print("=" * 60)

input_multi_stride = jnp.stack([
    jnp.arange(1., 26.).reshape(5, 5),
    jnp.arange(26., 51.).reshape(5, 5)
])

filter_multi_stride = jnp.array([
    [[1., 0.], [0., 1.]],
    [[0., 1.], [1., 0.]]
]).reshape(1, 2, 2, 2)

output_multi_stride = forward_conv(input_multi_stride, filter_multi_stride, stride=2, padding=0)
print(f"Input shape: {input_multi_stride.shape}")
print(f"Output shape: {output_multi_stride.shape}")

grad_output_multi_stride = jnp.ones_like(output_multi_stride)
grad_input_multi_stride = conv_backprop(grad_output_multi_stride, filter_multi_stride, 
                                        input_multi_stride.shape, stride=2, padding=0)

def loss_fn_multi_stride(x):
    return jnp.sum(forward_conv(x, filter_multi_stride, stride=2, padding=0))

grad_autodiff_multi_stride = jax.grad(loss_fn_multi_stride)(input_multi_stride)
print(f"\nMatch: {jnp.allclose(grad_input_multi_stride, grad_autodiff_multi_stride, atol=1e-5)}")


# ============================================================================
# Test 5: Simple Manual Verification
# ============================================================================
print("\n" + "=" * 60)
print("Test 5: Simple Manual Verification (2x2 -> 1x1)")
print("=" * 60)

input_simple = jnp.array([[1., 2.], [3., 4.]])
filter_simple = jnp.array([[1., 0.], [0., 1.]])

output_simple = forward_conv(input_simple, filter_simple)
print(f"Input:\n{input_simple}")
print(f"Filter:\n{filter_simple}")
print(f"Output: {output_simple}")  # Should be [[5.]]

grad_out_simple = jnp.array([[1.]])
grad_in_simple = conv_backprop(grad_out_simple, filter_simple, input_simple.shape)

print(f"\nGrad input:\n{grad_in_simple}")
print(f"Expected:\n[[1, 0],\n [0, 1]]")

def loss_simple(x):
    return jnp.sum(forward_conv(x, filter_simple))

grad_autodiff_simple = jax.grad(loss_simple)(input_simple)
print(f"\nGrad from autodiff:\n{grad_autodiff_simple}")
print(f"Match: {jnp.allclose(grad_in_simple, grad_autodiff_simple)}")
"""