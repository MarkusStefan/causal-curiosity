import math


def compute_conv_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    """
    Compute output size for a single dimension after convolution.
    
    Formula: output_size = floor((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
    
    Args:
        input_size: Input dimension size
        kernel_size: Convolution kernel size
        stride: Stride of convolution
        padding: Padding applied
        dilation: Dilation factor 
    
    Returns:
        Output dimension size
    """
    return math.floor((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1


def compute_conv2d_output_shape(input_shape, kernel_size, stride=1, padding=0, dilation=1):
    """
    Compute output shape for Conv2D layer.
    
    Args:
        input_shape: (height, width) or (channels, height, width)
        kernel_size: int or (kernel_h, kernel_w)
        stride: int or (stride_h, stride_w)
        padding: int or (pad_h, pad_w)
        dilation: int or (dil_h, dil_w)
    
    Returns:
        Output shape tuple
    """
    # Handle different input formats
    if len(input_shape) == 3:
        _, h, w = input_shape
    else:
        h, w = input_shape
    
    # Convert scalars to tuples
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # Compute output dimensions
    out_h = compute_conv_output_size(h, kernel_size[0], stride[0], padding[0], dilation[0])
    out_w = compute_conv_output_size(w, kernel_size[1], stride[1], padding[1], dilation[1])
    
    return (out_h, out_w)
