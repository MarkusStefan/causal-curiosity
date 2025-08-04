import math


def compute_conv_output_size(input_size, kernel_size, stride=1, padding=0):
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
    return math.floor((input_size + 2*padding - kernel_size) / stride) + 1


def compute_conv2d_output_shape(input_shape, kernel_size, stride=1, padding='same'):
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
    
    # Convert scalars to tuples if thei are not provided as tuples yet
    # assumes symmetry
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    elif isinstance(padding, str) and padding == 'same':
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

    
    # Compute output dimensions
    out_h = compute_conv_output_size(h, kernel_size[0], stride[0], padding[0])
    out_w = compute_conv_output_size(w, kernel_size[1], stride[1], padding[1])
    
    return (out_h, out_w)


def compute_padding(input_shape, output_shape, kernel_size, stride=1):
    """
    Compute padding needed to achieve a specific output shape.
    
    Args:
        input_shape: (height, width)
        output_shape: (out_height, out_width)
        kernel_size: int or (kernel_h, kernel_w)
        stride: int or (stride_h, stride_w)
    
    Returns:
        Padding tuple (pad_h, pad_w)
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    pad_h = max(0, ((output_shape[0] - 1) * stride[0] + kernel_size[0] - input_shape[0]) // 2)
    pad_w = max(0, ((output_shape[1] - 1) * stride[1] + kernel_size[1] - input_shape[1]) // 2)

    return (pad_h, pad_w)


if __name__ == '__main__':
    output_dim = compute_conv2d_output_shape(input_shape=(128, 128), kernel_size=(3,3), stride=3, padding='same')
    print(f"output 1: {output_dim}")
    output_dim = compute_conv2d_output_shape(output_dim, 3, 2)
    print(f"output 2: {output_dim}")
    output_dim = compute_conv2d_output_shape(output_dim, 3, 2)
    print(f"output 3: {output_dim}")
    output_dim = compute_conv2d_output_shape(output_dim, 3, 2)
    print(f"output 4: {output_dim}")
    output_dim = compute_conv2d_output_shape(output_dim, 3, 2)
    print(f"output 5: {output_dim}")
    output_dim = compute_conv2d_output_shape(output_dim, 3, 2)
    print(f"output 6: {output_dim}")