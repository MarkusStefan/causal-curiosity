

# CNN
## Params
- kernel size [low] (typically 3x3)
    - pro: capture fine-grained features, increase depth with fewer params per layer to learn more complex patterns, computationally cheaper
    - con: requires more layers to capture larger features 
    - when to use: good for feature abstractions through deeper networks, good for detailed patterns (e.g., image recognition tasks)

- stride [high] (tipically >= 1)
    - pro: reduces spatial dimensions, allows for larger receptive fields, captures more global features
    - con: lower resolution feature maps, may miss fine details, can lead to loss of information
    - when to use: useful alternative to pooling layers (downsampling) ... stride of 2 halves the spatial dimensions

- padding [same] (typically 'same'=depends on stride or 'valid'=0)
    - pro: enough 0 are added to prevent the output size from shrinking, preserves spatial dimensions
    - con: "unreal" 0s 
    - when to use: useful for maintaining spatial dimensions, especially in deeper networks

## Pooling
- max pooling [low] (typically 2x2)
    - pro: reduces spatial dimensions, retains important features, computationally efficient
    - con: can lose fine-grained information, may not be suitable for all tasks
    - when to use: commonly used in CNNs to downsample feature maps, effective for image data
- average pooling [low] (typically 2x2)
    - pro: smooths the feature maps, retains more information than max pooling, computationally efficient
    - con: can blur important features, may not be suitable for all tasks
    - when to use: useful when you want to retain more information, can be used in CNNs for downsampling
- in SOTA applications, pooling layers are often replaced with strided convolutions to maintain more information and learn more complex patterns as strided convolutions alow for flexible downsampling while learning spatial hierarchies

## Activation Functions
- ReLU (Rectified Linear Unit)
    - pro: mitigates vanishing gradient problem, computationally efficient, allows for sparse activation
    - con: can suffer from dying ReLU problem (neurons can become inactive if inputs are always negative)
    - when to use: widely used in hidden layers of CNNs, especially effective for image data
- Leaky ReLU
    - pro: allows a small, non-zero gradient when the unit is inactive, mitigates dying ReLU problem
    - con: introduces a small amount of noise, may not be as effective in all cases
    - when to use: useful when you want to avoid dead neurons, can be used in hidden layers of CNNs
- ELU (Exponential Linear Unit)
    - pro: smooths the activation function, allows for negative values, mitigates vanishing gradient problem
    - con: computationally more expensive than ReLU, can introduce noise
    - when to use: useful in deeper networks where smoothness is beneficial, can be used in hidden layers of CNNs


## Computing the output dimensions 
$$
\begin{align*}
W_{out} &= \frac{W_{in} + 2 \cdot P - K}{S} + 1 \\
H_{out} &= \frac{H_{in} + 2 \cdot P - K}{S} + 1 \\
C_{out} &= F
\end{align*}
$$
where:
- \(W_{out}\) = output width
- \(H_{out}\) = output height
- \(C_{out}\) = output channels
- \(W_{in}\) = input width
- \(H_{in}\) = input height
- \(P\) = padding
- \(K\) = kernel size
- \(S\) = stride
- \(F\) = number of filters


# VAE
## Encoder
- learn additional parameters (mean and variance) for each latent variable
- use a reparameterization trick to sample from the learned distribution
- typically consists of several convolutional layers followed by fully connected layers