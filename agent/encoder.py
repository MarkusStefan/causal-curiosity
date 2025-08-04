import torch 
from torch import nn 
from torch.nn import functional as F

import sys 
import os 

sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # to import utils

from utils.networks import compute_conv2d_output_shape, compute_padding


class CNNEncoder(nn.Module):
    def __init__(self, input_shape=(128, 128, 3), latent_dim=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        height, width, channels = input_shape
        assert width == height, "Only square inputs are supported in this config"
        assert width == 128, "Input should be 128x128 for this config"

        nr_filters_out = 16
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=nr_filters_out, kernel_size=3, stride=2, padding=1)
        self.act1 = nn.ReLU()

        nr_filters_in = nr_filters_out
        nr_filters_out *= 2  # 32
        self.conv2 = nn.Conv2d(nr_filters_in, nr_filters_out, kernel_size=3, stride=2, padding=1)
        self.act2 = nn.ReLU()

        nr_filters_in = nr_filters_out
        nr_filters_out *= 2  # 64
        self.conv3 = nn.Conv2d(nr_filters_in, nr_filters_out, kernel_size=3, stride=2, padding=1)
        self.act3 = nn.ReLU()

        nr_filters_in = nr_filters_out
        nr_filters_out *= 2  # 128
        self.conv4 = nn.Conv2d(nr_filters_in, nr_filters_out, kernel_size=3, stride=2, padding=1)
        self.act4 = nn.ReLU()

        self.flatten = nn.Flatten()

        # compute output dims after 4 convolutions
        output_dims = compute_conv2d_output_shape((height, width), kernel_size=3, stride=2, padding=1)
        output_dims = compute_conv2d_output_shape(output_dims, kernel_size=3, stride=2, padding=1)
        output_dims = compute_conv2d_output_shape(output_dims, kernel_size=3, stride=2, padding=1)
        output_dims = compute_conv2d_output_shape(output_dims, kernel_size=3, stride=2, padding=1)

        assert output_dims == (8, 8), f"Expected output dimensions to be (4, 4), got {output_dims}"

        flattened_dim_in = 8 * 8 * nr_filters_out  # = 2048
        flattened_dim_out = flattened_dim_in // 2  # = 1024
        self.bottleneck_fc1 = nn.Linear(flattened_dim_in, flattened_dim_out)
        self.bottleneck_fc2 = nn.Linear(flattened_dim_out, latent_dim)

        self.enc = nn.Sequential(
            self.conv1,
            self.act1,
            self.conv2,
            self.act2,
            self.conv3,
            self.act3,
            self.conv4,
            self.act4,
            self.flatten,
            self.bottleneck_fc1,
            self.bottleneck_fc2,
        )

    def forward(self, x):
        return self.enc(x)




class CNNVariationalEncoder(nn.Module):
    def __init__(self, input_shape=(128, 128, 3), latent_dim=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        height, width, channels = input_shape
        assert width == height, "Only square inputs are supported in this config"
        assert width == 128, "Input should be 128x128 for this config"

        nr_filters_out = 16
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=nr_filters_out, kernel_size=3, stride=2, padding=1)
        self.act1 = nn.ReLU()

        nr_filters_in = nr_filters_out
        nr_filters_out *= 2  # 32
        self.conv2 = nn.Conv2d(nr_filters_in, nr_filters_out, kernel_size=3, stride=2, padding=1)
        self.act2 = nn.ReLU()

        nr_filters_in = nr_filters_out
        nr_filters_out *= 2  # 64
        self.conv3 = nn.Conv2d(nr_filters_in, nr_filters_out, kernel_size=3, stride=2, padding=1)
        self.act3 = nn.ReLU()

        nr_filters_in = nr_filters_out
        nr_filters_out *= 2  # 128
        self.conv4 = nn.Conv2d(nr_filters_in, nr_filters_out, kernel_size=3, stride=2, padding=1)
        self.act4 = nn.ReLU()
        
        self.flatten = nn.Flatten()

        # The output of the last convolutional layer will have dimensions 8x8
        # (128 -> 64 -> 32 -> 16 -> 8)
        # 8 * 8 * 128 = 8192, not 2048 as in your comment. I'll correct the numbers.
        flattened_dim_in = 8 * 8 * nr_filters_out  # = 8192
        flattened_dim_out = flattened_dim_in // 2  # = 4096
        
        self.bottleneck_fc = nn.Linear(flattened_dim_in, flattened_dim_out)
        self.bottleneck_act = nn.ReLU()

        # VAE-specific layers
        self.fc_mu = nn.Linear(flattened_dim_out, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim_out, latent_dim)

        self.enc = nn.Sequential(
            self.conv1,
            self.act1,
            self.conv2,
            self.act2,
            self.conv3,
            self.act3,
            self.conv4,
            self.act4,
            self.flatten,
            self.bottleneck_fc,
            self.bottleneck_act,
        )


    def forward(self, x):
        hidden = self.enc(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        """ 
        Reparameterization trick to sample from the latent space, typically N(mu, exp(logvar)).
            --> this allows gradients to flow through the sampling process
            --> despite sampling from distribution (intractability) enabling backprop
        """
        std = torch.exp(0.5 * logvar) # 0.5 is needed to get the standard deviation from log variance
        eps = torch.randn_like(std)
        return mu + eps * std





# class CNNEncoder_(nn.Module):
#     def __init__(self, env, latent_dim=256, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         width, height, colors = env.observation_dims
#         nr_filters_in = colors # 3
#         nr_filters_out = 16
#         self.conv1 = nn.Conv2d(in_channels=nr_filters_in, out_channels=nr_filters_out, kernel_size=(3,3), stride=2, padding=1)
#         self.act1 = nn.ReLU()
#         output_dims = compute_conv2d_output_shape((height, width), kernel_size=(3,3), stride=4, padding=1)
#         print(f"output_dims: {output_dims}, nr_filters_out: {nr_filters_out}")

#         nr_filters_in = nr_filters_out 
#         nr_filters_out *= 2 # 32
#         self.conv2 = nn.Conv2d(nr_filters_in, nr_filters_out, kernel_size=(3,3), stride=2, padding=1)
#         self.act2 = nn.ReLU()
#         output_dims = compute_conv2d_output_shape(output_dims, kernel_size=(3,3), stride=2, padding=1)
#         print(f"output_dims: {output_dims}, nr_filters_out: {nr_filters_out}")

#         nr_filters_in = nr_filters_out 
#         nr_filters_out *= 2 # 64
#         self.conv3 = nn.Conv2d(nr_filters_in, nr_filters_out, kernel_size=(3,3), stride=2, padding=1)
#         self.act3 = nn.ReLU()
#         output_dims = compute_conv2d_output_shape(output_dims, kernel_size=(3,3), stride=2, padding=1)
#         print(f"output_dims: {output_dims}, nr_filters_out: {nr_filters_out}")

#         nr_filters_in = nr_filters_out 
#         nr_filters_out *= 2 # 128
#         self.conv4 = nn.Conv2d(nr_filters_in, nr_filters_out, kernel_size=(3,3), stride=2, padding=1)
#         self.act4 = nn.ReLU()
#         output_dims = compute_conv2d_output_shape(output_dims, kernel_size=(3,3), stride=2, padding=1)
#         print(f"output_dims: {output_dims}, nr_filters_out: {nr_filters_out}")

#         self.flatten = nn.Flatten()

#         flattened_dim_in = output_dims[0] * output_dims[1] * nr_filters_out
#         flattened_dim_out = flattened_dim_in // 2
#         print(f"flattened_dim_in: {flattened_dim_in}, output_dims: {flattened_dim_out}")
#         self.bottleneck_fc1 = nn.Linear(flattened_dim_in, flattened_dim_out)
#         flattened_dim_in = flattened_dim_out
#         flattened_dim_out = latent_dim
#         print(f"flattened_dim_in: {flattened_dim_in}, latent_dim: {latent_dim}")
#         self.bottleneck_fc2 = nn.Linear(flattened_dim_in, flattened_dim_out)

#         # model architecture
#         self.enc = nn.Sequential(
#             self.conv1,
#             self.act1,
#             self.conv2,
#             self.act2,
#             self.conv3,
#             self.act3,
#             self.conv4,
#             self.act4,
#             self.flatten,
#             self.bottleneck_fc1,
#             self.bottleneck_fc2,
#         )

#     def forward(self, x):
#         z = self.enc(x)
#         return z


if __name__ == '__main__':
    # Test the CNNEncoder
    env = type('Env', (), {'observation_dims': (128, 128, 3)})()  # Mock environment
    encoder = CNNEncoder(env)
    sample_input = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image
    output = encoder(sample_input)
    print(f"Output shape: {output.shape}")  # Should be (1, latent_dim)