import torch
from torch import nn

from encoder import CNNEncoder
from decoder import CNNDecoder


class AutoEncoder2d(nn.Module):

    def __init__(self, shape=(128, 128, 3), latent_dim=256):
        super().__init__()
        self.encoder = CNNEncoder(input_shape=shape, latent_dim=latent_dim)
        self.decoder = CNNDecoder(output_shape=shape, latent_dim=latent_dim)
        

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        :param x: Input tensor.
        :return: Reconstructed tensor.
        """
        z = self.encoder(x)
        x = self.decoder(z)
        return x