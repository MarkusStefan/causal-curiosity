import torch 
from torch import nn 
from torch.nn import functional as F


# what kernel size, stride, 
# should i use special (regularizaiton) tools 
# - dropout
# - batchnorm
# - activation functions
# how to upsample again and reconstruct the image
# -> U-NET style? ... maybe something more efficient
# what about RNNs ?
# is it advicable to keep symmetry btw encoder and decoder? 
# what pros/cons arise from using different nr and size of layers for the decoder? 



class CNNDecoder(nn.Module):
    def __init__(self, output_shape=(128, 128, 3), latent_dim=256):
        super().__init__()
        height, width, channels = output_shape
        assert height == width == 128, "Only 128x128 output supported"

        self.latent_dim = latent_dim
        self.initial_res = 8  # matches encoder bottleneck
        self.initial_channels = 128  # must match final conv output of encoder

        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, self.initial_res * self.initial_res * self.initial_channels)

        self.unflatten = nn.Unflatten(1, (self.initial_channels, self.initial_res, self.initial_res))

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # 8→16
        self.act1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)   # 16→32
        self.act2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)   # 32→64
        self.act3 = nn.ReLU()
        self.deconv4 = nn.ConvTranspose2d(16, channels, kernel_size=3, stride=2, padding=1, output_padding=1)  # 64→128
        # self.act4 = nn.Sigmoid()  # constrain output to [0,1]

        self.dec = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.unflatten,
            self.deconv1,
            self.act1,
            self.deconv2,
            self.act2,
            self.deconv3,
            self.act3,
            self.deconv4,
            # self.act4,
        )

    def forward(self, z):
        return self.dec(z)


if __name__ == '__main__':
    latent = torch.randn(1, 256)
    decoder = CNNDecoder(output_shape=(128, 128, 3))
    recon = decoder(latent)
    print("Decoder output shape:", recon.shape)  # should be (1, 3, 128, 128)
