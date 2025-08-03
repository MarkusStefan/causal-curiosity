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
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        height, width, colors = env.observation_dims
        self.output_shape = (colors, height, width)
        self.fc3 = nn.Linear(64 * 7 * 7, 512)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, colors, kernel_size=8, stride=4)
        # reconstruct the image


    def forward(self, z):
        x = self.enc()
        return x
