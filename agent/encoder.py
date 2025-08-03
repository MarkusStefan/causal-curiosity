import torch 
from torch import nn 
from torch.nn import functional as F


class CNNEncoder(nn.Module):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width, height, colors = env.observation_dims
        self.conv1 = nn.Conv2d(in_channels=colors, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # how to compute the output dims of self.conv layers?

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, width)
        self.enc = nn.Sequential(
            self.conv1, 
            self.conv2,
            self.conv3,
            self.fc1,
            self.fc2,
        )

    def forward(self, x):
        z = self.enc(x)
        return z
