

"""
Discriminator and Generator implementation from DCGAN paper
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            self._block(1,16, 3, 1, 1),  # img: 4x4
            nn.AvgPool1d(2),
            nn.LeakyReLU(0.2),
            self._block(16,32,3, 1, 1),
            nn.AvgPool1d(2),
            self._block(32,64,3, 1, 1),
            nn.AvgPool1d(2),
            self._block(64,128,3, 1, 1),
            nn.AvgPool1d(2),
            self._block(128,64,3, 1, 1),
            nn.AvgPool1d(2),
            nn.Flatten(),
            nn.Linear(2048,1),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, 32, 3, 1, 1),  # img: 4x4
            self._block(32,32,3,1,1),
            nn.AvgPool1d(2),
            self._block(32,64,3,1,1),
            self._block(64,64,3,1,1),
            nn.AvgPool1d(2),
            self._block(64,128,3,1,1),
            self._block(128,128,3,1,1),
            nn.AvgPool1d(2),
            self._block(128,256,3,1,1),
            self._block(256,256,3,1,1),
            nn.AvgPool1d(2),
            self._block(256,512,3,1,1),
            self._block(512,512,3,1,1),
            nn.AvgPool1d(2),
            self._block(512,1024,3,1,1),
            self._block(1024,1024,3,1,1),
            nn.AvgPool1d(2),
            self._block(1024,2048,3,1,1),
            self._block(2048,2048,3,1,1),
            nn.Flatten(),
            nn.Linear(32768,1024),
        )



    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"


# test()