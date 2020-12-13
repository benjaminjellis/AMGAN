"""Discriminator and Generator Models"""

import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, nz: int, ngf: int, NC: int):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # conv 1
            nn.ConvTranspose2d(in_channels = nz,
                               out_channels = ngf * 8,
                               kernel_size = 4,
                               stride = 1,
                               padding = 0,
                               bias = False),
            nn.BatchNorm2d(num_features = ngf * 8),
            nn.ReLU(True),

            # conv 2
            nn.ConvTranspose2d(in_channels = ngf * 8,
                               out_channels = ngf * 4,
                               kernel_size = 4,
                               stride = 2,
                               padding = 1,
                               bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # conv 3
            nn.ConvTranspose2d(in_channels = ngf * 4,
                               out_channels = ngf * 2,
                               kernel_size = 4,
                               stride = 2,
                               padding = 1,
                               bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # conv 4
            nn.ConvTranspose2d(in_channels = ngf * 2,
                               out_channels = ngf,
                               kernel_size = 4,
                               stride = 2,
                               padding = 1,
                               bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # conv 5
            nn.ConvTranspose2d(in_channels = ngf,
                               out_channels = NC,
                               kernel_size = 4,
                               stride = 2,
                               padding = 1,
                               bias = False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):

    def __init__(self, NC: int, ndf: int):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # conv 1
            nn.Conv2d(in_channels = NC,
                      out_channels = ndf,
                      kernel_size = 4,
                      stride = 2,
                      padding = 1,
                      bias = False
                      ),
            nn.LeakyReLU(negative_slope = 0.2,
                         inplace = True),
            # conv 2
            nn.Conv2d(in_channels = ndf,
                      out_channels = ndf * 2,
                      kernel_size = 4,
                      stride = 2,
                      padding = 1,
                      bias = False),
            nn.BatchNorm2d(num_features = ndf * 2),
            nn.LeakyReLU(negative_slope = 0.2,
                         inplace = True),
            # conv 3
            nn.Conv2d(in_channels = ndf * 2,
                      out_channels = ndf * 4,
                      kernel_size = 4,
                      stride = 2,
                      padding = 1,
                      bias = False),
            nn.BatchNorm2d(num_features = ndf * 4),
            nn.LeakyReLU(negative_slope = 0.2,
                         inplace = True),
            # conv 4
            nn.Conv2d(in_channels = ndf * 4,
                      out_channels = ndf * 8,
                      kernel_size = 4,
                      stride = 2,
                      padding = 1,
                      bias = False),
            nn.BatchNorm2d(num_features = ndf * 8),
            nn.LeakyReLU(negative_slope = 0.2,
                         inplace = True),
            # conv 5
            nn.Conv2d(in_channels = ndf * 8,
                      out_channels = 1,
                      kernel_size = 4,
                      stride = 1,
                      padding = 0,
                      bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
