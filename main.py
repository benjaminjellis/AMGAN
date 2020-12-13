"""Training Script"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models import Generator, Discriminator
from dataset import AMDataset

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

data_dir = "data/processed/"

BATCH_SIZE = 15
# used in data
IMAGE_SIZE = (64, 64)
# number of channels, 3 => colour images
NC = 3
# size of z latent vector
nz = 100
# size of feature maps in generator
ngf = 64
# size of feature maps in discriminator
ndf = 64
# number of workers
NUM_WORKERS = 2
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5


transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
)

dataset = AMDataset(data_path = data_dir, transform = transforms)

dataloader = DataLoader(dataset,
                        batch_size = BATCH_SIZE,
                        shuffle = True,
                        num_workers = NUM_WORKERS
                        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


generator = Generator(nz = nz, ngf = ngf, NC = NC).to(device)

generator.apply(weights_init)

discriminator = Discriminator(NC = NC, ndf = ndf)

discriminator.apply(weights_init)

# criterion is Binary Cross Entropy Loss
criterion = nn.BCELoss

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))