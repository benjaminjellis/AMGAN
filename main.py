"""Training Script"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models import Generator, Discriminator
from dataset import AMDataset

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import torchvision.utils as vutils
import random
from tqdm import tqdm


manualSeed = 25
random.seed(manualSeed)
torch.manual_seed(manualSeed)

data_dir = "data/processed/"

BATCH_SIZE = 10
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
NUM_WORKERS = 0
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# epochs
EPOCHS = 400

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
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device = device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr = lr, betas = (beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr = lr, betas = (beta1, 0.999))

# /// --- TRAINING --- \\\

img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(EPOCHS):

    dataloader = tqdm(dataloader)

    for i, data in enumerate(dataloader, 0):
        # first train the discriminator with real mini batch
        discriminator.zero_grad()
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype = torch.float, device = device)
        output = discriminator(real_cpu).view(-1)
        discriminator_error_real = criterion(output, label)
        discriminator_error_real.backward()
        D_x = output.mean().item()

        # first train the discriminator with fake mini batch
        # create a latent vector of noise
        noise_latent_vector = torch.randn(b_size, nz, 1, 1, device = device)
        # pass the latent vector to the generator which creates fake images
        fake_images = generator(noise_latent_vector)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = discriminator(fake_images.detach()).view(-1)
        discriminator_error_fake = criterion(output, label)
        discriminator_error_fake.backward()
        D_G_z1 = output.mean().item()
        discriminator_error = discriminator_error_real + discriminator_error_fake

        optimizerD.step()

        # now train generator
        generator.zero_grad()
        # we use real labels for fake images to maximise log(D(G(z))
        label.fill_(real_label)
        output = discriminator(fake_images).view(-1)
        generator_error = criterion(output, label)
        generator_error.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 5 == 0:
            print("\n[{}/{}][{}/{}] | Loss D: {} | Loss G: {} | D(x): {} | D(G(z)): {} / {}".format(epoch + 1, EPOCHS, i,
                                                                                                  len(dataloader),
                                                                                                  discriminator_error.item(),
                                                                                                  generator_error.item(),
                                                                                                  D_x,
                                                                                                  D_G_z1,
                                                                                                  D_G_z2
                                                                                                  ))
        G_losses.append(generator_error.item())
        D_losses.append(discriminator_error.item())

        if (iters % 500 == 0) or ((epoch == EPOCHS - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding = 2, normalize = True))

        iters += 1


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
#plt.show()


fig = plt.figure(figsize=(8,8))
plt.axis("off")
print(len(img_list))
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
ani.save("animation.gif")
