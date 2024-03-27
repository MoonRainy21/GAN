import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
import torchvision
from IPython.display import clear_output
from nn_model import Generator, Discriminator, init_weights
import os

os.makedirs("trainig", exist_ok=True)

MODEL_NAME = "small-dropout"
SHOW = False

CPU = False
MPS = False
CUDA = True

device = torch.device('cpu')
mps = False
if MPS:
  device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
  mps = torch.backends.mps.is_available()
if CUDA:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  cuda = torch.cuda.is_available()
if CPU:
  device = torch.device('cpu')
  mps = False

IMAGE_SIZE = 28 * 28
BATCH_SIZE = 128
EPOCHS = 300000
LEARNING_RATE_GEN = 0.05
LEARNING_RATE_DIS = 0.005
NOISE_DIM = 10

MNIST = torchvision.datasets.MNIST('.', train=True, download=True)
X_train = (MNIST.data.reshape(-1, IMAGE_SIZE) / 255).to(device)
y_train = (MNIST.targets).to(device)

generator_loss = nn.BCELoss()
discriminator_loss = nn.BCELoss()
generator = Generator(NOISE_DIM, IMAGE_SIZE, [512, 512, 256])
discriminator = Discriminator(IMAGE_SIZE, 1, [512, 256])
generator.apply(init_weights)
discriminator.apply(init_weights)
generator.to(device)
discriminator.to(device)

generator.load_state_dict(torch.load(f'model/generator-{MODEL_NAME}.pth',map_location=device))
discriminator.load_state_dict(torch.load(f'model/discriminator-{MODEL_NAME}.pth', map_location=device))

k0 = 5
k = k0
m = 1
optimizer_G = optim.SGD(generator.parameters(), lr=LEARNING_RATE_GEN)
optimizer_D = optim.SGD(discriminator.parameters(), lr=LEARNING_RATE_DIS)
for epoch in range(1, EPOCHS+1):
  for _ in range(k):
    optimizer_D.zero_grad()
    z = torch.randn(BATCH_SIZE, NOISE_DIM).to(device)
    fake_output = generator(z)
    real_output = X_train[np.random.choice(X_train.shape[0], BATCH_SIZE)]
    loss = discriminator_loss(discriminator(real_output), torch.ones(BATCH_SIZE, 1).to(device))
    loss += discriminator_loss(discriminator(fake_output), torch.zeros(BATCH_SIZE, 1).to(device))
    loss /= 2
    loss.backward()
    optimizer_D.step()
    dict_loss = loss.item()


  if epoch % 1000 == 0:
    z = torch.randn(BATCH_SIZE, NOISE_DIM).to(device)
    fake_output = generator(z)
    real_output = X_train[np.random.choice(X_train.shape[0], BATCH_SIZE)]
    loss = discriminator_loss(discriminator(real_output), torch.ones(BATCH_SIZE, 1).to(device))
    loss += discriminator_loss(discriminator(fake_output), torch.zeros(BATCH_SIZE, 1).to(device))
    loss /= 2
    dict_loss = loss.item()

    clear_output(wait=True)
    z = torch.randn(1, NOISE_DIM).to(device)
    fake_output = generator(z)
    plt.imshow(fake_output.view(28, 28).to('cpu').detach().numpy(), cmap='gray')
    plt.savefig(f'training/{MODEL_NAME}-{epoch}.png')
    if SHOW: plt.show()
    print(f'Epoch {epoch}/{EPOCHS} disc_loss: {loss.item()}')
    print(f'fake_probabliity: {discriminator(fake_output).mean()} real_probability: {discriminator(real_output).mean()}')
  for _ in range(m):
    optimizer_G.zero_grad()
    z = torch.randn(BATCH_SIZE, NOISE_DIM).to(device)
    fake_output = generator(z)
    loss = generator_loss(discriminator(fake_output), torch.ones(BATCH_SIZE, 1).to(device))
    asdf = (discriminator(fake_output))
    loss.backward()
    optimizer_G.step()
  if epoch % 1000 == 0:
    print(f'Epoch {epoch}/{EPOCHS} gen_loss: {loss.item()}')
  if epoch % 10000 == 0:
    torch.save(generator.state_dict(), f'training/{MODEL_NAME}-latest-gen.pth')
    torch.save(discriminator.state_dict(), f'training/{MODEL_NAME}-latest-disc.pth')
    