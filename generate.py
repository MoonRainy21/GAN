import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
import torchvision
from IPython.display import clear_output
from nn_model import Generator, Discriminator, init_weights

MODEL_NAME = "small-dropout"
SHOW = False

CPU = False
MPS = False
CUDA = True

IMAGE_SIZE = 28 * 28
BATCH_SIZE = 128
EPOCHS = 300000
LEARNING_RATE_GEN = 0.05
LEARNING_RATE_DIS = 0.005
NOISE_DIM = 10

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

generator = Generator(NOISE_DIM, IMAGE_SIZE, [512, 512, 256])
generator.load_state_dict(torch.load(f'model/generator-{MODEL_NAME}.pth',map_location=device))

images = []
for _ in range(4*4):
  z = torch.randn(1, NOISE_DIM).to(device)
  fake_output = generator(z)
  images.append(fake_output.view(28, 28).to('cpu').detach().numpy())

for idx, image in enumerate(images):
  plt.subplot(4,4, idx+1)
  plt.axis('off')
  plt.imshow(image)
plt.show()