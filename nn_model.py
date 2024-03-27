import torch.nn as nn
import torch

activation = nn.LeakyReLU(0.2, inplace=True)
def init_weights(m):
  if type(m) == nn.Linear:
    torch.nn.init.normal_(m.weight, mean=0, std=0.1)

def LinearBlock(hidden_size):
  blockfn = lambda input_size, output_size : [
    nn.Linear(input_size, output_size),
    nn.Dropout(p=0.2),
    activation
  ]
  return [
    blockfn(hidden_size[i//3], hidden_size[i//3+1])[i%3] for i in range(len(hidden_size)*3-3)
  ]

class Generator(nn.Module):
  def __init__(self, input_size: int, output_size: int, hidden_size: list):
    super().__init__()
    self.input_size = input_size
    self.model = nn.Sequential(
      nn.Linear(input_size, hidden_size[0]),
      activation,
      *LinearBlock(hidden_size),
      nn.Linear(hidden_size[-1], output_size),
      nn.Sigmoid() # from 0 to 1
    )
    
  def forward(self, x):
    return self.model(x.view(-1, self.input_size))
  
class Discriminator(nn.Module):
  def __init__(self, input_size: int, output_size: int, hidden_size: list):
    super().__init__()
    self.input_size = input_size
    self.ll = nn.Sequential(
      nn.Linear(input_size, hidden_size[0]),
      activation,
      *LinearBlock(hidden_size),
      nn.Linear(hidden_size[-1], output_size),
      nn.Sigmoid() # from 0 to 1
    )

  def forward(self, x):
    return self.ll(x.view(-1, self.input_size))