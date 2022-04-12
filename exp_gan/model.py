__all__ = ['SurrogateModel']

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateModel(nn.Module):

    def __init__(self, dim_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.Mish(inplace=True),
            nn.Linear(1024, 1024),
            nn.Mish(inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, prs, obs):
        obs_cat = torch.cat((obs.real, obs.imag), -1)
        x = torch.cat((prs.flatten(1), obs_cat.flatten(1)), 1)
        return self.net(x)


class Generator(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        ...


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        ...