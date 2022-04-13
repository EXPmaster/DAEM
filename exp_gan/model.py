__all__ = ['SurrogateModel', 'MitigateModel']

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


class MitigateModel(nn.Module):

    def __init__(self, num_layers, num_qubits, dim_in):
        super().__init__()
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        self.net = nn.Sequential(
            nn.Linear(dim_in, 128),
            nn.Mish(inplace=True),
            nn.Linear(128, 256),
            nn.Mish(inplace=True),
            nn.Linear(256, 16 * self.num_layers * self.num_qubits)
        )

    def forward(self, obs, exp_noisy):
        obs_cat = torch.cat((obs.real, obs.imag), -1)
        x = torch.cat((obs_cat.flatten(1), exp_noisy), 1)
        x = self.net(x)
        x = x.view(-1, self.num_layers, self.num_qubits, 16)
        return torch.softmax(x, -1)


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