__all__ = ['SurrogateModel', 'MitigateModel', 'Generator', 'Discriminator']

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateModel(nn.Module):

    def __init__(self, dim_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 256),
            nn.Mish(inplace=True),
            nn.Linear(256, 256),
            nn.Mish(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, prs, obs):
        obs_cat = torch.cat((obs.real, obs.imag), -1)
        x = torch.cat((prs.flatten(1), obs_cat.flatten(1)), 1)
        return self.net(x)


class MitigateModel(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.net = nn.Sequential(
            nn.Linear(dim_in, 128),
            nn.Mish(inplace=True),
            nn.Linear(128, 256),
            nn.Mish(inplace=True),
            nn.Linear(256, dim_out * 4)
        )

    def forward(self, obs, exp_noisy):
        obs_cat = torch.cat((obs.real, obs.imag), -1)
        # x = torch.cat((obs_cat.flatten(1), exp_noisy), 1)
        x = obs_cat.flatten(1)
        x = self.net(x)
        x = x.view(-1, self.dim_out, 4)
        return torch.softmax(x, -1)


class Generator(nn.Module):

    def __init__(self, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.Mish(inplace=True),
            nn.Linear(128, 256),
            nn.Mish(inplace=True),
            nn.Linear(256, 4 * dim_out)
        )

    def forward(self, obs):
        x = torch.cat((obs.real, obs.imag), -1).flatten(1)
        x = self.net(x)
        x = x.view(-1, self.dim_out, 4)
        return torch.softmax(x, -1)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, meas_in, obs):
        obs_cat = torch.cat((obs.real, obs.imag), -1).flatten(1)
        x = torch.cat((meas_in, obs_cat), 1)
        return self.discriminator(x)