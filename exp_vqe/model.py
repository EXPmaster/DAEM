__all__ = ['SurrogateModel', 'Generator', 'Discriminator']

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateModel(nn.Module):

    def __init__(self, num_qubits, num_obs=2, hidden_size=64):
        super().__init__()
        self.obs_embed = nn.Linear(2 * 2 * 2, hidden_size)
        self.pos_embed = nn.Embedding(num_qubits, hidden_size)
        self.param_embed = nn.Linear(1, hidden_size)
        self.prs_embed = nn.Linear(num_qubits * 4, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(4 * hidden_size, 256),
            nn.Mish(inplace=True),
            nn.Linear(256, 512),
            nn.Mish(inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, param, prs, obs, pos):
        obs_flat = torch.cat((obs.real, obs.imag), -1).flatten(2)
        obs_ebd = self.obs_embed(obs_flat)
        pos_ebd = self.pos_embed(pos)
        obs_ebd = obs_ebd + pos_ebd
        param_ebd = self.param_embed(param)
        prs_ebd = self.prs_embed(prs.flatten(1))
        x = torch.cat((param_ebd.unsqueeze(1), prs_ebd.unsqueeze(1), obs_ebd), 1)
        x = self.embed_ln(x).flatten(1)
        return self.net(x)


class Generator(nn.Module):

    def __init__(self, num_qubits, hidden_size=64):
        super().__init__()
        self.num_qubits = num_qubits
        self.obs_embed = nn.Linear(2 * 2 * 2, hidden_size)
        self.pos_embed = nn.Embedding(num_qubits, hidden_size)
        self.param_embed = nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(3 * hidden_size, 256),
            nn.Mish(inplace=True),
            nn.Linear(256, 512),
            nn.Mish(inplace=True),
            nn.Linear(512, 4 * num_qubits)
        )

    def forward(self, params, obs, pos):
        obs_flat = torch.cat((obs.real, obs.imag), -1).flatten(2)
        obs_ebd = self.obs_embed(obs_flat)
        pos_ebd = self.pos_embed(pos)
        obs_ebd = obs_ebd + pos_ebd
        param_ebd = self.param_embed(params)
        x = torch.cat((param_ebd.unsqueeze(1), obs_ebd), 1)
        x = self.embed_ln(x).flatten(1)
        x = self.net(x)
        x = x.view(-1, self.num_qubits, 4)
        return torch.softmax(x, -1)


class Discriminator(nn.Module):

    def __init__(self, num_qubits, hidden_size=64):
        super().__init__()
        self.num_qubits = num_qubits
        self.obs_embed = nn.Linear(2 * 2 * 2, hidden_size)
        self.pos_embed = nn.Embedding(num_qubits, hidden_size)
        self.meas_embed = nn.Linear(1, hidden_size)
        self.param_embed = nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.discriminator = nn.Sequential(
            nn.Linear(4 * hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, meas_in, params, obs, pos):
        obs_flat = torch.cat((obs.real, obs.imag), -1).flatten(2)
        obs_ebd = self.obs_embed(obs_flat)
        pos_ebd = self.pos_embed(pos)
        obs_ebd = obs_ebd + pos_ebd
        meas_ebd = self.meas_embed(meas_in)
        param_ebd = self.param_embed(params)
        x = torch.cat((meas_ebd.unsqueeze(1), param_ebd.unsqueeze(1), obs_ebd), 1)
        x = self.embed_ln(x).flatten(1)
        return self.discriminator(x)