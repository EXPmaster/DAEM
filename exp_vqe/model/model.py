__all__ = ['SuperviseModel', 'Generator', 'Discriminator']

import os
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit.library.standard_gates import HGate, SdgGate, IGate

from my_envs import IBMQEnv
from utils import partial_trace


class AEModel(nn.Module):

    def __init__(self, indim):
        super().__init__()
        self.net = nn.Sequential(
            # Encoder
            nn.Linear(indim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            # Decoder
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, indim)
            # nn.Tanh()
        )

    def forward(self, x):
        # real = torch.real(x)
        # imag = torch.imag(x)
        # x = torch.cat([real, imag], -1).flatten(1)
        # x = self.net(x)
        # real = x[:, :x.shape[1] // 2]
        # imag = x[:, x.shape[1] // 2:]
        # r = (real + 1j * imag).reshape(-1, 16, 16)
        # # density_mat = r + r.mH
        # density_mat = r
        # return density_mat / density_mat.diagonal(dim1=-2, dim2=-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
        return self.net(x)


class SuperviseModel(nn.Module):

    def __init__(self, num_qubits=6, hidden_size=128, mitigate_prob=False):
        super().__init__()
        self.mitigate_prob = mitigate_prob
        self.num_qubits = num_qubits
        if self.mitigate_prob:
            out_dim = 2 ** 2
            obs_embed_dim = num_qubits * 2 * 2 * 2 if self.num_qubits < 10 else 2 * 2 * 2 * 2
            exp_embed_dim = 12 * 2 ** 2
            # hidden_size = 256
        else:
            out_dim = 1
            obs_embed_dim = num_qubits * 2 * 2 * 2 if self.num_qubits < 10 else 2 * 2 * 2 * 2
            exp_embed_dim = 4  # 12
        self.obs_embed = nn.Linear(obs_embed_dim, hidden_size)
        self.param_embed = nn.Linear(1, hidden_size)
        self.exp_embed = nn.Linear(exp_embed_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        # for large scale data
        if self.num_qubits >= 10:
            self.pos_embed = nn.Embedding(50, hidden_size)
       
        self.net = nn.Sequential(
            nn.Linear(3 * hidden_size, 512),
            # nn.Dropout(0.5),
            nn.Mish(inplace=True),
            nn.Linear(512, 1024),
            # nn.Dropout(0.5),
            nn.Mish(inplace=True),
            nn.Linear(1024, 1024),
            nn.Mish(inplace=True),
            nn.Linear(1024, out_dim),
            # nn.Tanh()
        )

    def forward(self, params, obs, pos, scale, noisy_exp):
        obs_flat = torch.cat((obs.real, obs.imag), -1).flatten(1)
        obs_ebd = self.obs_embed(obs_flat)
        exp_ebd = self.exp_embed(noisy_exp.flatten(1))
        # for large scale
        if self.num_qubits > 10:
            pos_ebd = self.pos_embed(pos[:, 0])
            obs_ebd = obs_ebd + pos_ebd
        
        param_ebd = self.param_embed(params)
        # scale_ebd = self.scale_embed(scale)
        x = torch.cat((param_ebd, obs_ebd, exp_ebd), 1)
        # x = torch.cat((noise.unsqueeze(1), obs_ebd, scale_ebd.unsqueeze(1)), 1)
        # x = self.embed_ln(x).flatten(1)
        x = self.net(x)
        return x

    def construct_shadow(self, params, num_samples=10000):
        # from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
        shadow = np.zeros((2 ** self.num_qubits, 2 ** self.num_qubits), dtype=complex)
        paulis = ['X', 'Y', 'Z']
        hadamard = HGate().to_matrix()
        sdg = SdgGate().to_matrix()
        identity = IGate().to_matrix()
        zero_state = np.array([[1., 0.], [0., 0.]])
        one_state = np.array([[0., 0.], [0., 1.]])
        bit_states = [zero_state, one_state]
        unitaries = {'X': hadamard, 'Y': hadamard @ sdg, 'Z': identity}
        scale = torch.FloatTensor([0.])[None].to(params.device)

        for itr in range(num_samples):
            rho_snapshot = [1]
            for idx in range(self.num_qubits):
                rand_pauli = np.random.choice(paulis)
                U = unitaries[rand_pauli]
                obs = [np.eye(2) for _ in range(self.num_qubits)]
                obs[idx] = Pauli(rand_pauli).to_matrix()
                obs = torch.tensor(np.array(obs), dtype=torch.cfloat, device=params.device)[None]
                distr = self.forward(params, obs, None, scale).softmax(1).cpu().numpy()
                
                out_state_idx = np.random.choice(np.arange(2), p=distr.squeeze())
                out_state = bit_states[out_state_idx]
                local_rho = 3 * (U.conj().T @ out_state @ U) - identity
                rho_snapshot = np.kron(rho_snapshot, local_rho)
            shadow += rho_snapshot
        shadow /= num_samples
        # print(shadow)
        # print(np.linalg.eigvalsh(shadow / np.trace(shadow)))
        return shadow / np.trace(shadow)
