import argparse
import os
import functools
import itertools
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pathos
import cirq
from tqdm import tqdm

from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.quantum_info.operators import Operator, Pauli

from utils import gen_rand_pauli, gen_rand_obs, partial_trace
from my_envs import IBMQEnv, stable_softmax


class SurrogateDataset(Dataset):

    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        self.vqe_params = data_dict['vqe_param']
        self.probabilities = data_dict['probability']
        self.observables = data_dict['observable']
        self.noise_scale = data_dict['noise_scale']
        self.ground_truth = data_dict['meas_result']
        self.positions = data_dict['position']

    def __getitem__(self, idx):
        params, prs, obs = self.vqe_params[idx], self.probabilities[idx], self.observables[idx]
        pos, gts = self.positions[idx], self.ground_truth[idx]
        noise_scale = self.noise_scale[idx]
        # print(params, prs, obs, gts)
        return (
            torch.FloatTensor([params]),
            torch.FloatTensor(prs),
            torch.tensor(obs, dtype=torch.cfloat),
            torch.tensor(pos),
            torch.FloatTensor([noise_scale]),
            torch.FloatTensor([gts])
        )

    def __len__(self):
        return len(self.probabilities)


class SurrogateGenerator:

    def __init__(self, env_root, batch_size, itrs=50):
        # self.env = IBMQEnv.load(env_path)
        self.env_list = []
        for env_name in os.listdir(env_root):
            param = env_name.replace('.pkl', '').split('_')[-1]
            env_path = os.path.join(env_root, env_name)
            env = IBMQEnv(args, circ_path=circuit_path)
            self.env_list.append((env, param))
        self.num_miti_gates = self.env.count_mitigate_gates()
        self.batch_size = batch_size
        self.itrs = itrs
        self.cur_itr = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cur_itr < self.itrs:
            rand_val = torch.rand((self.batch_size, self.num_miti_gates, 4))
            prs = F.softmax(rand_val, dim=-1)
            rand_matrix = torch.randn((self.batch_size, 2, 2), dtype=torch.cfloat)
            obs = torch.bmm(rand_matrix.conj().transpose(-2, -1), rand_matrix)
            env, param = np.random.choice(self.env_list)
            meas = env.step(obs.numpy(), prs.numpy(), nums=2000)
            self.cur_itr += 1
            return param, prs, obs, torch.FloatTensor(meas)
        else:
            self.cur_itr = 0
            raise StopIteration


class MitigateDataset(Dataset):

    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.dataset = pickle.load(f)

    def __getitem__(self, idx):
        param, obs, pos, noise_scale, exp_noisy, exp_ideal = self.dataset[idx]
        param_converted = int((param + 0.2) * 5)
        obs_kron = np.kron(obs[0], obs[1])
        return (
            torch.FloatTensor([param]),
            torch.tensor([param_converted]),
            torch.tensor(obs, dtype=torch.cfloat),
            torch.tensor(obs_kron, dtype=torch.cfloat),
            torch.tensor(pos),
            torch.FloatTensor([noise_scale]),
            torch.FloatTensor([exp_noisy]),
            torch.FloatTensor([exp_ideal])
        )

    def __len__(self):
        return len(self.dataset)


def gen_mitigation_data_ibmq(args):
    env_root = args.env_root
    dataset = []

    for env_name in os.listdir(env_root):
        param = float(env_name.replace('.pkl', '').split('_')[-1])
        env_path = os.path.join(env_root, env_name)
        env = IBMQEnv.load(env_path)
        num_qubits = env.circuit.num_qubits
        ideal_state = env.simulate_ideal()
        for i in tqdm(range(args.num_data)):
            noise_scale, noisy_state = env.simulate_noisy()
            # rand_matrix = torch.randn((2, 2), dtype=torch.cfloat).numpy()
            # rand_obs = rand_matrix.conj().T @ rand_matrix
            rand_obs = [gen_rand_obs() for i in range(args.num_ops)]
            rand_idx = np.random.randint(num_qubits - 1)
            selected_qubits = list(range(rand_idx, rand_idx + args.num_ops))  # [rand_idx, rand_idx + 1]
            obs = [np.eye(2) for i in range(rand_idx)] + rand_obs +\
                  [np.eye(2) for i in range(rand_idx + len(selected_qubits), num_qubits)]
            obs_op = [Operator(o) for o in obs]
            obs_kron = functools.reduce(np.kron, obs[::-1])
            obs_ret = np.array(rand_obs)
            # rand_obs = np.diag([1., -1])
            # obs = np.kron(np.eye(2**3), rand_obs)
            exp_ideal = ideal_state.expectation_value(obs_kron).real  # (ideal_state.conj() @ np.kron(np.eye(2), rand_obs) @ ideal_state).real
            exp_noisy = noisy_state.expectation_value(obs_kron).real
            # exp_ideal = [round(ideal_state.expectation_value(obs_op[i], qargs=[i]).real, 8) for i in range(num_qubits)]
            # exp_noisy = [round(noisy_state.expectation_value(obs_op[i], qargs=[i]).real, 8) for i in range(num_qubits)]
            dataset.append([param, obs_ret, selected_qubits, noise_scale, exp_noisy, exp_ideal])

    with open(args.out_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'Generation finished. File saved to {args.out_path}')


def gen_mitigation_data_pauli(args):
    dataset = []
    paulis = ['I', 'X', 'Y', 'Z']

    for env_name in os.listdir(args.env_root):
        param = float(env_name.replace('.pkl', '').split('_')[-1])
        env_path = os.path.join(args.env_root, env_name)
        env = IBMQEnv.load(env_path)
        num_qubits = env.circuit.num_qubits
        op_str = 'I' * num_qubits
        # print(env.circuit)
        ideal_state = env.simulate_ideal()
        for idx in tqdm(range(num_qubits - 1)): # 5
            for obs1, obs2 in itertools.product(paulis, paulis): # 16
                rand_obs_string = op_str[:idx] + obs1 + obs2 + op_str[idx + 2:]
                if rand_obs_string == op_str: continue
                rand_obs = [Pauli(obs1).to_matrix(), Pauli(obs2).to_matrix()]
                selected_qubits = [idx, idx + 1]
                obs = Pauli(rand_obs_string)
                obs_ret = np.array(rand_obs)
                exp_ideal = ideal_state.expectation_value(obs).real
                for noise_scale in np.round(np.arange(0.01, 0.11, 0.01), 2): # 10
                    # rho = env.simulate_noisy(0.01)
                    # exp_noisy = rho.expectation_value(obs).real
                    # partial_rho = partial_trace(rho, selected_qubits, [2] * num_qubits)
                    # exp_2 = np.trace(partial_rho @ np.kron(obs_ret[0], obs_ret[1])).real
                    # print(exp_noisy, exp_2)
                    # assert False
                    for _ in range(100):
                        exp_noisy = env.sample_noisy(noise_scale, obs)
                        dataset.append([param, obs_ret, selected_qubits, noise_scale, exp_noisy, round(exp_ideal, 6)])
    with open(args.out_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'Generation finished. File saved to {args.out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-root', default='../environments/vqe_envs_train', type=str)
    parser.add_argument('--out-path', default='../data_mitigate/dataset.pkl', type=str)
    parser.add_argument('--num-ops', default=2, type=int)
    parser.add_argument('--num-data', default=20_000, type=int)  # 5000 for train
    args = parser.parse_args()
    # dataset = SurrogateDataset('../data_surrogate/env_vqe_data.pkl')
    # print(next(iter(dataset)))
    # dataset = SurrogateGenerator(args.env_path, batch_size=16, itrs=10)
    # for data in dataset:
    #     print(data)
    gen_mitigation_data_pauli(args)

    # import subprocess
    # command = 'cd .. && python circuit.py --data-name vqe.pkl --train-name trainset_vqe.pkl --test-name testset_vqe.pkl --split'
    # subprocess.Popen(command, shell=True).wait()
