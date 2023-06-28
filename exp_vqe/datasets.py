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
from tqdm import tqdm

from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import DensityMatrix, Statevector, random_density_matrix
from qiskit.circuit.library.standard_gates import SdgGate, HGate

from utils import gen_rand_pauli, gen_rand_obs, partial_trace
from my_envs import IBMQEnv
from autoencoder import StateGenerator
from circuit_parser import CircuitParser


class MitigateDataset(Dataset):

    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.dataset = pickle.load(f)

    def __getitem__(self, idx):
        param, obs, pos, noise_scale, exp_noisy, exp_ideal = self.dataset[idx]
        if isinstance(exp_noisy, float):
            exp_noisy = [exp_noisy]
        if isinstance(exp_ideal, float):
            exp_ideal = [exp_ideal]
        exp_noisy = np.array(exp_noisy)
        param_converted = np.rint((param - 0.4) * 10)
        # obs_kron = np.kron(obs[pos[0]], obs[pos[1]])
        obs_kron = np.kron(obs[0], obs[0])
        obs = np.array(obs)
        return (
            torch.FloatTensor([param]),
            torch.tensor([param_converted], dtype=int),
            torch.tensor(obs, dtype=torch.cfloat),
            torch.tensor(obs_kron, dtype=torch.cfloat),
            torch.FloatTensor(pos),
            torch.FloatTensor([noise_scale]),
            torch.FloatTensor(exp_noisy),
            torch.FloatTensor(exp_ideal)
        )

    def __len__(self):
        return len(self.dataset)


def cnots(dim):
    circ = QuantumCircuit(int(np.log2(dim)))
    for i in range(circ.num_qubits - 1):
        circ.cx(i, i + 1)
    for i in range(circ.num_qubits - 1):
        circ.cx(i, i + 1)
    return Operator(circ)


def gen_train_val_identity(args, miti_prob=False):
    trainset = []
    testset = []
    paulis = ['X', 'Y', 'Z']
    paulis_to_projector = {'X': HGate().to_matrix(), 'Y': HGate().to_matrix() @ SdgGate().to_matrix(), 'Z': np.eye(2)}
    env_root = args.env_root
    circ_parser = CircuitParser()
    for circuit_name in tqdm(os.listdir(env_root)):
        param = float(circuit_name.replace('.pkl', '').split('_')[-1])
        # if param not in [0.6, 0.7, 1.0, 1.1, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]: continue
        circuit_path = os.path.join(env_root, circuit_name)
        env = IBMQEnv(circ_path=circuit_path)
        num_qubits = env.circuit.num_qubits
        op_str = 'I' * num_qubits
        cnot_op = cnots(2 ** num_qubits)

        state_array = []
        for _ in range(500):
            ideal_noisy_states = {}
            ideal_state = random_density_matrix(2 ** num_qubits)
            ideal_noisy_states[0.0] = ideal_state.evolve(cnot_op)
            hamiltonian = circ_parser.construct_train(env.circuit)[0]
            for noise_scale in np.round(np.arange(0.05, 0.29, 0.02), 3):
                noisy_state = env.simulate_noisy(noise_scale, hamiltonian, init_rho=ideal_state.data)
                ideal_noisy_states[noise_scale] = noisy_state
            state_array.append(ideal_noisy_states)

        for indicator, ideal_noisy_states in enumerate(state_array):
            for idx in range(num_qubits - 1): # 5
                for obs1, obs2 in itertools.product(paulis, paulis): # 16
                    rand_obs_string = op_str[:idx] + obs1 + obs2 + op_str[idx + 2:]
                    if rand_obs_string == op_str: continue
                    obs_ret = [np.eye(2) for _ in range(num_qubits)]
                    selected_qubits = [idx, idx + 1]
                    noisy_expectations = []

                    if not miti_prob:
                        # rand_obs = [Pauli(obs1).to_matrix(), Pauli(obs2).to_matrix()]
                        obs_ret[idx] = Pauli(obs1).to_matrix()
                        obs_ret[idx + 1] = Pauli(obs2).to_matrix()
                        obs = Pauli(rand_obs_string)
                        obs_ret = np.array(obs_ret)
                        exp_ideal = round(ideal_noisy_states[0.0].expectation_value(obs).real, 6)
                        for noise_scale in np.round(np.arange(0.05, 0.29, 0.02), 3): # 10
                            noisy_expectations.append(round(ideal_noisy_states[noise_scale].expectation_value(obs).real, 6))
                    else:
                        obs_ret[idx] = paulis_to_projector[obs1]
                        obs_ret[idx + 1] = paulis_to_projector[obs2]
                        obs = Operator(functools.reduce(np.kron, obs_ret[::-1]))
                        exp_ideal = ideal_noisy_states[0.0].evolve(obs).probabilities([idx, idx + 1])
                        for noise_scale in np.round(np.arange(0.05, 0.29, 0.02), 3): # 10
                            noisy_expectations.append(ideal_noisy_states[noise_scale].evolve(obs).probabilities([idx, idx + 1]))
                    noisy_expectations = np.array(noisy_expectations)

                    if indicator < 400:
                        trainset.append([param, obs_ret, selected_qubits, noise_scale, noisy_expectations, exp_ideal])
                    else:
                        testset.append([param, obs_ret, selected_qubits, noise_scale, noisy_expectations, exp_ideal])
    train_path = os.path.join(args.out_root, args.out_train)
    with open(train_path, 'wb') as f:
        pickle.dump(trainset, f)
    print(f'Generation finished. Train file saved to {train_path}')
    test_path = os.path.join(args.out_root, args.out_val)
    with open(test_path, 'wb') as f:
        pickle.dump(testset, f)
    print(f'Generation finished. Train file saved to {test_path}')


def gen_test_identity(args, miti_prob=False):
    testset = []
    paulis = ['X', 'Y', 'Z']
    paulis_to_projector = {'X': HGate().to_matrix(), 'Y': HGate().to_matrix() @ SdgGate().to_matrix(), 'Z': np.eye(2)}
    env_root = args.env_root
    for circuit_name in tqdm(os.listdir(env_root)):
        param = float(circuit_name.replace('.pkl', '').split('_')[-1])
        # if param not in [0.6, 0.7, 1.0, 1.1, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]: continue
        circuit_path = os.path.join(env_root, circuit_name)
        env = IBMQEnv(circ_path=circuit_path)
        num_qubits = env.circuit.num_qubits
        op_str = 'I' * num_qubits

        ideal_noisy_states = {}
        ideal_noisy_states[0.0] = env.simulate_ideal()
        for noise_scale in np.round(np.arange(0.05, 0.29, 0.02), 3):
            noisy_state = env.simulate_noisy(noise_scale)
            ideal_noisy_states[noise_scale] = noisy_state

        for idx in range(num_qubits - 1): # 5
            for obs1, obs2 in itertools.product(paulis, paulis): # 16
                rand_obs_string = op_str[:idx] + obs1 + obs2 + op_str[idx + 2:]
                if rand_obs_string == op_str: continue
                obs_ret = [np.eye(2) for _ in range(num_qubits)]
                selected_qubits = [idx, idx + 1]
                noisy_expectations = []

                if not miti_prob:
                    obs_ret[idx] = Pauli(obs1).to_matrix()
                    obs_ret[idx + 1] = Pauli(obs2).to_matrix()
                    obs = Pauli(rand_obs_string)
                    obs_ret = np.array(obs_ret)
                    exp_ideal = round(ideal_noisy_states[0.0].expectation_value(obs).real, 6)
                    for noise_scale in np.round(np.arange(0.05, 0.29, 0.02), 3): # 10
                        noisy_expectations.append(round(ideal_noisy_states[noise_scale].expectation_value(obs).real, 6))
                else:
                    obs_ret[idx] = paulis_to_projector[obs1]
                    obs_ret[idx + 1] = paulis_to_projector[obs2]
                    obs = Operator(functools.reduce(np.kron, obs_ret[::-1]))
                    exp_ideal = ideal_noisy_states[0.0].evolve(obs).probabilities([idx, idx + 1])
                    for noise_scale in np.round(np.arange(0.05, 0.29, 0.02), 3): # 10
                        noisy_expectations.append(ideal_noisy_states[noise_scale].evolve(obs).probabilities([idx, idx + 1]))
                noisy_expectations = np.array(noisy_expectations)
                testset.append([param, obs_ret, selected_qubits, noise_scale, noisy_expectations, exp_ideal])
    test_path = os.path.join(args.out_root, args.out_test)
    with open(test_path, 'wb') as f:
        pickle.dump(testset, f)
    print(f'Generation finished. Train file saved to {test_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-root', default='../environments/circuits/vqe_4l', type=str)
    parser.add_argument('--out-root', default='../data_mitigate/phasedamp', type=str)
    parser.add_argument('--out-train', default='new_train_vqe4l.pkl', type=str)
    parser.add_argument('--out-val', default='new_val_vqe4l.pkl', type=str)
    parser.add_argument('--out-test', default='new_test_vqe4l.pkl', type=str)
    parser.add_argument('--mitigate-prob', action='store_true', default=False, help='if mitigate probability')
    args = parser.parse_args()
    
    if not os.path.exists(args.out_root):
        os.makedirs(args.out_root)

    gen_train_val_identity(args, args.mitigate_prob)
    gen_test_identity(args, args.mitigate_prob)

    # import subprocess
    # command = 'cd .. && python circuit.py --data-name vqe.pkl --train-name trainset_vqe.pkl --test-name testset_vqe.pkl --split'
    # subprocess.Popen(command, shell=True).wait()
