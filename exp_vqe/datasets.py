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


class CvDataset(Dataset):

    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.dataset = pickle.load(f)
    
    def __getitem__(self, idx):
        time, prob_noisy, prob_ideal = self.dataset[idx]
        return (
            torch.FloatTensor([time]),
            torch.FloatTensor(prob_noisy),
            torch.FloatTensor(prob_ideal)
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


def gen_train_val_identity2(args, miti_prob=False):
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
        for _ in range(200):
            ideal_noisy_states = {}
            ideal_state = random_density_matrix(2 ** num_qubits)
            ideal_noisy_states[0.0] = ideal_state.evolve(cnot_op)
            for noise_scale in np.round(np.arange(0.05, 0.29, 0.02), 3):
                noisy_state = env.simulate_noisy(noise_scale, init_rho=ideal_state.data)
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

                    if indicator < 150:
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


def gen_continuous_train_val(args):
    import qutip
    N = 15
    chi = 1 * 2 * np.pi              # Kerr-nonlinearity
    tlist = np.linspace(0, 1, 20) # time

    a = qutip.destroy(N)
    n = qutip.num(N)
    x = a + a.dag()
    p = -1j * (a - a.dag())
    # H = qutip.Qobj(np.eye(N))
    H = 0.5 * chi * a.dag() * a.dag() * a * a
    grid = 48

    num_alpha = 500
    alphas = 1.5  # np.linspace(1, 2, num_alpha)

    trainset = []
    valset = []
    all_wigner_probs = []
    all_noisy_dynamic_states = []
    xvec = np.linspace(-4, 4, 48)
    for i in tqdm(range(1, len(tlist) + 1)):
        psi0 = qutip.coherent(N, alphas)
        psi0 = psi0.unit()
        rho = psi0 * psi0.dag()
        rho_train = qutip.mesolve(H, rho, tlist[:i], c_ops=[0.6 * a]).states[-1]
        gammas = np.linspace(0.6, 0.8, 5)

        noisy_results = []
        for gamma in gammas:
            c_ops = [gamma * a, ]
            single_noisy_results = []
            for time_idx in range(2, len(tlist) + 1, 2):
                times = tlist[:time_idx]
                result = qutip.mesolve([H, lambda t, args: np.sign(times[-1] / 2 - t)], rho_train, times, c_ops=c_ops).states[-1]
                # result = qutip.mesolve(H, rho, times, c_ops=c_ops).states[-1]
                # ideal_result = rho
                # ideal_result = qutip.mesolve(H, rho, times).states[-1]
                # print(times)
                # print(qutip.fidelity(result, simulated))
                single_noisy_results.append(result)
            noisy_results.append(single_noisy_results)
            
        for t_idx, time in enumerate(tlist):
            if (t_idx + 1) % 2 != 0: continue
            W_noisy = []
            for noise_scale in range(len(noisy_results)):
                noisy_state = noisy_results[noise_scale][t_idx // 2]
                W = qutip.wigner(noisy_state, xvec, xvec, g=2)
                W_noisy.append(np.abs(W / W.sum()))
            W_noiseless = qutip.wigner(rho_train, xvec, xvec, g=2)
            W_noiseless /= W_noiseless.sum()
            W_noiseless = np.abs(W_noiseless)
            trainset.append([time, np.array(W_noisy), W_noiseless])

    train_path = os.path.join(args.out_root, args.out_train)
    with open(train_path, 'wb') as f:
        pickle.dump(trainset, f)
    print(f'Generation finished. Train file saved to {train_path}')
    # test_path = os.path.join(args.out_root, args.out_val)
    # with open(test_path, 'wb') as f:
    #     pickle.dump(valset, f)
    # print(f'Generation finished. Train file saved to {test_path}')


def gen_continuous_test(args):
    import qutip
    N = 15
    chi = 1 * 2 * np.pi              # Kerr-nonlinearity
    tlist = np.linspace(0, 1, 20) # time

    a = qutip.destroy(N)
    n = qutip.num(N)
    x = a + a.dag()
    p = -1j * (a - a.dag())
    H = 0.5 * chi * a.dag() * a.dag() * a * a
    grid = 48

    num_alpha = 11
    alpha = 1.5  # 2

    all_wigner_probs = []
    all_noisy_dynamic_states = []
    testset = []
    xvec = np.linspace(-4, 4, 48)
    wigner_probs = []
    wigner_probs_ideal = []
    psi0 = qutip.coherent(N, alpha)
    psi0 = psi0.unit()
    rho = psi0 * psi0.dag()
    gammas = np.linspace(0.6, 0.8, 5)
    noisy_results = []
    for gamma in gammas:
        c_ops = [gamma * a, ]
        result = qutip.mesolve(H, rho, tlist, c_ops=c_ops)
        noisy_results.append(result.states)
    noiseless_result = qutip.mesolve(H, rho, tlist).states
    # print(fidelity(result.states[0],noiseless_result.states[0]))
    for t_idx, (time, noiseless_state) in enumerate(zip(tlist, noiseless_result)):
        if (t_idx + 1) % 2 != 0: continue
        W_noisy = []
        for noise_scale in range(len(noisy_results)):
            noisy_state = noisy_results[noise_scale][t_idx]
            W = qutip.wigner(noisy_state, xvec, xvec, g=2)
            W_noisy.append(np.abs(W / W.sum()))
        W_noiseless = qutip.wigner(noiseless_state, xvec, xvec, g=2)
        W_noiseless /= W_noiseless.sum()
        W_noiseless = np.abs(W_noiseless)
        # a = W_noisy[0].flatten()
        # b = W_noiseless.flatten()
        # print(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        testset.append([time, np.array(W_noisy), W_noiseless])
        
    test_path = os.path.join(args.out_root, args.out_test)
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
    parser.add_argument('--out-root', default='../data_mitigate/continuous_variable', type=str)
    parser.add_argument('--out-train', default='new_train_coherent.pkl', type=str)
    parser.add_argument('--out-val', default='new_val_coherent.pkl', type=str)
    parser.add_argument('--out-test', default='new_test_coherent.pkl', type=str)
    parser.add_argument('--mitigate-prob', action='store_true', default=False, help='if mitigate probability')
    args = parser.parse_args()
    
    if not os.path.exists(args.out_root):
        os.makedirs(args.out_root)

    # gen_train_val_identity2(args, args.mitigate_prob)
    # gen_test_identity(args, args.mitigate_prob)
    gen_continuous_train_val(args)
    gen_continuous_test(args)

    # import subprocess
    # command = 'cd .. && python circuit.py --data-name vqe.pkl --train-name trainset_vqe.pkl --test-name testset_vqe.pkl --split'
    # subprocess.Popen(command, shell=True).wait()
