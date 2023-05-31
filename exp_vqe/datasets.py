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

from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.quantum_info.operators import Operator, Pauli

from utils import gen_rand_pauli, gen_rand_obs, partial_trace
from my_envs import IBMQEnv
from autoencoder import StateGenerator


class MitigateDataset(Dataset):

    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.dataset = pickle.load(f)

    def __getitem__(self, idx):
        param, obs, pos, noise_scale, exp_noisy, exp_ideal = self.dataset[idx]
        if isinstance(exp_noisy, float):
            exp_noisy = [exp_noisy]
            exp_ideal = [exp_ideal]
        param_converted = np.rint((param - 0.4) * 10)
        # obs_kron = np.kron(obs[pos[0]], obs[pos[1]])
        obs_kron = np.kron(obs[0], obs[0])
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
    paulis = ['X', 'Y', 'Z']

    for env_name in tqdm(os.listdir(args.env_root)):
        param = float(env_name.replace('.pkl', '').split('_')[-1])
        env_path = os.path.join(args.env_root, env_name)
        env = IBMQEnv.load(env_path)
        num_qubits = env.circuit.num_qubits
        op_str = 'I' * num_qubits
        ideal_state = env.simulate_ideal()
        for idx in range(num_qubits - 1): # 5
            for noise_scale in np.round(np.arange(0.05, 0.29, 0.01), 3): # 10
                noisy_state = env.simulate_noisy(noise_scale)
                for obs1, obs2 in itertools.product(paulis, paulis): # 16
                    rand_obs_string = op_str[:idx] + obs1 + obs2 + op_str[idx + 2:]
                    if rand_obs_string == op_str: continue
                    obs_ret = [np.eye(2) for _ in range(num_qubits)]
                    # rand_obs = [Pauli(obs1).to_matrix(), Pauli(obs2).to_matrix()]
                    obs_ret[idx] = Pauli(obs1).to_matrix()
                    obs_ret[idx + 1] = Pauli(obs2).to_matrix()
                    selected_qubits = [idx, idx + 1]
                    obs = Pauli(rand_obs_string)
                    obs_ret = np.array(obs_ret)
                    exp_ideal = ideal_state.expectation_value(obs).real
                    exp_noisy = noisy_state.expectation_value(obs).real
                    # for _ in range(100):
                    #     # sample_noisy = env.sample_noisy(noise_scale, obs)
                    #     sample_noisy = np.random.normal(exp_noisy, 0.0001)
                    dataset.append([param, obs_ret, selected_qubits, noise_scale, round(exp_noisy, 6), round(exp_ideal, 6)])
    out_path = os.path.join(args.out_root, args.out_train)
    with open(out_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'Generation finished. File saved to {out_path}')


def gen_mitigation_data_pauli_sampling(args):
    dataset = []
    paulis = ['X', 'Y', 'Z']

    for env_name in os.listdir(args.env_root):
        param = float(env_name.replace('.pkl', '').split('_')[-1])
        env_path = os.path.join(args.env_root, env_name)
        env = IBMQEnv.load(env_path)
        num_qubits = env.circuit.num_qubits
        op_str = 'I' * num_qubits
        ideal_state = env.simulate_ideal()

        for idx in range(num_qubits - 1): # 3
            rho_0 = env.simulate_noisy(0.01)
            reduced_rho = [partial_trace(rho_0, [idx, idx+1], [2] * num_qubits)]
            for r in env.state_table:
                reduced_rho.append(partial_trace(r, [idx, idx+1], [2] * num_qubits))
            reduced_rho = np.array(reduced_rho)

            for obs1, obs2 in itertools.product(paulis, paulis): # 9
                rand_obs_string = op_str[:idx] + obs1 + obs2 + op_str[idx + 2:]
                # if rand_obs_string == op_str: continue
                rand_obs = [Pauli(obs1).to_matrix(), Pauli(obs2).to_matrix()]
                selected_qubits = [idx, idx + 1]
                obs = Pauli(rand_obs_string)
                obs_ret = np.array(rand_obs)
                exp_ideal = ideal_state.expectation_value(obs).real
                obs_kron = np.kron(rand_obs[0], rand_obs[1])
                meas_results_qp = np.trace(reduced_rho @ obs_kron, axis1=-2, axis2=-1).real

                for noise_scale in np.round(np.arange(0.02, 0.11, 0.01), 2): # 9
                    exp_noisy = env.simulate_noisy(noise_scale).expectation_value(obs).real
                    with tqdm(total=100, desc=f'Param: {param}, index: {idx}, observable: {obs1}{obs2}, '
                                             f'noise_scale: {noise_scale}') as pbar:
                        cnt = 0
                        while cnt < 100:
                            rnd_val = np.random.rand(*meas_results_qp.shape) * 2 - 1
                            exp_qp = np.sum(rnd_val * meas_results_qp)
                            if abs(exp_qp - exp_noisy) < 5e-4:
                                cnt += 1
                                pbar.update(1)
                                dataset.append([param, obs_ret, selected_qubits, noise_scale, rnd_val, round(exp_noisy, 6), round(exp_ideal, 6)])
    with open(args.out_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'Generation finished. File saved to {args.out_path}')


def gen_mitigation_data_pauli_distribution(args):
    # from ibmq_circuit_transformer import TransformToClifford
    from torchquantum import switch_little_big_endian_state
    dataset = []
    paulis = ['X', 'Y', 'Z']
    states = StateGenerator('cat', 6, shuffle=False).dataset

    for env_name in os.listdir(args.env_root):
        param = float(env_name.replace('.pkl', '').split('_')[-1])
        env_path = os.path.join(args.env_root, env_name)
        env = IBMQEnv.load(env_path)
        num_qubits = env.circuit.num_qubits
        op_str = 'I' * num_qubits
        ideal_state = env.simulate_ideal()
        exp_ideal = ideal_state.probabilities()
        encoder = env.circuit.copy()

        for state_idx, state in tqdm(enumerate(states)):
            base_circuit = QuantumCircuit(num_qubits)
            state = switch_little_big_endian_state(state)
            base_circuit.initialize(state, range(num_qubits))
            base_circuit = base_circuit.compose(encoder)
            for idx in range(num_qubits):
                for obs1 in paulis:
                    circuit = base_circuit.copy()
                    if obs1 == 'X':
                        circuit.h(num_qubits - 1 - idx)
                    elif obs1 == 'Y':
                        circuit.sdg(num_qubits - 1 - idx)
                        circuit.h(num_qubits - 1 - idx)
                    obs_ret = [np.eye(2) for _ in range(num_qubits)]
                    obs_ret[idx] = Pauli(obs1).to_matrix()
                    selected_qubits = [idx, idx + 1]
                    obs_ret = np.array(obs_ret)
                    for noise_scale in np.round(np.arange(0.05, 0.29, 0.01), 3): # 10
                        exp_noisy = env.simulate_noisy(noise_scale, circuit).probabilities([num_qubits - 1 - idx])
                        dataset.append([state_idx, obs_ret, selected_qubits, noise_scale, exp_noisy, exp_ideal])

    out_path = os.path.join(args.out_root, 'dataset_ae6l.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'Generation finished. File saved to {out_path}')


def gen_test_data_pauli(args):
    dataset = []
    paulis = ['X', 'Y', 'Z']

    for env_name in tqdm(os.listdir(args.env_test)):
        param = float(env_name.replace('.pkl', '').split('_')[-1])
        env_path = os.path.join(args.env_test, env_name)
        env = IBMQEnv.load(env_path)
        num_qubits = env.circuit.num_qubits
        op_str = 'I' * num_qubits
        ideal_state = env.simulate_ideal()
        for idx in range(num_qubits - 1): # 5
            for obs1, obs2 in itertools.product(paulis, paulis): # 16
                rand_obs_string = op_str[:idx] + obs1 + obs2 + op_str[idx + 2:]
                if rand_obs_string == op_str: continue
                obs_ret = [np.eye(2) for _ in range(num_qubits)]
                obs_ret[idx] = Pauli(obs1).to_matrix()
                obs_ret[idx + 1] = Pauli(obs2).to_matrix()
                selected_qubits = [idx, idx + 1]
                obs = Pauli(rand_obs_string)
                obs_ret = np.array(obs_ret)
                exp_ideal = ideal_state.expectation_value(obs).real
                exp_noisy = env.simulate_noisy(0.05).expectation_value(obs).real
                dataset.append([param, obs_ret, rand_obs_string, selected_qubits, 0.05, round(exp_noisy, 6), round(exp_ideal, 6)])
    out_test = os.path.join(args.out_root, args.out_test)
    with open(out_test, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'Generation finished. File saved to {out_test}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-root', default='../environments/noise_models/phase_damping/vqe_h_train_2l', type=str)
    parser.add_argument('--env-test', default='../environments/noise_models/phase_damping/vqe_h_train_2l', type=str)
    parser.add_argument('--out-root', default='../data_mitigate/phasedamp', type=str)
    parser.add_argument('--out-train', default='dataset_vqe2l.pkl', type=str)
    parser.add_argument('--out-test', default='testset_train.pkl', type=str)
    parser.add_argument('--num-ops', default=2, type=int)
    parser.add_argument('--num-data', default=20_000, type=int)  # 5000 for train
    args = parser.parse_args()
    
    if not os.path.exists(args.out_root):
        os.makedirs(args.out_root)
    gen_mitigation_data_pauli(args)
    gen_test_data_pauli(args)
    # gen_mitigation_data_pauli_distribution(args)
    # import subprocess
    # command = 'cd .. && python circuit.py --data-name vqe.pkl --train-name trainset_vqe.pkl --test-name testset_vqe.pkl --split'
    # subprocess.Popen(command, shell=True).wait()
