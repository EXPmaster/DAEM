import pickle
import os
import pathos
import functools
import numpy as np
from tqdm import tqdm

import cirq
import cirq.ops as ops
from cirq.circuits import InsertStrategy
import torch

from qiskit import IBMQ, Aer, QuantumCircuit, transpile, execute
from qiskit.providers.aer import AerSimulator
import qiskit.opflow as opflow
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.quantum_info import DensityMatrix

from ibmq_circuit_transformer import TransformCircWithPr, TransformCircWithIndex, add_miti_gates_to_circuit
from utils import AverageMeter, ConfigDict, gen_rand_pauli
from circuit_lib import *


class IBMQEnv:

    def __init__(self, args=None, **kwargs):
        if 'pkl_file' in kwargs:
            pkl_file = kwargs['pkl_file']
            self.config = pkl_file['config']
            self.circuit = pkl_file['circuit']
            self.state_table = pkl_file['state_table']
            self.backend = pkl_file['noisy_backend']
            self.miti_circuit = pkl_file['miti_circuit']
        else:
            assert args is not None, 'Arguments must be provided.'
            self.config = ConfigDict(
                num_layers=args.num_layers,
                num_qubits=args.num_qubits,
            )
            if 'circ_path' in kwargs:
                self.circuit = self.load_circuit(kwargs['circ_path'])
            else:
                self.circuit = self._gen_new_circuit()

            # IBMQ.load_account()
            # provider = IBMQ.get_provider(
            #     hub='ibm-q',
            #     group='open',
            #     project='main'
            # )
            # backend = provider.get_backend(self.config.backend)
            # self.backend = AerSimulator.from_backend(backend)

            noise_model = NoiseModel()
            error_1 = noise.depolarizing_error(0.01, 1)  # single qubit gates
            error_2 = noise.depolarizing_error(0.01, 2)
            noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'i', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'])
            noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cy', 'cz', 'ch', 'crz', 'swap', 'cu1', 'cu3', 'rzz'])

            self.backend = AerSimulator(noise_model=noise_model)

            self.state_table = self.build_state_table()

        self.transformer = TransformCircWithPr(self.config.num_qubits)

    def _gen_new_circuit(self):
        # return swaptest().decompose().decompose()
        return random_circuit(self.config.num_qubits, self.config.num_layers, max_operands=2)

    def load_circuit(self, circ_path):
        with open(circ_path, 'rb') as f:
            circuit = pickle.load(f)
        return circuit

    def count_mitigate_gates(self, circuit=None):
        if circuit is None:
            circuit = self.miti_circuit
        num = 0
        for item in circuit:
            num += item[0].name == 'id'
        return num

    def build_state_table(self, shots_per_sim=3000):
        # print('Building look up table...')
        tsfm = TransformCircWithIndex()
        # print(self.circuit)
        mitigation_circuit = add_miti_gates_to_circuit(self.circuit)
        self.miti_circuit = mitigation_circuit.copy()
        mitigation_circuit.save_density_matrix()
        num_identity = self.count_mitigate_gates(mitigation_circuit)
        table_size = len(tsfm.basis_ops) ** num_identity
        # print(mitigation_circuit)
        # print(table_size)
        lut = []
        for i in tqdm(range(table_size)):
            circuit = tsfm(mitigation_circuit, i)
            t_circuit = transpile(circuit, self.backend)
            result_noisy = self.backend.run(t_circuit, shots=shots_per_sim).result()
            lut.append(result_noisy.data()['density_matrix'])

        return np.stack(lut)

    def _modify_circuit(self, p):
        return self.transformer(self.circuit, p)

    def step(self, obs_batch, p_batch, num_repeat=1000):
        # cum_p_batch = np.cumsum(p_batch, -1)
        # batch_size = len(p_batch)
        # u = np.random.rand(num_repeat, batch_size, p_batch.shape[1], 1)
        # choice_idx = (u < cum_p_batch[None]).argmax(-1).reshape(num_repeat, batch_size, -1)
        # idx_str = np.char.mod('%d', choice_idx)
        # indices = np.apply_along_axis(lambda arr: int(''.join(arr), 4), -1, idx_str)
        # selected_rho = self.state_table[indices]
        # observables = np.stack([functools.reduce(np.kron, item[::-1]) for item in obs_batch])
        # # meas_results = (selected_state_vec[:, :, None, :] @ obs_batch @ selected_state_vec[:, :, :, None]).squeeze(-1).real
        # # meas_results = []
        # # for i, idx in enumerate(indices):
        # #     rhos = self.state_table[idx]
        # #     batch_results = []
        # #     for i_rho, rho in enumerate(rhos):
        # #         batch_results.append([DensityMatrix(rho).expectation_value(obs_batch[i_rho][j], qargs=[j]).real for j in range(obs_batch.shape[1])])
        # #     meas_results.append(batch_results)
        # # 
        # # print(np.mean(meas_results, 0).shape)
        # meas_results = np.trace(selected_rho @ observables, axis1=-2, axis2=-1).real

        # Select p_batch of each permulation of gates, i.e., [III...I, XII...I, ..., XZZ...Z, YZZ...Z, ZZZ...Z]
        num_mitigate = p_batch.shape[1]
        select_indices = np.array(np.meshgrid(*[np.arange(4)[::-1] for _ in range(num_mitigate)])).reshape(num_mitigate, -1).T[::-1, ::-1]
        selected_p = np.array([item[np.arange(num_mitigate), select_indices] for item in p_batch])
        probabilities = np.prod(selected_p, axis=-1)
        observables = np.stack([functools.reduce(np.kron, item[::-1]) for item in obs_batch])
        if use_gpu:
            observables = torch.from_numpy(observables).to(device)
            meas_results = (self.state_table[None] @ observables[:, None, :, :]).diagonal(dim1=-2, dim2=-1).sum(dim=-1).real
            meas_results = meas_results.cpu().numpy()
        else:
            meas_results = np.trace(self.state_table[None] @ observables[:, None, :, :], axis1=-2, axis2=-1).real
        
        return np.sum(meas_results * probabilities, 1)

    def simulate_ideal(self, shots=1000):
        circuit = self.circuit.copy()
        circuit.save_statevector()
        backend = Aer.get_backend('aer_simulator')
        results = backend.run(transpile(circuit, backend), shots=shots).result()
        return results.get_statevector(circuit)

    def simulate_noisy(self, shots=10000):
        circuit = self.circuit.copy()
        circuit.save_density_matrix()
        t_circuit = transpile(circuit, self.backend)
        results = self.backend.run(t_circuit, shots=shots).result()
        return results.data()['density_matrix']

    def save(self, path):
        save_data = dict(
            config=self.config,
            circuit=self.circuit,
            state_table=self.state_table,
            noisy_backend=self.backend,
            miti_circuit=self.miti_circuit
        )
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f'Environment saved at {path}.')

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print('Environment loaded.')
        return cls(pkl_file=data)


def stable_softmax(x):
    maxval = x.max(-1, keepdims=True)
    x_exp = np.exp(x - maxval)
    return x_exp / x_exp.sum(-1, keepdims=True)


def gen_surrogate_dataset(env, data_num_per_run):
    num_qubits = env.circuit.num_qubits
    rand_val = np.random.rand(data_num_per_run, num_qubits, 4) * 2 - 1
    pr = stable_softmax(rand_val)
    observable = []
    obs_return = []
    position = []
    for _ in range(data_num_per_run):
        select_idx = np.random.randint(num_qubits - 1)
        cur_data = [gen_rand_pauli() if i in (select_idx, select_idx + 1) else np.eye(2) for i in range(num_qubits)]
        assert len(cur_data) == num_qubits
        observable.append(cur_data)
        obs_return.append([cur_data[select_idx], cur_data[select_idx + 1]])
        position.append([select_idx, select_idx + 1])

    # observable = np.stack([[gen_rand_obs() for _ in range(num_qubits)] for _ in range(data_num_per_run)])
    # observable = np.stack([cirq.unitary(cirq.Z)] * args.data_num)
    
    results = env.step(observable, pr, num_repeat=2000)
    return pr, obs_return, position, results


def main(args):
    env_root = '../environments/vqe_envs'
    data_num_per_run = 10
    pr = []
    observable = []
    position = []
    results = []
    params = []
    for env_name in os.listdir(env_root):
        param = float(env_name.replace('.pkl', '').split('_')[-1])
        env_path = os.path.join(env_root, env_name)
        env = IBMQEnv.load(env_path)
        if use_gpu:
            env.state_table = torch.from_numpy(env.state_table).to(device)
        for _ in tqdm(range(args.data_num // data_num_per_run)):
            p, o, i, r = gen_surrogate_dataset(env, data_num_per_run)
            pr.append(p)
            observable.append(o)
            position.append(i)
            results.append(r)
            params.append([param] * data_num_per_run)

    # print(np.concatenate(params).shape)
    # print(np.concatenate(pr).shape)
    # print(np.concatenate(observable).shape)
    # print(np.concatenate(results).shape)
    save_data = dict(
        vqe_param=np.concatenate(params),
        probability=np.concatenate(pr),
        observable=np.concatenate(observable),
        position=np.concatenate(position),
        meas_result=np.concatenate(results)
    )
    with open(args.save_path, 'wb') as f:
        pickle.dump(save_data, f)


def build_env_vqe(args):
    vqe_circuit_path = '../environments/circuits'
    save_root = '../environments/vqe_envs'
    for circuit_name in os.listdir(vqe_circuit_path):
        circuit_path = os.path.join(vqe_circuit_path, circuit_name)
        env = IBMQEnv(args, circ_path=circuit_path)
        env.save(os.path.join(save_root, circuit_name))


if __name__ == '__main__':
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    from collections import namedtuple
    import warnings
    warnings.filterwarnings('ignore')

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda') if use_gpu else 'cpu'
    print(device)
    
    ArgsClass = namedtuple('args', ['num_layers', 'num_qubits', 'backend', 'save_path', 'data_num'])
    args = ArgsClass(3, 4, 'ibmq_santiago', '../data_surrogate/env_vqe_data.pkl', 3000)
    # env = IBMQEnv(args)
    # env.save(args.env_path)
    # print(env.circuit)
    # build_env_vqe(args)
    main(args)
    
