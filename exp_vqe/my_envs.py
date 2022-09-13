import pickle
import os
import pathos
import numpy as np
from tqdm import tqdm

import cirq
import cirq.ops as ops
from cirq.circuits import InsertStrategy

from qiskit import IBMQ, Aer, QuantumCircuit, transpile, execute
from qiskit.providers.aer import AerSimulator
import qiskit.opflow as opflow
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise

from ibmq_circuit_transformer import TransformCircWithPr, TransformCircWithIndex, add_miti_gates_to_circuit
from utils import AverageMeter, ConfigDict
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
            # counts = result_noisy.get_counts()
            # lut.append(self.state_vector(counts, shots=shots_per_sim))
            lut.append(result_noisy.data()['density_matrix'])

        return np.stack(lut)

    @staticmethod
    def state_vector(counts, shots):
        probs = {'0': 0, '1': 0}
        for output in ['0', '1']:
            if output in counts:
                probs[output] = counts[output] / shots
            else:
                probs[output] = 0
        return np.sqrt(np.array([probs['0'], probs['1']]))

    def _modify_circuit(self, p):
        return self.transformer(self.circuit, p)

    def step(self, obs_batch, p_batch, nums=1000):
        # if len(p_batch.shape) == 4:
        #     return np.array([self._get_mean_val(obs, p, nums) for obs, p in zip(obs_batch, p_batch)])
        # else:
        #     return self._get_mean_val(obs_batch, p_batch, nums)
        cum_p_batch = np.cumsum(p_batch, -1)
        batch_size = len(p_batch)
        u = np.random.rand(nums, batch_size, p_batch.shape[1], 1)
        choice_idx = (u < cum_p_batch[None]).argmax(-1).reshape(nums, batch_size, -1)
        idx_str = np.char.mod('%d', choice_idx)
        idx = np.apply_along_axis(_map_fn, -1, idx_str)
        # selected_state_vec = self.state_table[idx]
        # meas_results = (selected_state_vec[:, :, None, :] @ obs_batch @ selected_state_vec[:, :, :, None]).squeeze(-1).real
        selected_rho = self.state_table[idx]
        meas_results = np.trace(selected_rho @ np.kron(np.eye(2 ** 3)[None], obs_batch), axis1=-2, axis2=-1).real
        return np.mean(meas_results, 0)[:, None]

    def simulate_ideal(self, shots=1000):
        circuit = self.circuit.copy()
        circuit.save_statevector()
        backend = Aer.get_backend('aer_simulator')
        results = backend.run(transpile(circuit, backend), shots=shots).result()
        # counts = results.get_counts()
        # return self.state_vector(counts, shots)
        return results.get_statevector(circuit)

    def simulate_noisy(self, shots=10000):
        circuit = self.circuit.copy()
        circuit.save_density_matrix()
        t_circuit = transpile(circuit, self.backend)
        results = self.backend.run(t_circuit, shots=shots).result()
        return results.data()['density_matrix']

    def _apply_step(self, obs, p):
        circuit = self._modify_circuit(p)
        t_circuit = transpile(circuit, self.backend)
        result_noisy = self.backend.run(t_circuit, shots=100).result()
        counts = result_noisy.get_counts()
        return self.measure_obs(counts, obs, shots=1)
        
    def _get_mean_val(self, observable, p, nums):
        avg = AverageMeter()
        pool = pathos.multiprocessing.Pool(processes=16)
        data_queue = []
        for _ in range(nums):
            data_queue.append(pool.apply_async(self._apply_step, args=(observable, p)))
        pool.close()
        pool.join()
        sum_val = sum([item.get() for item in data_queue])
        return sum_val / nums

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


def _map_fn(arr):
    string = ''.join(arr)
    return int(string, 4)


def gen_surrogate_dataset(args, env_path):
    env = IBMQEnv.load(env_path)
    print('Start.')
    rand_val = np.random.randn(args.data_num, env.count_mitigate_gates(), 4)
    pr = stable_softmax(rand_val)
    observable = np.stack([cirq.unitary(cirq.Z)] * args.data_num)
    
    results = env.step(observable, pr, nums=1000)
    print('Done.')
    return pr, observable, results


def main(args):
    env_root = '../environments/vqe_envs'
    pr = []
    observable = []
    results = []
    params = []
    for env_name in os.listdir(env_root):
        param = env_name.replace('.pkl', '').split('_')[-1]
        env_path = os.path.join(env_root, env_name)
        p, o, r = gen_surrogate_dataset(args, env_path)
        pr.append(p)
        observable.append(o)
        results.append(r)
        params.append([param] * args.data_num)

    save_data = dict(
        vqe_param=np.concatenate(params),
        probability=np.concatenate(pr),
        observable=np.concatenate(observable),
        meas_result=np.concatenate(results)
    )
    with open('../data_surrogate/env_vqe_data.pkl', 'wb') as f:
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
    
    ArgsClass = namedtuple('args', ['num_layers', 'num_qubits', 'backend', 'env_path', 'data_num'])
    args = ArgsClass(3, 4, 'ibmq_santiago', '../environments/swaptest.pkl', 1000)
    # env = IBMQEnv(args)
    # env.save(args.env_path)
    # print(env.circuit)
    # build_env_vqe(args)
    main(args)
    