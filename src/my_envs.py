import pickle
import os
import random
import functools
import numpy as np
from tqdm import tqdm

import torch

from qiskit import IBMQ, Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, execute
from qiskit.providers.aer import AerSimulator
import qiskit.opflow as opflow
# from qiskit.providers.aer.noise import NoiseModel
# import qiskit.providers.aer.noise as noise
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info.operators import Pauli

from utils import AverageMeter, ConfigDict, gen_rand_pauli
from hamiltonian_simulator import NonMarkovianSimulator, DepolarizeSimulator, DephaseSimulator, AmpdampSimulator
from circuit_parser import CircuitParser


class IBMQEnv:

    def __init__(self, args=None, **kwargs):
        if 'pkl_file' in kwargs:
            pkl_file = kwargs['pkl_file']
            self.circuit = pkl_file['circuit']
            # self.state_table = pkl_file['state_table']
            self.backends = pkl_file['noisy_backend']
        else:
            if 'circ_path' in kwargs:
                self.circuit = self.load_circuit(kwargs['circ_path'])
            else:
                raise ValueError('No circuit path provided.')

            count = -1
            angles = []
            # self.params = []
            for gate in self.circuit:
                if gate.operation.name in ['u', 'u3']:
                    count += 1
                    gate.operation.label = f'parameter_{count}'
                    # gate.operation.name = f'parameter_{count}'
                    angles.append(sum([abs(float(x)) for x in gate.operation.params]))
                    # self.params.append([float(x) for x in gate.operation.params])
                # print(gate)
            self.backends = {}
            # func = lambda x: 0.2 * np.sqrt(1 - np.exp(-(1 - np.cos(25 * np.arctan(x))) / (1 + x ** 2) ** (25 / 2)))
            # func = lambda x: (1 - np.exp(-(1 - (np.cos(15 * np.arctan(x))) / (1 + x ** 2) ** (15 / 2))))
            func = lambda x: 1 - np.exp(-0.6 * (1 - (np.cos(5 * np.arctan(3 * x))) / (1 + (3 * x) ** 2) ** (5 / 2)))
            # angles = [np.exp(-1. / x) for x in angles]
            for i in np.round(np.arange(0, 0.3, 0.001), 3):
                self.backends[i] = DephaseSimulator(i)  # NonMarkovianSimulator(i)  # DepolarizeSimulator(i)

            # self.state_table = self.build_state_table()

    def load_circuit(self, circ_path):
        with open(circ_path, 'rb') as f:
            circuit = pickle.load(f)
        return circuit

    def count_mitigate_gates(self, circuit=None):
        if circuit is None:
            circuit = self.miti_circuit
        num = 0
        for item in circuit:
            num += item[0].label == 'miti'
        return num

    def simulate_ideal(self, init_state=None, shots=1000):
        execute_circ = QuantumCircuit(self.circuit.num_qubits)
        if init_state is not None:
            execute_circ.initialize(init_state, range(self.circuit.num_qubits))
        execute_circ.append(self.circuit, range(self.circuit.num_qubits))
        execute_circ.save_statevector()
        backend = Aer.get_backend('aer_simulator')
        results = backend.run(transpile(execute_circ, backend, optimization_level=0), shots=shots).result()
        return results.get_statevector(execute_circ)

    def simulate_noisy(self, noise_scale, circuit=None, init_rho=None, train=False, shots=3000):
        backend = self.backends[noise_scale]
        if circuit is None:
            cur_circuit = self.circuit.copy()
        else:
            cur_circuit = circuit

        if isinstance(backend, AerSimulator):
            cur_circuit.save_density_matrix()
            t_circuit = transpile(cur_circuit, backend, optimization_level=0)
            results = backend.run(t_circuit, shots=shots).result()
            return results.data()['density_matrix']
        else:
            final_rho = backend.run(cur_circuit, init_rho=init_rho, simulate_identity=train)
            return DensityMatrix(final_rho)

    def save(self, path):
        save_data = dict(
            circuit=self.circuit,
            # state_table=self.state_table,
            noisy_backend=self.backends
        )
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f'Environment saved at {path}.')

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # print('Environment loaded.')
        return cls(pkl_file=data)


def build_env_vqe(vqe_circuit_path, save_root):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for circuit_name in os.listdir(vqe_circuit_path):
        circuit_path = os.path.join(vqe_circuit_path, circuit_name)
        env = IBMQEnv(circ_path=circuit_path)
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
    
    vqe_circuit_path = '../environments/circuits/vqe_2l'
    save_root = '../environments/noise_models/phase_damping/vqe_h_train_2l'
    build_env_vqe(vqe_circuit_path, save_root)

    # vqe_circuit_path = '../environments/circuits_test_4l'
    # # save_root = '../environments/amp_damping/vqe_envs_test_4l'
    # build_env_vqe(vqe_circuit_path, save_root)
    