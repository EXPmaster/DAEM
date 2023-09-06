import os
import numpy as np
import pickle

from qiskit import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliOp

from mitiq import cdr
from mitiq.cdr import generate_training_circuits
from sklearn.linear_model import LinearRegression

from my_envs import IBMQEnv


class CDRTrainer:

    def __init__(self, env_root, pretrain_path=None):
        self.envs = {}
        if pretrain_path is None:
            for circuit_name in os.listdir(env_root):
                param = float(circuit_name.replace('.pkl', '').split('_')[-1])
                circuit_path = os.path.join(env_root, circuit_name)
                self.envs[param] = IBMQEnv(circ_path=circuit_path)
        else:
            with open(pretrain_path, 'rb') as f:
                self.reg = pickle.load(f)

    def fit(self, param, observable):
        circuit = self.envs[param].circuit
        circuit = transpile(circuit, basis_gates=['h', 's', 'rz', 'cx'])
        training_circuits = generate_training_circuits(
            circuit,
            num_training_circuits=150,
            fraction_non_clifford=0.0,
            method="uniform",
        )
        ideal_results = []
        noisy_results = []
        for circuit in training_circuits:
            circuit = transpile(circuit, basis_gates=['cx', 'u'])
            ideal_results.append(self._simulate_ideal(circuit, observable))
            noisy_results.append(self.envs[param].simulate_noisy(0.05, circuit).expectation_value(observable).real)
        ideal_results = np.array(ideal_results).reshape(-1, 1)
        noisy_results = np.array(noisy_results).reshape(-1, 1)
        self.reg = LinearRegression()
        self.reg.fit(noisy_results, ideal_results)
    
    def predict(self, noisy_data):
        return self.reg.predict(noisy_data)[0][0]

    def _simulate_ideal(self, circuit, observable):
        state_vector = Statevector(circuit)
        return state_vector.expectation_value(observable).real
        
    def _simulate_noisy(self, param, circuit, observable):
        tcircuit = circuit.copy()
        tcircuit.save_density_matrix()
        noisy_result = self.noisy_backend.run(transpile(tcircuit, self.noisy_backend)).result()
        density_matrix = noisy_result.data()['density_matrix']
        return density_matrix.expectation_value(observable).real


if __name__ == '__main__':
    import torch
    from TensornetSimulator import MPOCircuitSimulator, swaptest
    from cuquantum import cutensornet as cutn
    from tqdm import tqdm

    training_circuits = generate_training_circuits(
        swaptest(5),
        num_training_circuits=100,
        fraction_non_clifford=0.0,
        method="uniform",
    )
    handle = cutn.create()
    device = 'cuda'
    options = {'handle': handle}
    dtype = torch.complex64

    pauli_z = torch.tensor([[1., 0.], [0., -1.]], dtype=dtype, device=device)

    with open('../data_mitigate/data_st11q_pd/states.pkl', 'rb') as f:
        input_states = pickle.load(f)[:100]

    noise_level = 1 - np.exp(-2 * 0.05)
    ref_state = torch.tensor([[1., 0.], [0., 0.]], dtype=dtype, device=device)
    ideal_results = []
    noisy_results = []
    for circuit, states in tqdm(zip(training_circuits, input_states)):
        state1, state2 = states
        state1_t = torch.from_numpy(state1).to(dtype=dtype, device=device)
        state2_t = torch.from_numpy(state2).to(dtype=dtype, device=device)
        state1_t = torch.outer(state1_t, state1_t.conj())
        state2_t = torch.outer(state2_t, state2_t.conj())
        init_rho = torch.kron(ref_state, torch.kron(state1_t, state2_t))
        backend = MPOCircuitSimulator(
                    circuit,
                    'dephasing',
                    dtype=dtype,
                    device=device,
                    options=options
                )
        ideal_results.append(round(backend.run([pauli_z], (0,), init_rho=init_rho).real, 6))
        noisy_results.append(round(backend.run([pauli_z], (0,), noise_level=noise_level, init_rho=init_rho).real, 6))
    ideal_results = np.array(ideal_results).reshape(-1, 1)
    noisy_results = np.array(noisy_results).reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(noisy_results, ideal_results)
    with open('cdr_st11q.pkl', 'wb') as f:
        pickle.dump(reg, f)