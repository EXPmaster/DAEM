import numpy as np
import pickle
import functools
from qiskit import transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliOp

from mitiq import cdr
from mitiq.cdr import generate_training_circuits
from mitiq.zne.scaling import fold_gates_at_random


class ZNETrainer:

    def __init__(self):
        pass

    def fit_and_predict(self, circuit, observable, backends):
        # folded_circuit = fold_gates_at_random(circuit, 0.5)
        noise_levels = np.round(np.arange(0.05, 0.29, 0.01), 3)
        noisy_results = []
        # func = lambda x: 0.2 * np.sqrt(1 - np.exp(-(1 - np.cos(25 * np.arctan(x))) / (1 + x ** 2) ** (25 / 2)))
        # func = lambda x: 0.2 * (1 - np.exp(-(1 - np.cos(15 * np.arctan(x))) / (1 + x ** 2) ** (15 / 2)))
        for n in noise_levels:
            noise_model = backends[n]
            # error_1 = noise.depolarizing_error(n, 1)  # single qubit gates
            # # error_2 = noise.depolarizing_error(n, 2)
            # # error_1 = noise.amplitude_damping_error(n)
            # # error_2 = error_1.tensor(error_1)
            # noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'i', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'])
            # noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cy', 'cz', 'ch', 'crz', 'swap', 'cu1', 'cu3', 'rzz'])
            # noise_model = NoiseModel()
            # for j in range(12):
            #     error_1 = noise.amplitude_damping_error(func(n))
            #     noise_model.add_all_qubit_quantum_error(error_1, f"parameter_{j}", range(circuit.num_qubits))
            #     noise_model.add_basis_gates(['u'])
            noisy_results.append(self.simulate_noisy(circuit, observable, noise_model))
        noisy_results = np.array(noisy_results)
        params = np.polyfit(noise_levels, noisy_results, deg=2)
        return params  # [-1]

    def simulate_ideal(self, circuit, observable):
        state_vector = Statevector(circuit)
        return state_vector.expectation_value(observable).real
        
    def simulate_noisy(self, circuit, observable, noise_model):
        circuit = circuit.copy()
        circuit.save_density_matrix()
        noisy_backend = noise_model  # AerSimulator(noise_model=noise_model)
        noisy_result = noisy_backend.run(transpile(circuit, noisy_backend, optimization_level=0)).result()
        density_matrix = noisy_result.data()['density_matrix']
        return density_matrix.expectation_value(observable).real


if __name__ == '__main__':
    with open('../environments/circuits_test_4l/vqe_0.4.pkl', 'rb') as f:
        circuit = pickle.load(f)

    noise_model = NoiseModel()
    error_1 = noise.depolarizing_error(0.02, 1)  # single qubit gates
    error_2 = noise.depolarizing_error(0.02, 2)
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'i', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cy', 'cz', 'ch', 'crz', 'swap', 'cu1', 'cu3', 'rzz'])

    obs = PauliOp(Pauli('IIZY'))
    zne_model = ZNETrainer()
    mitigated_result = zne_model.fit_and_predict(circuit, obs)
    ideal_result = zne_model.simulate_ideal(circuit, obs)
    noisy_result = zne_model.simulate_noisy(circuit, obs, noise_model)
    print(mitigated_result, ideal_result, noisy_result)
    print(abs(mitigated_result - ideal_result))
    # noisy_data = zne_model.simulate_noisy(circuit)
    # noisy_data = np.array(noisy_data).reshape(-1, 1)
    # print(noisy_data.shape)
    # print(cdr_t.predict(noisy_data))