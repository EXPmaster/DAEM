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

    def fit_and_predict(self, circuit, observable):
        # folded_circuit = fold_gates_at_random(circuit, 0.5)
        noise_levels = np.arange(0.01, 0.1, 0.01)
        noisy_results = []
        for n in noise_levels:
            noise_model = NoiseModel()
            error_1 = noise.depolarizing_error(n, 1)  # single qubit gates
            error_2 = noise.depolarizing_error(n, 2)
            noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'i', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'])
            noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cy', 'cz', 'ch', 'crz', 'swap', 'cu1', 'cu3', 'rzz'])
            noisy_results.append(self.simulate_noisy(circuit, observable, noise_model))
        noisy_results = np.array(noisy_results)
        params = np.polyfit(noise_levels, noisy_results, deg=2)
        return params[-1]

    def simulate_ideal(self, circuit, observable):
        state_vector = Statevector(circuit)
        return state_vector.expectation_value(observable).real
        
    def simulate_noisy(self, circuit, observable, noise_model):
        circuit = circuit.copy()
        circuit.save_density_matrix()
        noisy_backend = AerSimulator(noise_model=noise_model)
        noisy_result = noisy_backend.run(transpile(circuit, noisy_backend)).result()
        density_matrix = noisy_result.data()['density_matrix']
        return density_matrix.expectation_value(observable).real


if __name__ == '__main__':
    with open('../environments/circuits_test_4l/vqe_0.4.pkl', 'rb') as f:
        circuit = pickle.load(f)

    noise_model = NoiseModel()
    error_1 = noise.depolarizing_error(0.01, 1)  # single qubit gates
    error_2 = noise.depolarizing_error(0.01, 2)
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'i', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cy', 'cz', 'ch', 'crz', 'swap', 'cu1', 'cu3', 'rzz'])

    obs = PauliOp(Pauli('IIZZ'))
    zne_model = ZNETrainer()
    mitigated_result = zne_model.fit_and_predict(circuit, obs)
    ideal_result = zne_model.simulate_ideal(circuit, obs)
    noisy_result = zne_model.simulate_noisy(circuit, obs, noise_model)
    print(mitigated_result, ideal_result, noisy_result)
    # noisy_data = zne_model.simulate_noisy(circuit)
    # noisy_data = np.array(noisy_data).reshape(-1, 1)
    # print(noisy_data.shape)
    # print(cdr_t.predict(noisy_data))