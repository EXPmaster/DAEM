import numpy as np
import pickle
import functools
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliOp
from qiskit.quantum_info import DensityMatrix

from mitiq import cdr
from mitiq.cdr import generate_training_circuits
from mitiq.zne.scaling import fold_gates_at_random

from circuit_parser import CircuitParser


class ZNETrainer:

    def __init__(self, noisy_env=None):
        self.env = noisy_env

    def fit_and_predict(self, exps):
        # folded_circuit = fold_gates_at_random(circuit, 0.5)
        # noise_levels = np.round(np.arange(0.05, 0.29, 0.02), 3)
        noise_levels = np.linspace(0.005, 0.1, 4)
        noisy_results = exps
        # for n in noise_levels:
        #     noisy_results.append(self.simulate_noisy(observable, n))
        noisy_results = np.array(noisy_results)
        params = np.polyfit(noise_levels, noisy_results, deg=2)
        return params  # [-1]

    def simulate_ideal(self, observable):
        state_vector = self.env.simulate_ideal()
        return state_vector.expectation_value(observable).real
        
    def simulate_noisy(self, observable, noise_scale):
        density_matrix = self.env.simulate_noisy(noise_scale)
        return density_matrix.expectation_value(observable).real

    def plot_fig(self, exps):
        scales = np.round(np.arange(0.05, 0.29, 0.02), 3)
        fig = plt.figure()
        plt.plot(scales, exps)
        # plt.xscale('log')
        plt.legend(['simulation'])
        plt.xlabel('Noise scale')
        plt.ylabel('Expectation of observable')
        plt.savefig('../imgs/results_diff_scales_new_pd.png')


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