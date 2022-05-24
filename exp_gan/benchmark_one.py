import pickle
import random
from functools import partial
import argparse
from cirq import DensityMatrixSimulator, depolarize

from mitiq.interface.mitiq_qiskit import qiskit_utils
from mitiq import zne, cdr, pec
from mitiq.interface import convert_to_mitiq
from mitiq.pec.representations.depolarizing import represent_operations_in_circuit_with_local_depolarizing_noise


import qiskit
from qiskit import IBMQ, Aer, QuantumCircuit, transpile, execute
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
import numpy as np

from my_envs import IBMQEnv
from circuit_lib import DQCp, swaptest


def benchmark_zne(circuit):
    # circuit.measure(0, 0)
    # circuit = transpile(circuit, basis_gates=["u1", "u2", "u3", "cx"])
    # circuit = transpile(circuit, basis_gates=['x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg', "rx", "ry", "rz", "cx", 'cy', 'cz', 'ch', "u1", "u2", "u3"])
    circuit,_ = convert_to_mitiq(circuit)

    obs = np.diag([1., -1.])
    obs = np.kron(obs, np.eye(2**4))

    expectation_ideal = execute_pec(circuit, obs, noise_level=0.0)
    print('ideal value: ', expectation_ideal)
    expectation_noisy = execute_pec(circuit, obs)

    zero_noise_value = zne.execute_with_zne(circuit, partial(execute_pec, obs=obs))

    original_deviation = abs(expectation_ideal - expectation_noisy)
    mitigated_deviation = abs(expectation_ideal - zero_noise_value)
    print('no mitigation: ', original_deviation)
    print('zne mitigation: ', mitigated_deviation)
    print('mitigation ratio: ', original_deviation / mitigated_deviation)


def benchmark_cdr(circuit):
    # circuit = transpile(circuit, basis_gates=["rx", "ry", "rz", "cx"])
    circuit,_ = convert_to_mitiq(circuit)
    # circuit.measure(0, 0)
    # print(circuit)

    obs = np.diag([1., -1.])
    obs = np.kron(obs, np.eye(2**4))
    expectation_ideal = execute_pec(circuit, obs, noise_level=0.0)
    print('ideal value', expectation_ideal)
    expectation_noisy = execute_pec(circuit, obs)
    mitigated_measurement = cdr.execute_with_cdr(
        circuit,
        partial(execute_pec, obs=obs),
        num_training_circuits=30,
        simulator=partial(execute_pec, obs=obs, noise_level=0.0),
        seed=0,
        # fraction_non_clifford=0.6
        # fit_function=cdr.linear_fit_function_no_intercept
    ).real

    original_deviation = abs(expectation_ideal - expectation_noisy)
    mitigated_deviation = abs(expectation_ideal - mitigated_measurement)
    print('no mitigation: ', original_deviation)
    print('cdr mitigation: ', mitigated_deviation)
    print('mitigation ratio: ', original_deviation / mitigated_deviation)


def executor(circ, obs, shots):
    return qiskit_utils.execute_with_shots_and_noise(circ, obs, noise_model, shots)


def sim(circ, obs, shots):
    return qiskit_utils.execute_with_shots(circ, obs, shots)


def benchmark_pec(circuit):
    # circuit = transpile(circuit, basis_gates=["rx", "ry", "rz", "cx"])
    # circuit = transpile(circuit, basis_gates=['x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg', "rx", "ry", "rz", "cx", 'cy', 'cz', 'ch', "u1", "u2", "u3"])
    circuit,_ = convert_to_mitiq(circuit)
    noise_level = 0.01
    reps = represent_operations_in_circuit_with_local_depolarizing_noise(circuit, noise_level)

    obs = np.diag([1., -1.])
    obs = np.kron(obs, np.eye(2**4))
    # expectation_ideal = ideal_state.expectation_value(obs).real
    expectation_ideal = execute_pec(circuit, obs, noise_level=0.0)
    print('ideal value', expectation_ideal)
    expectation_noisy = execute_pec(circuit, obs)
    pec_value = pec.execute_with_pec(circuit, partial(execute_pec, obs=obs), representations=reps, num_samples=100000)
    
    original_deviation = abs(expectation_ideal - expectation_noisy)
    mitigated_deviation = abs(expectation_ideal - pec_value)
    print('no mitigation: ', original_deviation)
    print('pec mitigation: ', mitigated_deviation)
    print('mitigation ratio: ', original_deviation / mitigated_deviation)
    

def execute_pec(mitiq_circuit, obs, noise_level=0.01):
    # Replace with code based on your frontend and backend.
    # mitiq_circuit, _ = convert_to_mitiq(circuit)
    # We add a simple noise model to simulate a noisy backend.
    noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return np.trace(obs @ rho).real


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-path', default='../environments/ibmq_random.pkl', type=str)
    parser.add_argument('--method', default='zne', type=str, help='[zne, cdr, pec]')
    parser.add_argument('--test-data', default='../data_mitigate/testset_randomcirc.pkl', type=str)
    args = parser.parse_args()
    # IBMQ.load_account()
    # provider = IBMQ.get_provider(
    #     hub='ibm-q',
    #     group='open',
    #     project='main'
    # )
    # backend = provider.get_backend(env.config.backend)
    # noise_model = NoiseModel.from_backend(backend, readout_error=False, thermal_relaxation=False)
    # env = IBMQEnv.load(args.env_path)
    circuit = swaptest().decompose().decompose()

    noise_model = NoiseModel()
    error_1 = depolarizing_error(0.01, 1)  # single qubit gates
    error_2 = depolarizing_error(0.01, 2)
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'i', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cy', 'cz', 'ch', 'crz', 'swap', 'cu1', 'cu3', 'rzz'])

    with open(args.test_data, 'rb') as f:
        dataset = pickle.load(f)

    print('Benchmarking method: {}...'.format(args.method))
    if args.method == 'zne':
        benchmark_zne(circuit)
    elif args.method == 'cdr':
        benchmark_cdr(circuit)
    elif args.method == 'pec':
        benchmark_pec(circuit)
    else:
        print('No such method [{}]!'.format(args.method))