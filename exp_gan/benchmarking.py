import pickle
import random
from functools import partial
import argparse
from cirq import DensityMatrixSimulator, depolarize

from mitiq.interface.mitiq_qiskit import qiskit_utils
from mitiq import zne, cdr, pec
from mitiq.interface import convert_to_mitiq
from mitiq.pec.representations.depolarizing import represent_operations_in_circuit_with_global_depolarizing_noise


import qiskit
from qiskit import IBMQ, Aer, QuantumCircuit, transpile, execute
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
import numpy as np

from my_envs import IBMQEnv


def benchmark_zne():
    circuit = env.circuit.copy()
    # circuit.measure(0, 0)
    # circuit = transpile(circuit, basis_gates=["u1", "u2", "u3", "cx"])
    circuit = transpile(circuit, basis_gates=['x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg', "rx", "ry", "rz", "cx", 'cy', 'cz', 'ch', "u1", "u2", "u3"])
    scale_factors = np.arange(1, 3, 0.5)  # [1., 1.5, 2., 2.5, 3.]
    folded_circuits = [
        zne.scaling.fold_gates_at_random(circuit, scale)
        for scale in scale_factors
    ]
    shots = 20000
    array = []
    array_nomiti = []
    # for obs, noisy, ideal in dataset[:5]:
    #     # obs = np.diag([1., -1.])
    #     obs = np.kron(np.eye(2**3), obs)
    #     # expectation_ideal = ideal_state.expectation_value(obs).real
    #     expectation_ideal = sim(circuit, obs, shots)
    #     expectation_noisy = executor(circuit, obs, shots)
    #     expectation_values = [
    #         qiskit_utils.execute_with_shots_and_noise(
    #             circ,
    #             obs,
    #             noise_model,
    #             shots,
    #             seed=0
    #         ) for circ in folded_circuits
    #     ]
    #     # zero_noise_value = zne.ExpFactory.extrapolate(scale_factors, expectation_values, asymptote=0.5)
    #     zero_noise_value = zne.LinearFactory.extrapolate(scale_factors, expectation_values)
    #     array.append(abs(expectation_ideal - zero_noise_value))
    #     array_nomiti.append(abs(expectation_ideal - expectation_noisy))
    #     # print(array)
    #     # print(array_nomiti)
    #     # assert False
    for obs, noisy, ideal in dataset:
        # obs = np.diag([1., -1.])
        obs = np.kron(obs, np.eye(2**3))
        # expectation_ideal = ideal_state.expectation_value(obs).real
        expectation_ideal = execute_pec(circuit, obs, noise_level=0.0)
        expectation_noisy = execute_pec(circuit, obs)

        zero_noise_value = zne.execute_with_zne(circuit, partial(execute_pec, obs=obs))
        array.append(abs(expectation_ideal - zero_noise_value))
        array_nomiti.append(abs(expectation_ideal - expectation_noisy))

    original_deviation = round(np.mean(array_nomiti), 6)
    mitigated_deviation = round(np.mean(array), 6)
    print('no mitigation: ', original_deviation)
    print('zne mitigation: ', mitigated_deviation)
    print('mitigation ratio: ', original_deviation / mitigated_deviation)


def benchmark_cdr():
    ideal_state = env.simulate_ideal()
    circuit = env.circuit.copy()
    circuit = transpile(circuit, basis_gates=["rx", "ry", "rz", "cx"])
    # circuit.measure(0, 0)
    # print(circuit)
    shots = 10000

    array = []
    array_nomiti = []
    # for obs, _, _ in dataset[:5]:
    #     # obs = np.diag([1., -1.])
    #     obs = np.kron(np.eye(2**3), obs)
    #     # print(executor(circuit, obs, shots=shots))
    #     expectation_noisy = executor(circuit, obs, shots=shots)
    #     expectation_ideal = sim(circuit, obs, shots=shots)  # ideal_state.expectation_value(obs).real
    #     mitigated_measurement = cdr.execute_with_cdr(
    #         circuit,
    #         partial(executor, obs=obs, shots=shots),
    #         num_training_circuits=30,
    #         simulator=partial(sim, obs=obs, shots=shots),
    #         seed=0,
    #         # fraction_non_clifford=0.6
    #         # fit_function=cdr.linear_fit_function_no_intercept
    #     ).real
    #     array.append(abs(expectation_ideal - mitigated_measurement))
    #     array_nomiti.append(abs(expectation_noisy - expectation_ideal))

    for obs, noisy, ideal in dataset:
        # obs = np.diag([1., -1.])
        obs = np.kron(obs, np.eye(2**3))
        # print(executor(circuit, obs, shots=shots))
        expectation_ideal = execute_pec(circuit, obs, noise_level=0.0)
        expectation_noisy = execute_pec(circuit, obs)
        mitigated_measurement = cdr.execute_with_cdr(
            circuit,
            partial(execute_pec, obs=obs),
            # num_training_circuits=30,
            simulator=partial(execute_pec, obs=obs, noise_level=0.0),
            seed=0,
            # fraction_non_clifford=0.6
            # fit_function=cdr.linear_fit_function_no_intercept
        ).real
        array.append(abs(expectation_ideal - mitigated_measurement))
        array_nomiti.append(abs(expectation_noisy - expectation_ideal))
    # print(array)
    # print(array_nomiti)
    original_deviation = round(np.mean(array_nomiti), 6)
    mitigated_deviation = round(np.mean(array), 6)
    print('no mitigation: ', original_deviation)
    print('cdr mitigation: ', mitigated_deviation)
    print('mitigation ratio: ', original_deviation / mitigated_deviation)


def executor(circ, obs, shots):
    return qiskit_utils.execute_with_shots_and_noise(circ, obs, noise_model, shots)


def sim(circ, obs, shots):
    return qiskit_utils.execute_with_shots(circ, obs, shots)


def benchmark_pec():
    circuit = env.circuit.copy()
    circuit = transpile(circuit, basis_gates=["rx", "ry", "rz", "cx"])
    noise_level = 0.01
    reps = represent_operations_in_circuit_with_global_depolarizing_noise(circuit, noise_level)
    array = []
    array_nomiti = []
    for obs, exp_noisy, exp_ideal in dataset[:1]:
        # obs = np.diag([1., -1.])
        # obs = np.kron(np.eye(2**3), obs)
        obs = np.kron(obs, np.eye(2**3))
        # expectation_ideal = ideal_state.expectation_value(obs).real
        expectation_ideal = execute_pec(circuit, obs, noise_level=0.0)
        expectation_noisy = execute_pec(circuit, obs)
        pec_value = pec.execute_with_pec(circuit, partial(execute_pec, obs=obs), representations=reps)
        array.append(abs(expectation_ideal - pec_value))
        array_nomiti.append(abs(expectation_noisy - expectation_ideal))
        
    print(array)
    print(array_nomiti)
    original_deviation = round(np.mean(array_nomiti), 6)
    mitigated_deviation = round(np.mean(array), 6)
    print('no mitigation: ', original_deviation)
    print('pec mitigation: ', mitigated_deviation)
    print('mitigation ratio: ', original_deviation / mitigated_deviation)
    

def execute_pec(circuit, obs, noise_level=0.01):
    # Replace with code based on your frontend and backend.
    mitiq_circuit, _ = convert_to_mitiq(circuit)
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
    env = IBMQEnv.load(args.env_path)

    noise_model = NoiseModel()
    error_1 = depolarizing_error(0.01, 1)  # single qubit gates
    error_2 = depolarizing_error(0.01, 2)
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'i', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cy', 'cz', 'ch', 'crz', 'swap', 'cu1', 'cu3', 'rzz'])

    with open(args.test_data, 'rb') as f:
        dataset = pickle.load(f)

    print('Benchmarking method: {}...'.format(args.method))
    if args.method == 'zne':
        benchmark_zne()
    elif args.method == 'cdr':
        benchmark_cdr()
    elif args.method == 'pec':
        benchmark_pec()
    else:
        print('No such method [{}]!'.format(args.method))