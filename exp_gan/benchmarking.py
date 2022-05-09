import pickle
from functools import partial
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
import torch

from my_envs import IBMQEnv


def benchmark_zne():
    # env.gen_new_circuit_without_id()
    ideal_state = env.simulate_ideal()
    circuit = env.circuit.copy()
    # circuit.measure(0, 0)
    circuit = transpile(circuit, basis_gates=["u1", "u2", "u3", "cx"])
    
    scale_factors = np.arange(1, 4, 0.5)  # [1., 1.5, 2., 2.5, 3.]
    folded_circuits = [
        zne.scaling.fold_gates_at_random(circuit, scale)
        for scale in scale_factors
    ]
    shots = 10000
    # backend_ideal = Aer.get_backend('aer_simulator')
    # results = backend_ideal.run(circuit, shots=shots).result()
    # counts_ideal = results.get_counts()

    # backend_simulator = AerSimulator.from_backend(backend)
    # result = qiskit.execute(
    #     experiments=folded_circuits,
    #     backend=backend_simulator,
    #     noise_model=noise_model,
    #     basis_gates=noise_model.basis_gates,
    #     optimization_level=0,  # Important to preserve folded gates.
    #     shots=shots,
    # ).result()
    # counts = [result.get_counts(i) for i in range(len(scale_factors))]
    array = []
    for obs, _, _ in dataset:
        # obs = np.diag([1., -1.])
        obs = np.kron(np.eye(2**3), obs)
        expectation_ideal = ideal_state.expectation_value(obs).real
        # expectation_ideal = sim(circuit, obs, shots)
        print('sim_noisy: ', executor(circuit, obs, shots))
        expectation_values = [
            qiskit_utils.execute_with_shots_and_noise(
                circ,
                obs,
                noise_model,
                shots
            ) for circ in folded_circuits
        ]
        # expectation_ideal = env.measure_obs(counts_ideal, obs, shots)
        # expectation_values = [env.measure_obs(c, obs, shots).real for c in counts]
        # zero_noise_value = zne.ExpFactory.extrapolate(scale_factors, expectation_values, asymptote=0.5)
        zero_noise_value = zne.LinearFactory.extrapolate(scale_factors, expectation_values)
        array.append(abs(expectation_ideal - zero_noise_value))
        print(expectation_ideal)
        print(zero_noise_value)
        assert False
    print(round(np.mean(array), 6))


def benchmark_cdr():
    ideal_state = env.simulate_ideal()
    circuit = env.circuit.copy()
    circuit = transpile(circuit, basis_gates=["rx", "ry", "rz", "cx"])
    # circuit.measure(0, 0)
    # print(circuit)
    shots = 10000

    array = []
    for obs, _, _ in dataset[:100]:
        # obs = np.diag([1., -1.])
        obs = np.kron(np.eye(2**3), obs)
        # print(executor(circuit, obs, shots=shots))
        expectation_ideal = ideal_state.expectation_value(obs).real
        mitigated_measurement = cdr.execute_with_cdr(
            circuit,
            partial(executor, obs=obs, shots=shots),
            num_training_circuits=30,
            simulator=partial(sim, obs=obs, shots=shots),
            seed=0,
            # fraction_non_clifford=0.6
            # fit_function=cdr.linear_fit_function_no_intercept
        ).real
        array.append(abs(expectation_ideal - mitigated_measurement))
        # print(expectation_ideal)
        # print(mitigated_measurement)
        # assert False
    print(round(np.mean(array), 6))


def executor(circ, obs, shots):
    return qiskit_utils.execute_with_shots_and_noise(circ, obs, noise_model, shots)


def sim(circ, obs, shots):
    return qiskit_utils.execute_with_shots(circ, obs, shots)


def benchmark_pec():
    ideal_state = env.simulate_ideal()
    circuit = env.circuit.copy()
    circuit = transpile(circuit, basis_gates=["rx", "ry", "rz", "cx"])
    noise_level = 0.01
    reps = represent_operations_in_circuit_with_local_depolarizing_noise(circuit, noise_level)
    array = []
    for obs, exp_noisy, exp_ideal in dataset[:100]:
        # obs = np.diag([1., -1.])
        # obs = np.kron(np.eye(2**3), obs)
        obs = np.kron(obs, np.eye(2**3))
        # expectation_ideal = ideal_state.expectation_value(obs).real
        print(execute_pec(circuit, obs))
        print('exp_noisy: ', exp_noisy)
        print('exp_ideal: ', exp_ideal)
        expectation_ideal = execute_pec(circuit, obs, noise_level=0.0)
        pec_value = pec.execute_with_pec(circuit, partial(execute_pec, obs=obs), representations=reps)
        array.append(abs(expectation_ideal - pec_value))
        
        print(expectation_ideal)
        print(pec_value)
        assert False
    print(round(np.mean(array), 6))

    

def execute_pec(circuit, obs, noise_level=0.01):
    # Replace with code based on your frontend and backend.
    mitiq_circuit, _ = convert_to_mitiq(circuit)
    # We add a simple noise model to simulate a noisy backend.
    noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return np.trace(obs @ rho).real





if __name__ == '__main__':
    env = IBMQEnv.load('../environments/ibmq_random.pkl')
    # IBMQ.load_account()
    # provider = IBMQ.get_provider(
    #     hub='ibm-q',
    #     group='open',
    #     project='main'
    # )
    # backend = provider.get_backend(env.config.backend)
    # noise_model = NoiseModel.from_backend(backend, readout_error=False, thermal_relaxation=False)
    noise_model = NoiseModel()
    error_1 = depolarizing_error(0.001, 1)  # single qubit gates
    error_2 = depolarizing_error(0.01, 2)
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'i', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cy', 'cz', 'ch', 'crz', 'swap', 'cu1', 'cu3', 'rzz'])
    with open('../data_mitigate/testset_3.pkl', 'rb') as f:
        dataset = pickle.load(f)

    # benchmark_zne()
    # benchmark_cdr()
    benchmark_pec()