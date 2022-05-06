import pickle
from functools import partial

from mitiq.interface.mitiq_qiskit import qiskit_utils
from mitiq import zne, cdr
import qiskit
from qiskit import IBMQ, Aer, QuantumCircuit, transpile, execute
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
import numpy as np
import torch

from my_envs import IBMQEnv


def benchmark_zne():
    env.gen_new_circuit_without_id()
    ideal_state = env.simulate_ideal()
    circuit = env.circuit.copy()
    # circuit.measure(0, 0)
    
    scale_factors = np.arange(1, 3, 0.5)# [1., 1.5, 2., 2.5, 3.]
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
        obs = np.diag([1., -1.])
        obs = np.kron(np.eye(2), obs)
        expectation_ideal = ideal_state.expectation_value(obs).real
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
    env.gen_new_circuit_without_id()
    ideal_state = env.simulate_ideal()
    circuit = env.circuit.copy()
    # circuit.measure(0, 0)
    print(circuit)
    shots = 10000

    array = []
    for obs, _, _ in dataset:
        obs = np.diag([1., -1.])
        obs = np.kron(np.eye(2), obs)
        print(executor(circuit, obs, shots=shots))
        expectation_ideal = ideal_state.expectation_value(obs).real
        mitigated_measurement = cdr.execute_with_cdr(
            circuit,
            partial(executor, obs=obs, shots=shots),
            num_training_circuits=20,
            simulator=partial(sim, obs=obs, shots=shots),
            seed=0,
            fit_function=cdr.linear_fit_function_no_intercept
        ).real
        array.append(abs(expectation_ideal - mitigated_measurement))
        print(expectation_ideal)
        print(mitigated_measurement)
        assert False
    print(round(np.mean(array), 6))


def benchmark_pec():
    env.gen_new_circuit_without_id()
    ideal_state = env.simulate_ideal()
    circuit = env.circuit.copy()
    circuit.measure(0, 0)
    
    scale_factors = np.arange(1, 3, 0.5)# [1., 1.5, 2., 2.5, 3.]
    folded_circuits = [
        zne.scaling.fold_gates_at_random(circuit, scale)
        for scale in scale_factors
    ]
    shots = 10000
    backend_ideal = Aer.get_backend('aer_simulator')
    results = backend_ideal.run(circuit, shots=shots).result()
    counts_ideal = results.get_counts()

    backend_simulator = AerSimulator.from_backend(backend)
    result = qiskit.execute(
        experiments=folded_circuits,
        backend=backend_simulator,
        noise_model=noise_model,
        basis_gates=noise_model.basis_gates,
        optimization_level=0,  # Important to preserve folded gates.
        shots=shots,
    ).result()
    counts = [result.get_counts(i) for i in range(len(scale_factors))]
    array = []
    for obs, _, _ in dataset:
        # obs = np.diag([1., -1.])
        # expectation_ideal = ideal_state.expectation_value(np.kron(np.eye(2), obs)).real
        expectation_ideal = env.measure_obs(counts_ideal, obs, shots)
        expectation_values = [env.measure_obs(c, obs, shots).real for c in counts]
        # zero_noise_value = zne.ExpFactory.extrapolate(scale_factors, expectation_values, asymptote=0.5)
        zero_noise_value = zne.LinearFactory.extrapolate(scale_factors, expectation_values)
        array.append(abs(expectation_ideal - zero_noise_value))
        # print(expectation_ideal)
        # print(zero_noise_value)
        # assert False
    print(round(np.mean(array), 6))


def executor(circ, obs, shots):
    return qiskit_utils.execute_with_shots_and_noise(circ, obs, noise_model, shots)


def sim(circ, obs, shots):
    return qiskit_utils.execute_with_shots(circ, obs, shots)


if __name__ == '__main__':
    env = IBMQEnv.load('../environments/ibmq1.pkl')
    IBMQ.load_account()
    provider = IBMQ.get_provider(
        hub='ibm-q',
        group='open',
        project='main'
    )
    backend = provider.get_backend(env.config.backend)
    noise_model = NoiseModel.from_backend(backend, readout_error=False, thermal_relaxation=False)
    with open('../data_mitigate/testset_1.pkl', 'rb') as f:
        dataset = pickle.load(f)

    benchmark_zne()
    # benchmark_cdr()