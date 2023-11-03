import os
import pickle
import numpy as np
import torch
from cuquantum import contract, contract_path, CircuitToEinsum, tensor
from cuquantum import cutensornet as cutn

from qiskit.quantum_info import Statevector
from qiskit.circuit.random import random_circuit
from qiskit import QuantumCircuit

from tqdm import tqdm

from basic_circuits_bak import swaptest


def generate_circuits(nums, save_path, train):
    for i in tqdm(range(nums)):
        rand_circuit1 = random_circuit(num_qubits_per_state, 4, max_operands=2)
        rand_circuit2 = random_circuit(num_qubits_per_state, 4, max_operands=2)
        ideal_circuit = swaptest(num_qubits_per_state, [rand_circuit1, rand_circuit2], train=train)
        noisy_circuits = []
        for n_lv in np.linspace(0.05, 0.2, 5):
            n_circuits = []
            for c_num in range(100):
                n_circuits.append(swaptest(num_qubits_per_state, [rand_circuit1, rand_circuit2], p=n_lv, train=train))
            noisy_circuits.append(n_circuits)
        circuits = [ideal_circuit, noisy_circuits]
        with open(os.path.join(save_path, f'{i}.pkl'), 'wb') as f:
            pickle.dump(circuits, f)


def generate_dataset(env_root, save_path, train=False):
    pauli_z = np.array([[1., 0.], [0., -1.]])
    identity = np.eye(2)
    dataset = []
    for circuit_name in tqdm(os.listdir(env_root)):
        circuit_path = os.path.join(env_root, circuit_name)
        with open(circuit_path, 'rb') as f:
            ideal_circuit, noisy_circuits = pickle.load(f)
        meas_ideal = get_expval(ideal_circuit, dtype, options)
        meas_noisy = [np.round(np.mean([get_expval(circ, dtype, options) for circ in n_circuits]), 6)
                        for n_circuits in noisy_circuits]
        param = 0.0
        obs_ret = [pauli_z, identity]
        selected_qubits = [0, 1]
        noise_scale = np.linspace(0.05, 0.2, 5)
        dataset.append([param, obs_ret, selected_qubits, noise_scale, meas_noisy, meas_ideal])
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def get_expval(circuit, dtype, options=None):
    myconverter = CircuitToEinsum(circuit, dtype=dtype, backend=torch)
    pauli_string = 'Z' + 'I' * (circuit.num_qubits - 1)
    expression, operands = myconverter.expectation(pauli_string, lightcone=True)
    expec = contract(expression, *operands, options=options).real
    return round(expec.item(), 6)


if __name__ == '__main__':
    torch.set_default_device('cuda:0')
    handle = cutn.create()
    options = {'handle': handle}
    dtype = torch.complex128

    num_qubits_per_state = 6
    TRAIN_PATH = 'circuits/st13q_circ_train'
    VAL_PATH = 'circuits/st13q_circ_val'
    TEST_PATH = 'circuits/st13q_circ_test'
    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)
    if not os.path.exists(VAL_PATH):
        os.makedirs(VAL_PATH)
    if not os.path.exists(TEST_PATH):
        os.makedirs(TEST_PATH)
    # generate_circuits(150, TRAIN_PATH, train=True)
    # generate_circuits(50, VAL_PATH, train=True)
    # generate_circuits(20, TEST_PATH, train=False)
    generate_dataset(TRAIN_PATH, 'dataset/st13q_train.pkl')
    generate_dataset(VAL_PATH, 'dataset/st13q_val.pkl')
    generate_dataset(TEST_PATH, 'dataset/st13q_test.pkl')

    cutn.destroy(handle)
