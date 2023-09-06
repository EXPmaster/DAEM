import pickle
import os
import numpy as np
import torch
import torch.cuda as cuda
from cuquantum import cutensornet as cutn
from qiskit.quantum_info import DensityMatrix, Operator, Pauli, state_fidelity, random_statevector, random_density_matrix
from tqdm import tqdm

from basic_circuits import swaptest
from mpo_circuit_simulator import MPOCircuitSimulator


def get_random_states(num_qubits, save_path, total_num=170):
    # obtain random states for swap test. Filter those with fidelity > 0.4
    states = []
    count = 0
    with tqdm(total=total_num) as pbar:
        while count < 170:
            state1 = random_statevector(2 ** num_qubits)
            state2 = random_statevector(2 ** num_qubits)
            fid = state_fidelity(state1, state2)
            if fid > 0.3:
                states.append([state1.data, state2.data])
                count += 1
                pbar.update(1)
    with open(os.path.join(save_path, 'states.pkl'), 'wb') as f:
        pickle.dump(states, f)


def generate_data(states, backend, save_root, file_name, train=True):
    dataset = []
    noise_levels = np.linspace(0.005, 0.1, 4)
    noises = 1 - np.exp(-2 * noise_levels)
    print(noises)
    ref_state = torch.tensor([[1., 0.], [0., 0.]], dtype=dtype, device=device)
    for state1, state2 in tqdm(states):
        state1_t = torch.from_numpy(state1).to(dtype=dtype, device=device)
        state2_t = torch.from_numpy(state2).to(dtype=dtype, device=device)
        state1_t = torch.outer(state1_t, state1_t.conj())
        state2_t = torch.outer(state2_t, state2_t.conj())
        if train:
            init_rho = torch.kron(
                torch.from_numpy(random_density_matrix(2).data).to(dtype=dtype, device=device),
                torch.kron(state1_t, state2_t)
            )
        else:
            init_rho = torch.kron(ref_state, torch.kron(state1_t, state2_t))
        meas_ideal = round(backend.run([pauli_z], (0,), init_rho=init_rho).real, 6)
        # print(np.abs(np.inner(state1.conj(), state2)) ** 2, meas_ideal)
        meas_noisy = [
            round(backend.run([pauli_z], (0,), noise_level=i, init_rho=init_rho).real, 6)
            for i in noises
        ]
        param = 0.0
        obs_ret = [pauli_z.cpu().numpy(), np.eye(2)]
        selected_qubits = [0, 1]
        dataset.append([param, obs_ret, selected_qubits, noise_levels, meas_noisy, meas_ideal])
    
    with open(os.path.join(save_root, file_name), 'wb') as f:
        pickle.dump(dataset, f)


def generate_data_parallel(states, circuit, backend, save_root, file_name, train=True):
    backend
    dataset = []
    noise_levels = np.linspace(0.05, 0.15, 4)
    noises = 1 - np.exp(-2 * noise_levels)
    print(noises)
    ref_state = torch.tensor([[1., 0.], [0., 0.]], dtype=dtype, device=device)
    streams = [cuda.Stream() for i in range(len(noises) + 1)]
    pauli_zs = [pauli_z.clone() for _ in range(len(streams))]
    backends = [
        MPOCircuitSimulator(
            circuit,
            'dephasing',
            dtype=dtype,
            device=device,
            options=options[k]
        ) for k in range(len(noises))
    ]
    for state1, state2 in tqdm(states):
        state1_t = torch.from_numpy(state1).to(dtype=dtype, device=device)
        state2_t = torch.from_numpy(state2).to(dtype=dtype, device=device)
        state1_t = torch.outer(state1_t, state1_t.conj())
        state2_t = torch.outer(state2_t, state2_t.conj())
        if train:
            init_rho = torch.kron(
                torch.from_numpy(random_density_matrix(2).data).to(dtype=dtype, device=device),
                torch.kron(state1_t, state2_t)
            )
        else:
            init_rho = torch.kron(ref_state, torch.kron(state1_t, state2_t))
        init_rhos = [init_rho.clone() for _ in range(len(streams))]
        
        meas_noisy = [None for _ in range(len(noises))]
        with cuda.stream(streams[-1]):
            meas_ideal = round(backend.run([pauli_zs[-1]], (0,), init_rho=init_rhos[-1]).real, 6)
            # print(f'stream -1, meas_ideal:', meas_ideal)

        for i in range(len(noises)):
            with cuda.stream(streams[i]):
                meas_noisy[i] = round(backends[i].run([pauli_zs[i]], (0,), noise_level=noises[i], init_rho=init_rhos[i]).real, 6)
                # print(f'stream {i}, meas_noisy:', meas_noisy[i])

        for i in range(len(streams)):
            streams[i].synchronize()
        param = 0.0
        obs_ret = [pauli_z.cpu().numpy(), np.eye(2)]
        selected_qubits = [0, 1]
        dataset.append([param, obs_ret, selected_qubits, noise_levels, meas_noisy, meas_ideal])
    
    with open(os.path.join(save_root, file_name), 'wb') as f:
        pickle.dump(dataset, f)


def generate_clifford_data():
    ...


if __name__ == '__main__':
    SAVE_ROOT = 'data_st11q_pd'
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)
    
    num_qubits_per_state = 5
    if not os.path.exists(os.path.join(SAVE_ROOT, 'states.pkl')):
        get_random_states(num_qubits_per_state, SAVE_ROOT)

    handles = [cutn.create() for _ in range(5)]
    device = 'cuda'
    options = [{'handle': handles[i]} for i in range(len(handles))]
    dtype = torch.complex64

    pauli_z = torch.tensor([[1., 0.], [0., -1.]], dtype=dtype, device=device)
    circuit_train = swaptest(num_qubits_per_state, train=True)
    circuit_test = swaptest(num_qubits_per_state, train=False)

    with open(os.path.join(SAVE_ROOT, 'states.pkl'), 'rb') as f:
        states = pickle.load(f)

    backend_train = MPOCircuitSimulator(
        circuit_train,
        'dephasing',
        dtype=dtype,
        device=device,
        options=options[-1]
    )

    backend_test = MPOCircuitSimulator(
        circuit_test,
        'dephasing',
        dtype=dtype,
        device=device,
        options=options[-1]
    )

    generate_data_parallel(states[:100], circuit_train, backend_train, SAVE_ROOT, 'trainset.pkl')
    generate_data_parallel(states[100:150], circuit_train, backend_train, SAVE_ROOT, 'valset.pkl')
    generate_data_parallel(states[150:], circuit_test, backend_test, SAVE_ROOT, 'testset.pkl', train=False)
    
    for i in range(len(handles)):
        cutn.destroy(handles[i])

    