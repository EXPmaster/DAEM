import pickle
import os
import argparse
import numpy as np
import cirq
import cirq.ops as ops
from cirq.circuits import InsertStrategy
from tqdm import tqdm
import sympy.parsing.sympy_parser as sympy_parser
from multiprocessing import Pool


def generate_data(num_data, num_layers, num_qubits, threshold=0.3, num_samples=10_000):
    qubit_gates = [
        [ops.rx, ops.ry, ops.rz, ops.X, ops.Y, ops.Z, ops.H],
        [ops.CNOT, ops.CZ]
    ]
    data_list = []
    count = 0
    while True:
        # qubit1 = cirq.GridQubit(0, 0)
        # qubit2 = cirq.LineQubit(0)
        # qubit = cirq.NamedQubit('qubit')
        rad = np.random.uniform(-np.pi / 2, np.pi / 2)
        input_qbits = [cirq.NamedQubit(f'in_qubit{i}') for i in range(num_qubits)]
        circuit = cirq.Circuit([ops.rx(rad)(qbit) for qbit in input_qbits])
        # operations = [cirq.unitary(ops.rx(rad))]# [np.zeros(num_qubits)] 
        for layer in range(num_layers - 1):
            gate_idx = np.random.choice([0, 0, 0, 1])
            gate_set = qubit_gates[gate_idx]
            if gate_idx == 0:
                gates = []
                for qubit in input_qbits:
                    g = np.random.choice(gate_set)
                    if g in [ops.rx, ops.ry, ops.rz]:
                        g = g(np.random.uniform(-np.pi / 2, np.pi / 2))
                    gates.append(g(qubit))
                circuit.append(gates, strategy=InsertStrategy.NEW_THEN_INLINE)

            else:
                gate = np.random.choice(gate_set)
                select_qubit_idx = np.random.choice(num_qubits, size=2, replace=False)
                circuit.append([gate(*[input_qbits[i] for i in select_qubit_idx])], strategy=InsertStrategy.NEW_THEN_INLINE)
        operations = cirq.unitary(circuit)
        z_obs = cirq.Z(input_qbits[0])# cirq.PauliString(cirq.Z(q) for q in input_qbits)
        # Simulate ideal circuit
        # output_state = cirq.Simulator().simulate(circuit).final_state_vector
        output_value = cirq.Simulator().simulate_expectation_values(circuit, observables=[z_obs])
        expectation_ideal = round(output_value[0].real, 4)
        if abs(expectation_ideal) < threshold:
            continue

        # Simulate noisy circuit
        noisy_circuit = circuit.with_noise(cirq.depolarize(p=0.01))
        collector = cirq.PauliSumCollector(noisy_circuit, z_obs, samples_per_term=num_samples)
        collector.collect(sampler=cirq.DensityMatrixSimulator())
        expectation_noisy = collector.estimated_energy()
        
        count += 1
        print('\rProcess: {:.2f}%'.format(100 * count / num_data), end='')
        data_list.append([operations, expectation_ideal, expectation_noisy])
        if count >= num_data:
            break

    return data_list


def generate_data_v2(num_data, num_layers, num_qubits, threshold=0.3, num_samples=10_000):
    qubit_gates = [
        [ops.rx, ops.ry, ops.rz, ops.X, ops.Y, ops.Z, ops.H, ops.S, ops.T],
        [ops.CNOT, ops.CZ]
    ]
    observable_list = [ops.X, ops.Y, ops.Z]
    data_list = []
    count = 0
    while True:
        rad = np.random.uniform(-np.pi / 2, np.pi / 2)
        input_qbits = cirq.NamedQubit.range(num_qubits, prefix="q")
        circuit = cirq.Circuit([ops.rx(rad)(qbit) for qbit in input_qbits])
        # operations = [cirq.unitary(ops.rx(rad))]# [np.zeros(num_qubits)] 
        for layer in range(num_layers - 1):
            gate_idx = np.random.choice([0, 0, 0, 1])
            gate_set = qubit_gates[gate_idx]
            if gate_idx == 0:
                gates = []
                for qubit in input_qbits:
                    g = np.random.choice(gate_set)
                    if g in [ops.rx, ops.ry, ops.rz]:
                        g = g(np.random.uniform(-np.pi / 2, np.pi / 2))
                    gates.append(g(qubit))
                circuit.append(gates, strategy=InsertStrategy.NEW_THEN_INLINE)
            else:
                gate = np.random.choice(gate_set)
                select_qubit_idx = np.random.choice(num_qubits, size=2, replace=False)
                circuit.append([gate(*[input_qbits[i] for i in select_qubit_idx])], strategy=InsertStrategy.NEW_THEN_INLINE)
                circuit.append([ops.I(input_qbits[i]) for i in range(num_qubits) if i not in select_qubit_idx], strategy=InsertStrategy.INLINE)
        operations = cirq.unitary(circuit)
        obs = cirq.PauliString(np.random.choice(observable_list)(input_qbits[0]))  # cirq.PauliString(np.random.choice(observable_list)(q) for q in input_qbits)    

        # ********** Simulate ideal circuit **********
        # output_state = cirq.Simulator().simulate(circuit).final_state_vector
        ops_moments = np.stack([cirq.unitary(m) for m in circuit])
        assert ops_moments.shape == (num_layers, 2 ** num_qubits, 2 ** num_qubits), 'illegal gate matrix shape'
        output_value = cirq.Simulator().simulate_expectation_values(circuit, observables=[obs])
        expectation_ideal = round(output_value[0].real, 5)
        if abs(expectation_ideal) < threshold:
            continue

        # ********** Simulate noisy circuit **********
        noisy_circuit = circuit.with_noise(cirq.depolarize(p=0.01))
        # collector = cirq.PauliSumCollector(noisy_circuit, obs, samples_per_term=num_samples)
        # collector.collect(sampler=cirq.DensityMatrixSimulator())
        # expectation_noisy = collector.estimated_energy()

        try:
            rho = cirq.DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
            # psum = cirq.PauliSum.from_boolean_expression(
            #     sympy_parser.parse_expr('x0'),
            #     {'x0': cirq.NamedQubit('q0')})
            psum = obs
            energy = psum.expectation_from_density_matrix(
                state=rho, qubit_map={q: i for i, q in enumerate(input_qbits)}
            )
        except Exception as e:
            continue
        expectation_noisy = round(energy.real, 5)
        # print(expectation_ideal, expectation_noisy)
        count += 1
        print('\rProcess: {:.2f}%'.format(100 * count / num_data), end='')
        data_list.append([operations, ops_moments, cirq.unitary(obs), expectation_ideal, expectation_noisy])
        if count >= num_data:
            break

    return data_list


def split_datset(args, dataset, train_ratio=0.8):
    print('spliting dataset...')
    train_len = int(len(dataset) * train_ratio)
    trainset = dataset[:train_len]
    testset = dataset[train_len:]
    with open(os.path.join(args.save_dir, args.train_name), 'wb') as f:
        pickle.dump(trainset, f)
    with open(os.path.join(args.save_dir, args.test_name), 'wb') as f:
        pickle.dump(testset, f)
    print(f'train len: {len(trainset)}, test len: {len(testset)}')


def run_multiprocess(args):
    print('generating new data...')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    pool = Pool(processes=args.num_workers)
    data_queue = []

    for i in range(args.num_circuits // 1000):
        data_queue.append(pool.apply_async(generate_data_v2, args=(1000, args.depth, args.num_qubits, args.threshold)))
    pool.close()
    pool.join()

    dataset = []
    for item in data_queue:
        dataset += item.get()
    with open(os.path.join(args.save_dir, args.data_name), 'wb') as f:
        pickle.dump(dataset, f)
    
    return dataset


def run_test():
    print('runing test...')
    generate_data_v2(100, 6, 4)
    print('\nDone without saving data.')


def toy_circuit():
    input_qbit = cirq.NamedQubit(f'in_qubit')
    circuit = cirq.Circuit([ops.I(input_qbit)])
    circuit.append(ops.Z(input_qbit))
    z_obs = cirq.Z(input_qbit)
    output_value = cirq.Simulator().simulate_expectation_values(circuit, observables=[z_obs])
    print(output_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', default=5, type=int, help='depth of the circuit')
    parser.add_argument('--num-qubits', default=4, type=int, help='the number of qubits used in the circuit')
    parser.add_argument('--num-circuits', default=600_000, type=int, help='the number of circuits to build')
    parser.add_argument('--threshold', default=0.005, type=float, help='keep data for exp_ideal >= threshold')
    parser.add_argument('--num-workers', default=16, type=int, help='the number of processes used in generating data')
    parser.add_argument('--save-dir', default='./data2', type=str)
    parser.add_argument('--data-name', default='dataset_0.pkl', type=str)
    parser.add_argument('--train-name', default='trainset_1.pkl', type=str)
    parser.add_argument('--test-name', default='testset_1.pkl', type=str)
    parser.add_argument('--generate', default=False, action='store_true', help='whether or not generate new data')
    parser.add_argument('--split', default=False, action='store_true', help='whether or not split train/test set')
    parser.add_argument('--ratio', default=0.8, type=float, help='train set ratio')
    args = parser.parse_args()

    if args.generate:
        dataset = run_multiprocess(args)
        print()
    elif args.split:
        data_path = os.path.join(args.save_dir, args.data_name)
        assert os.path.exists(data_path), 'File not exists.'
        print('loading data...')
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        run_test()

    if args.split:
        print(f'Length of data: {len(dataset)}')
        split_datset(args, dataset, train_ratio=args.ratio)
    print('Done.')
    # toy_circuit()
