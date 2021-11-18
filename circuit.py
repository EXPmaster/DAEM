import pickle
import numpy as np
import cirq
import cirq.ops as ops
from cirq.circuits import InsertStrategy
from tqdm import tqdm


def generate_data(num_data, num_layers, num_qubits):
    gate_list = [ops.rx, ops.rz, ops.CNOT]
    data_list = []
    for idx in tqdm(range(num_data)):
        # qubit1 = cirq.GridQubit(0, 0)
        # qubit2 = cirq.LineQubit(0)
        # qubit = cirq.NamedQubit('qubit')
        init_qbits = [cirq.NamedQubit(f'p_qubit{i}') for i in range(num_qubits)]
        rad = np.random.uniform(-np.pi / 2, np.pi / 2)
        prepare_circuit = cirq.Circuit([ops.rx(rad)(qbit) for qbit in init_qbits])
        init_state = cirq.Simulator().simulate(prepare_circuit).final_state_vector

        input_qbits = [cirq.NamedQubit(f'in_qubit{i}') for i in range(num_qubits)]
        circuit = cirq.Circuit([ops.rx(rad)(qbit) for qbit in input_qbits])
        # operations = [cirq.unitary(ops.rx(rad))]# [np.zeros(num_qubits)] 
        for layer in range(num_layers - 1):
            gate_idx = np.random.choice(len(gate_list))
            gate = gate_list[gate_idx]
            if gate in [ops.rx, ops.rz]:
                select_num = 1
                gate = gate(np.random.uniform(-np.pi / 2, np.pi / 2))
            else:
                select_num = 2
            select_qubit_idx = np.random.choice(num_qubits, size=select_num, replace=False)
            # operation = np.ones(num_qubits) * -1
            # operation[select_qubit_idx] = gate_idx
            # operations.append(operation)
            circuit.append([gate(*[input_qbits[i] for i in select_qubit_idx])], strategy=InsertStrategy.NEW_THEN_INLINE)
        operations = cirq.unitary(circuit)
        # operations = np.array(operations)
        output_state = cirq.Simulator().simulate(circuit).final_state_vector
        data_list.append([init_state, operations, output_state])
    return data_list


def split_datset(dataset, train_ratio=0.8):
    train_len = int(len(dataset) * train_ratio)
    trainset = dataset[:train_len]
    testset = dataset[train_len:]
    with open('data/trainset.pkl', 'wb') as f:
        pickle.dump(trainset, f)
    with open('data/testset.pkl', 'wb') as f:
        pickle.dump(testset, f)
    print(f'train len: {len(trainset)}, test len: {len(testset)}')


if __name__ == '__main__':
    dataset = generate_data(100000, 5, 5)
    with open('data/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    split_datset(dataset)