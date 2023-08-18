import argparse
import os
import pickle
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import CSwapGate


def swaptest(n):
    qr = QuantumRegister(2 * n + 1, name='q')
    # cr = ClassicalRegister(1)
    circ = QuantumCircuit(qr)
    circ.h(qr[0])
    for idx in range(1, n + 1):
        circ.cswap(qr[0], qr[idx], qr[n + idx])
    circ.h(qr[0])
    return circ


def trotter_step(n, delta=0.5):
    circ = QuantumCircuit(n + 1)
    for i in range(n):
        circ.cx(i, n)
    circ.rz(delta * 2, n)
    for i in reversed(range(n)):
        circ.cx(i, n)
    return circ


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_qubits', type=int, default=10, help='Number of qubits per state.')
    parser.add_argument('--out_path', type=str, default='../environments/circuits/trotter', help='Output circuit dir.')
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    circuit = swaptest(args.num_qubits)
    circuit = transpile(circuit, basis_gates=['cx', 'u'])
    
    print(circuit)
    with open(os.path.join(args.out_path, 'tr11q_1.pkl'), 'wb') as f:
        pickle.dump(circuit, f)
    print('circuit generate finished.')
