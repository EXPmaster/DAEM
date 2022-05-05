__all__ = ['DQCp']


from qiskit import QuantumCircuit


def DQCp():
    circ = QuantumCircuit(2, 1)
    circ.h(0)
    circ.h(1)
    circ.cx(0, 1)
    circ.rz(np.pi/3, 1)
    circ.cx(0, 1)
    circ.h(0)
    circ.barrier()

    return circ