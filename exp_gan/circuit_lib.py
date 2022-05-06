__all__ = ['DQCp', 'swaptest']


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from PIL import Image


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


def swaptest():
    qc = QuantumRegister(1, name='qc')
    qa = QuantumRegister(2, name='qa')
    qb = QuantumRegister(2, name='qb')
    cr = ClassicalRegister(1)
    circ = QuantumCircuit(qc, qa, qb, cr)
    circ.h(qc)
    circ.h(qa[0])
    circ.cx(qa[0], qa[1])
    cswap = Cswap().to_instruction()
    circ.append(cswap, [qc, qa[0], qb[0]])
    circ.append(cswap, [qc, qa[1], qb[1]])
    circ.h(0)
    return circ

def CCNot():
    qr = QuantumRegister(3)
    circ = QuantumCircuit(qr, name='CCNot')
    circ.h(qr[2])
    circ.cx(qr[1], qr[2])
    circ.tdg(qr[2])
    circ.cx(qr[0], qr[2])
    circ.t(qr[2])
    circ.cx(qr[1], qr[2])
    circ.tdg(qr[2])
    circ.cx(qr[0], qr[2])
    circ.t(qr[1])
    circ.t(qr[2])
    circ.cx(qr[0], qr[1])
    circ.h(qr[2])
    circ.t(qr[0])
    circ.tdg(qr[1])
    circ.cx(qr[0], qr[1])
    return circ


def Cswap():
    qr = QuantumRegister(3)
    circ = QuantumCircuit(qr, name='Cswap')
    ccnot = CCNot().to_instruction()
    circ.append(ccnot, [qr[0], qr[1], qr[2]])
    circ.append(ccnot, [qr[0], qr[2], qr[1]])
    circ.append(ccnot, [qr[0], qr[1], qr[2]])
    return circ


def test_circuit():
    circ = QuantumCircuit(3)
    circ.cswap(0, 1, 2)
    return circ


if __name__ == '__main__':
    circuit = swaptest()
    circuit = test_circuit()
    graph = circuit.draw(output='latex', scale=3.0, initial_state=True)
    graph.save('swaptest.png', quality=95)