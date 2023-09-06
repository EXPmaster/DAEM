import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister


def ccnot(kernel, c1, c2, t, train=False):
    if not train:
        kernel.h(t)
        kernel.cx(c2, t)
        kernel.tdg(t)
        kernel.cx(c1, t)
        kernel.t(t)
        kernel.cx(c2, t)
        kernel.tdg(t)
        kernel.cx(c1, t)
        kernel.t(c2)
        kernel.t(t)
        kernel.cx(c1, c2)
        kernel.h(t)
        kernel.t(c1)
        kernel.tdg(c2)
        kernel.cx(c1, c2)
    else:
        kernel.i(t)
        kernel.cx(c2, t)
        kernel.i(t)
        kernel.cx(c1, t)
        kernel.i(t)
        kernel.cx(c2, t)
        kernel.i(t)
        kernel.cx(c1, t)
        kernel.i(c2)
        kernel.i(t)
        kernel.cx(c1, c2)
        kernel.i(t)
        kernel.i(c1)
        kernel.i(c2)
        kernel.cx(c1, c2)
    

def cswap(kernel, c, t1, t2, train=False):
    ccnot(kernel, c, t1, t2, train)
    ccnot(kernel, c, t2, t1, train)
    ccnot(kernel, c, t1, t2, train)
    # add dephasing noise


def swaptest(nqps, init_state=None, p=0.0, train=False):
    """
    Args:
        nqps: number of qubits per state.
    """
    q = QuantumRegister(2 * nqps + 1)
    circuit = QuantumCircuit(q)
    if init_state is not None:
        # init_state = switch_little_big_endian_state(init_state)
        # circuit.initialize(init_state, q[:])
        circuit.append(init_state[0], q[1: nqps + 1])
        circuit.append(init_state[1], q[1 + nqps:])

    circuit.h(q[0])
    for i in range(nqps):
        if np.random.rand() < p: circuit.z(q[i + 1])
        if np.random.rand() < p: circuit.z(q[nqps + i + 1])
        cswap(circuit, q[0], q[i + 1], q[nqps + 1 + i], train)
    circuit.h(q[0])
    return circuit


def switch_little_big_endian_state(state):
    reshape = [2] * int(np.log2(state.size))
    original_shape = state.shape
    state = state.reshape(reshape)

    axes = list(range(len(state.shape)))
    axes.reverse()

    mat = np.transpose(state, axes=axes).reshape(original_shape)

    return mat
