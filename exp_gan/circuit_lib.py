__all__ = ['DQCp', 'swaptest', 'random_circuit']


import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from PIL import Image
from qiskit.circuit import Reset
from qiskit.circuit.library.standard_gates import (
    IGate,
    U1Gate,
    U2Gate,
    U3Gate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    RXGate,
    RYGate,
    RZGate,
    CXGate,
    CYGate,
    CZGate,
    CHGate,
    CRZGate,
    CU1Gate,
    CU3Gate,
    SwapGate,
    RZZGate,
    CCXGate,
    CSwapGate,
)
from qiskit.circuit.exceptions import CircuitError


def random_circuit(
    num_qubits, depth, max_operands=3, measure=False, conditional=False, reset=False, seed=None
):
    """Generate random circuit of arbitrary size and form.

    This function will generate a random circuit by randomly selecting gates
    from the set of standard gates in :mod:`qiskit.extensions`. For example:

    .. jupyter-execute::

        from qiskit.circuit.random import random_circuit

        circ = random_circuit(2, 2, measure=True)
        circ.draw(output='mpl')

    Args:
        num_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
        max_operands (int): maximum operands of each gate (between 1 and 3)
        measure (bool): if True, measure all qubits at the end
        conditional (bool): if True, insert middle measurements and conditionals
        reset (bool): if True, insert middle resets
        seed (int): sets random seed (optional)

    Returns:
        QuantumCircuit: constructed circuit

    Raises:
        CircuitError: when invalid options given
    """
    if max_operands < 1 or max_operands > 3:
        raise CircuitError("max_operands must be between 1 and 3")

    one_q_ops = [
        # IGate,
        U1Gate,
        U2Gate,
        U3Gate,
        XGate,
        YGate,
        ZGate,
        HGate,
        SGate,
        SdgGate,
        TGate,
        TdgGate,
        RXGate,
        RYGate,
        RZGate,
    ]
    one_param = [U1Gate, RXGate, RYGate, RZGate, RZZGate, CU1Gate, CRZGate]
    two_param = [U2Gate]
    three_param = [U3Gate, CU3Gate]
    two_q_ops = [CXGate, CYGate, CZGate, CHGate, CRZGate, CU1Gate, CU3Gate, SwapGate, RZZGate]  # SwapGate
    three_q_ops = [CCXGate, CSwapGate]

    qr = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, "c")
        qc.add_register(cr)

    if reset:
        one_q_ops += [Reset]

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # apply arbitrary random operations at every depth
    for _ in range(depth):
        # choose either 1, 2, or 3 qubits for the operation
        remaining_qubits = list(range(num_qubits))
        rng.shuffle(remaining_qubits)
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = rng.choice(range(max_possible_operands)) + 1
            operands = [remaining_qubits.pop() for _ in range(num_operands)]
            if num_operands == 1:
                operation = rng.choice(one_q_ops)
            elif num_operands == 2:
                operation = rng.choice(two_q_ops)
            elif num_operands == 3:
                operation = rng.choice(three_q_ops)
            if operation in one_param:
                num_angles = 1
            elif operation in two_param:
                num_angles = 2
            elif operation in three_param:
                num_angles = 3
            else:
                num_angles = 0
            angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
            register_operands = [qr[i] for i in operands]
            op = operation(*angles)

            # with some low probability, condition on classical bit values
            if conditional and rng.choice(range(10)) == 0:
                value = rng.integers(0, np.power(2, num_qubits))
                op.condition = (cr, value)

            qc.append(op, register_operands)

    if measure:
        qc.measure(qr, cr)

    return qc


def DQCp():
    circ = QuantumCircuit(2, 1)
    circ.h(0)
    circ.h(1)
    circ.cx(0, 1)
    circ.rz(np.pi/3, 1)
    circ.cx(0, 1)
    circ.h(0)

    return circ


def swaptest():
    qr = QuantumRegister(5, name='q')
    cr = ClassicalRegister(1)
    circ = QuantumCircuit(qr, cr)
    circ.h(qr[0])
    circ.h(qr[1])
    circ.cx(qr[1], qr[2])
    cswap = Cswap().to_instruction()
    circ.append(cswap, [qr[0], qr[1], qr[3]])
    circ.append(cswap, [qr[0], qr[2], qr[4]])
    circ.h(qr[0])
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
    # circuit = swaptest()
    # circuit = test_circuit()
    from my_envs import IBMQEnv
    env = IBMQEnv.load('../environments/ibmq_random.pkl')
    circuit = env.circuit  # DQCp()
    circuit.draw(output='latex', filename='random_circ.pdf', idle_wires=False, initial_state=True)
    # graph.save('../imgs/dqcp.png', quality=95)