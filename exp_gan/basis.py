__all__ = ['RandomObservable', 'GateRX', 'GateRY', 'GateRZ', 'GateRYZ', 'GateRZX', 'GateRXY', 'GatePiX',
        'GatePiY', 'GatePiZ', 'GatePiYZ', 'GatePiZX', 'GatePiXY']


import cirq
import numpy as np
from scipy.stats import unitary_group


class RandomObservable(cirq.Gate):
    def __init__(self):
        super(RandomObservable, self)
        self.matrix = unitary_group.rvs(2)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return self.matrix

    def _circuit_diagram_info_(self, args):
        return "RndOBS"


class GatePi(cirq.Gate):
    def __init__(self):
        super(GatePi, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return np.array([
            [1.0+0.0j,  0.0],
            [0.0, 0.0]
        ])

    def _circuit_diagram_info_(self, args):
        return "Pi"


class GateS(cirq.Gate):
    def __init__(self):
        super(GateS, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return (np.eye(2) - 1j * cirq.unitary(cirq.Z)) / np.sqrt(2)

    def _circuit_diagram_info_(self, args):
        return "S"


class GateRX(cirq.Gate):
    def __init__(self):
        super(GateRX, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return (np.eye(2) + 1j * cirq.unitary(cirq.X)) / np.sqrt(2)

    def _circuit_diagram_info_(self, args):
        return "GRx"


class GateRY(cirq.Gate):
    def __init__(self):
        super(GateRY, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return (np.eye(2) + 1j * cirq.unitary(cirq.Y)) / np.sqrt(2)

    def _circuit_diagram_info_(self, args):
        return "GRy"


class GateRZ(cirq.Gate):
    def __init__(self):
        super(GateRZ, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return (np.eye(2) + 1j * cirq.unitary(cirq.Z)) / np.sqrt(2)

    def _circuit_diagram_info_(self, args):
        return "GRz"


class GateRYZ(cirq.Gate):
    def __init__(self):
        super(GateRYZ, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return (cirq.unitary(cirq.Y) + cirq.unitary(cirq.Z)) / np.sqrt(2)

    def _circuit_diagram_info_(self, args):
        return "GRyz"


class GateRZX(cirq.Gate):
    def __init__(self):
        super(GateRZX, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return (cirq.unitary(cirq.Z) + cirq.unitary(cirq.X)).astype(np.cfloat) / np.sqrt(2)

    def _circuit_diagram_info_(self, args):
        return "GRzx"


class GateRXY(cirq.Gate):
    def __init__(self):
        super(GateRXY, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return (cirq.unitary(cirq.X) + cirq.unitary(cirq.Y)) / np.sqrt(2)

    def _circuit_diagram_info_(self, args):
        return "GRxy"


class GatePiX(cirq.Gate):
    def __init__(self):
        super(GatePiX, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return (np.eye(2) + cirq.unitary(cirq.X)).astype(np.cfloat) / 2

    def _circuit_diagram_info_(self, args):
        return "PiX"


class GatePiY(cirq.Gate):
    def __init__(self):
        super(GatePiY, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return (np.eye(2) + cirq.unitary(cirq.Y)) / 2

    def _circuit_diagram_info_(self, args):
        return "PiY"


class GatePiZ(cirq.Gate):
    def __init__(self):
        super(GatePiZ, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return (np.eye(2) + cirq.unitary(cirq.Z)).astype(np.cfloat) / 2

    def _circuit_diagram_info_(self, args):
        return "PiZ"


class GatePiYZ(cirq.Gate):
    def __init__(self):
        super(GatePiYZ, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return (cirq.unitary(cirq.Y) + 1j * cirq.unitary(cirq.Z)) / 2

    def _circuit_diagram_info_(self, args):
        return "PiYZ"


class GatePiZX(cirq.Gate):
    def __init__(self):
        super(GatePiZX, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return (cirq.unitary(cirq.Z) + 1j * cirq.unitary(cirq.X)) / 2

    def _circuit_diagram_info_(self, args):
        return "PiZX"


class GatePiXY(cirq.Gate):
    def __init__(self):
        super(GatePiXY, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return (cirq.unitary(cirq.X) + 1j * cirq.unitary(cirq.Y)) / 2

    def _circuit_diagram_info_(self, args):
        return "PiXY"


if __name__ == '__main__':
    print(cirq.unitary(GatePiX()))
