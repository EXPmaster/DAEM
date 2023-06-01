import pickle
from collections import OrderedDict, namedtuple

import numpy as np
from qiskit.quantum_info.operators import Pauli
from qiskit.opflow import PauliOp


class CircuitParser:
    
    """Parse circuit into Hamiltonian operators layerwise"""

    Operator = namedtuple('Operator', ['op_str', 'params', 'coupling'])
    Hamiltonian = namedtuple('Hamiltonian', ['system', 'bath'])

    def __init__(self):
        self.num_qubits = None
        self.u3_index_mapping = {0: 2, 1: 0, 2: 1}  # index the location of parameters in u3 gate.

    def parse(self, qc):
        self.num_qubits = qc.num_qubits
        parsed_operators = OrderedDict()
        gates_to_be_parsed = []
        qubit_list = []
        layer = 0
        for gate in qc:
            operation = gate.operation
            if operation.name == 'barrier':
                continue
            qubit_indices = [x.index for x in gate.qubits]
            if len(qubit_indices) == 1 and qubit_indices[0] not in qubit_list:
                qubit_list.append(qubit_indices[0])
                gates_to_be_parsed.append(gate)
            elif len(qubit_indices) == 2:
                # Parse CNOT immediately. If there are other gates in the gates_to_be_parsed, parse them first
                if len(gates_to_be_parsed):
                    parsed_operators[f'layer{layer}'] = self._parse_operator(gates_to_be_parsed, 'u')
                    layer += 1
                    qubit_list = []
                    gates_to_be_parsed = []
                parsed_operators[f'layer{layer}'] = self._parse_operator([gate], 'cx')
                layer += 1
            else:
                # Meet redundant qubit index. Parse the previous ones.
                parsed_operators[f'layer{layer}'] = self._parse_operator(gates_to_be_parsed, 'u')
                layer += 1
                qubit_list = qubit_indices
                gates_to_be_parsed = [gate]
                    
        if len(gates_to_be_parsed):
            parsed_operators[f'layer{layer}'] = self._parse_operator(gates_to_be_parsed, 'u')

        hamiltonians = self._construct_hamiltonian(parsed_operators)
        return hamiltonians

    def _parse_operator(self, gates, gate_type):
        operators = []
        if gate_type == 'u':
            pauli_strings = ['Z', 'X', 'Z']
            for i in range(3):
                op_strings = []
                params = []
                couplings = []
                for gate in gates:
                    op_str = ['I'] * self.num_qubits
                    coupling = ['I'] * self.num_qubits
                    param = gate.operation.params[self.u3_index_mapping[i]] + np.pi / 2 * (i - 1)
                    qubit_index = gate.qubits[0].index
                    op_str[qubit_index] = pauli_strings[i]
                    coupling[qubit_index] = 'Z'

                    op_strings.append(''.join(op_str))
                    params.append(param / 2)
                    couplings.append(''.join(coupling))
                operators.append(self.Operator(op_strings, params, couplings))

        elif gate_type == 'cx':
            gate = gates[0]
            qubits = [x.index for x in gate.qubits]
            pauli_strings = ['ZI', 'ZX', 'IX']
            op_strings = []
            params = []
            couplings = []
            for qubit_idx in qubits:
                coupling = ['I'] * self.num_qubits
                coupling[qubit_idx] = 'Z'
                couplings.append(''.join(coupling))
            for i in range(3):
                op_str = ['I'] * self.num_qubits
                param = 0.25 * np.pi * (-1) ** (i + 1)
                op_str[qubits[0]] = pauli_strings[i][0]
                op_str[qubits[1]] = pauli_strings[i][1]
                op_strings.append(''.join(op_str))
                params.append(param)
            operators.append(self.Operator(op_strings, params, couplings))

        return operators

    def _construct_hamiltonian(self, operators):
        """Assume endtime is always 1, i.e.,
        for each layer, evolve to t = 1 is equivalent to executing the original circuit.
        """
        layerwise_hamiltonians = []
        for layer, operator_list in operators.items():
            for operator in operator_list:
                system_hamiltonian = sum([PauliOp(Pauli(operator.op_str[i]), coeff=operator.params[i])
                                          for i in range(len(operator.op_str))])
                bath_hamiltonian = sum([PauliOp(Pauli(coupling), coeff=1)
                                        for coupling in operator.coupling])
                layerwise_hamiltonians.append(self.Hamiltonian(system_hamiltonian, bath_hamiltonian))
        return layerwise_hamiltonians

    
if __name__ == '__main__':
    with open('../environments/circuits/vqe_4l/vqe_0.4.pkl', 'rb') as f:
        circuit = pickle.load(f)
    
    parser = CircuitParser()
    hs = parser.parse(circuit)

    from hamiltonian_simulator import HamiltonianSimulator, AnalyticSimulator
    from qiskit.quantum_info import Statevector
    state = Statevector(circuit).data
    # backend = HamiltonianSimulator(noise_scale=0.05)
    backend = AnalyticSimulator(noise_scale=0.3)
    final_rho = backend.run(hs, verbose=True)
    print(state.conj() @ final_rho @ state)
