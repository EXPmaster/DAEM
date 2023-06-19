import pickle
from collections import OrderedDict, namedtuple

import numpy as np
from qiskit.quantum_info.operators import Pauli
from qiskit.opflow import PauliOp
from scipy.stats import rv_continuous


class CircuitParser:
    
    """Parse circuit into Hamiltonian operators layerwise"""

    Operator = namedtuple('Operator', ['op_str', 'params', 'coupling'])
    Hamiltonian = namedtuple('Hamiltonian', ['system', 'bath'])

    def __init__(self):
        self.num_qubits = None
        self.u3_index_mapping = {0: 2, 1: 0, 2: 1}  # index the location of parameters in u3 gate.

    def parse(self, qc, return_hamiltonian=True):
        self.num_qubits = qc.num_qubits
        parsed_operators = OrderedDict()
        gates_to_be_parsed = []
        qubit_list = []
        layer = 0
        for gate in qc:
            operation = gate.operation
            if operation.name == 'barrier':
                continue
            qubit_indices = [qc.find_bit(x).index for x in gate.qubits]  # [x.index for x in gate.qubits]
            if len(qubit_indices) == 1 and qubit_indices[0] not in qubit_list:
                qubit_list.append(qubit_indices[0])
                gates_to_be_parsed.append(gate)
            elif len(qubit_indices) == 2:
                # Parse CNOT immediately. If there are other gates in the gates_to_be_parsed, parse them first
                if len(gates_to_be_parsed):
                    parsed_operators[layer] = self._parse_operator(gates_to_be_parsed, 'u')
                    layer += 1
                    qubit_list = []
                    gates_to_be_parsed = []
                parsed_operators[layer] = self._parse_operator([gate], 'cx')
                layer += 1
            else:
                # Meet redundant qubit index. Parse the previous ones.
                parsed_operators[layer] = self._parse_operator(gates_to_be_parsed, 'u')
                layer += 1
                qubit_list = qubit_indices
                gates_to_be_parsed = [gate]
                    
        if len(gates_to_be_parsed):
            parsed_operators[layer] = self._parse_operator(gates_to_be_parsed, 'u')

        if return_hamiltonian:
            hamiltonians = self._construct_hamiltonian(parsed_operators)
        else:
            hamiltonians = parsed_operators
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
                    param = float(gate.operation.params[self.u3_index_mapping[i]]) + np.pi / 2 * (i - 1)
                    qubit_index = gate.qubits[0].index
                    op_str[qubit_index] = pauli_strings[i]
                    coupling[qubit_index] = 'Z'

                    op_strings.append(''.join(op_str[::-1]))
                    params.append(param / 2)
                    couplings.append(''.join(coupling[::-1]))
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
                couplings.append(''.join(coupling[::-1]))
            for i in range(3):
                op_str = ['I'] * self.num_qubits
                param = 0.25 * np.pi * (-1) ** (i + 1)
                op_str[qubits[0]] = pauli_strings[i][0]
                op_str[qubits[1]] = pauli_strings[i][1]
                op_strings.append(''.join(op_str[::-1]))
                params.append(param)
            operators.append(self.Operator(op_strings, params, couplings))
        
        else:
            raise ValueError(f'Unknown gate type {gate_type}.')

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

    def construct_train(self, circuit, train_num=1):
        op_string_map = {
            'ZIII': 'ZZII', 'IZII': 'IZZI', 'IIZI': 'IIZZ', 'IIIZ': 'IIIZ',
            'XIII': 'XIII', 'IXII': 'XXII', 'IIXI': 'XXXI', 'IIIX': 'XXXX',
        }
        # op_string_map = {
        #     'ZIIIII': 'ZIIIII', 'IZIIII': 'ZZIIII', 'IIZIII': 'ZZZIII', 'IIIZII': 'ZZZZII', 'IIIIZI': 'ZZZZZI', 'IIIIIZ': 'ZZZZZZ',
        #     'XIIIII': 'XXIIII', 'IXIIII': 'IXXIII', 'IIXIII': 'IIXXII', 'IIIXII': 'IIIXXI', 'IIIIXI': 'IIIIXX', 'IIIIIX': 'IIIIIX'
        # }
        """Construct training circuit Hamiltonians from given Hamiltonian."""
        org_operators = self.parse(circuit, return_hamiltonian=False)
        sin_sampler = sin_prob_dist(a=0, b=np.pi)
        # for k, v in org_operators.items():
        #     print(k, v)
        # print('------------------')
        # print(org_operators)
        ret_hamils = []
        for t_itr in range(train_num):
            queue = []
            circuit_hs = OrderedDict()
            for layer, operator_list in org_operators.items():
                indicator = sum(['XZ' in operator for operator in operator_list[0].op_str])
                
                if not indicator:
                    if len(queue) == 1:
                        ops = queue.pop(0)
                        operators = []
                        for i in range(len(ops)):
                            op_str = [op_string_map[item] for item in ops[len(ops) - 1 - i].op_str]
                            params = [-item for item in ops[len(ops) - 1 - i].params]
                            couplings = operator_list[i].coupling
                            operators.append(self.Operator(op_str, params, couplings))
                        circuit_hs[layer] = operators
                    else:
                        ops = operator_list
                        operators = []
                        for i in range(len(ops)):
                            op_str = ops[i].op_str
                            if i != 1:
                                params = 2 * np.pi * np.random.uniform(size=len(ops[i].params)) + np.pi / 2 * (i - 1)
                            else:
                                params = sin_sampler.rvs(size=len(ops[i].params))
                            couplings = ops[i].coupling
                            operators.append(self.Operator(op_str, params, couplings))
                        circuit_hs[layer] = operators
                        queue.append(operators)

                else:
                    circuit_hs[layer] = operator_list
            H = self._construct_hamiltonian(circuit_hs)
            ret_hamils.append(H)
            
        return ret_hamils


class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)

    
if __name__ == '__main__':
    with open('../environments/circuits/vqe_4l/vqe_0.4.pkl', 'rb') as f:
        circuit = pickle.load(f)
    
    parser = CircuitParser()
    hs = parser.parse(circuit)
