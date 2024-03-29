import cirq
import cirq.ops as ops
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import TransformationPass, Layout
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.circuit.library import IGate, XGate, YGate, ZGate


class TransformCircWithPr(TransformationPass):

    def __init__(self, num_qubits):
        super().__init__()
        self.basis_ops = [
            ops.I, ops.X, ops.Y, ops.Z,
            # GateRX(), GateRY(), GateRZ(), GateRYZ(),
            # GateRZX(), GateRXY(), GatePiX(), GatePiY(),
            # GatePiZ(), GatePiYZ(), GatePiZX(), GatePiXY()
        ]
        self.num_qubits = num_qubits

    def run(self, dag, p):
        """Run the pass."""
        idx = 0
        # iterate over all operations
        for node in dag.op_nodes():
            # if we hit a RYY or RZZ gate replace it
            if node.op.name == 'id':
                cur_pr = p[idx // self.num_qubits, idx % self.num_qubits, :].ravel()
                gate = np.random.choice(self.basis_ops, p=cur_pr)
                # calculate the replacement
                replacement = QuantumCircuit(1)
                replacement.unitary(cirq.unitary(gate), 0, label=str(gate))

                # replace the node with our new decomposition
                dag.substitute_node_with_dag(node, circuit_to_dag(replacement))
                idx += 1

        return dag
    
    def __call__(self, circuit, p, property_set=None):
        result = self.run(circuit_to_dag(circuit), p)

        result_circuit = circuit

        if isinstance(property_set, dict):  # this includes (dict, PropertySet)
            property_set.clear()
            property_set.update(self.property_set)

        if isinstance(result, DAGCircuit):
            result_circuit = dag_to_circuit(result)
        elif result is None:
            result_circuit = circuit.copy()

        if self.property_set["layout"]:
            result_circuit._layout = self.property_set["layout"]
        if self.property_set["clbit_write_latency"] is not None:
            result_circuit._clbit_write_latency = self.property_set["clbit_write_latency"]
        if self.property_set["conditional_latency"] is not None:
            result_circuit._conditional_latency = self.property_set["conditional_latency"]

        return result_circuit


class TransformCircWithIndex(TransformationPass):

    def __init__(self):
        super().__init__()
        self.basis_ops = [
            # XGate(), ZGate()
            IGate(), XGate(), YGate(), ZGate()
            # ops.I, ops.X, ops.Y, ops.Z,
            # GateRX(), GateRY(), GateRZ(), GateRYZ(),
            # GateRZX(), GateRXY(), GatePiX(), GatePiY(),
            # GatePiZ(), GatePiYZ(), GatePiZX(), GatePiXY()
        ]

    def run(self, dag, idx):
        """Run the pass."""
        count = 0
        idx_org = idx
        # iterate over all operations
        for node in dag.op_nodes():
            # if we hit a RYY or RZZ gate replace it
            if node.op.label == 'miti':
                cur_idx = idx % len(self.basis_ops)
                gate = self.basis_ops[cur_idx]
                # calculate the replacement
                qr = QuantumRegister(1)
                replacement = QuantumCircuit(qr)
                replacement.append(gate, qr)

                # replace the node with our new decomposition
                dag.substitute_node_with_dag(node, circuit_to_dag(replacement))
                count += 1
                idx //= len(self.basis_ops)
                if count == 4:
                    idx = idx_org
                    count = 0
        return dag
    
    def __call__(self, circuit, idx, property_set=None):
        result = self.run(circuit_to_dag(circuit), idx)

        result_circuit = circuit

        if isinstance(property_set, dict):  # this includes (dict, PropertySet)
            property_set.clear()
            property_set.update(self.property_set)

        if isinstance(result, DAGCircuit):
            result_circuit = dag_to_circuit(result)
        elif result is None:
            result_circuit = circuit.copy()

        if self.property_set["layout"]:
            result_circuit._layout = self.property_set["layout"]
        if self.property_set["clbit_write_latency"] is not None:
            result_circuit._clbit_write_latency = self.property_set["clbit_write_latency"]
        if self.property_set["conditional_latency"] is not None:
            result_circuit._conditional_latency = self.property_set["conditional_latency"]

        return result_circuit


class TransformToClifford(TransformationPass):

    def __init__(self):
        super().__init__()
        self.clifford_list = ['I','X','Y','Z','S','XS',
                              'YS','ZS','H','XH','YH','ZH',
                              'SH','XSH','YSH','ZSH','HS','XHS',
                              'YHS','ZHS','SHS','XSHS','YSHS','ZSHS']

    @staticmethod
    def create_clifford_circuit(string):
        clifford_circuit = QuantumCircuit(1)
        for c in string:
            if c == 'I':
                clifford_circuit.i(0)
            elif c == 'X':
                clifford_circuit.x(0)
            elif c == 'Y':
                clifford_circuit.y(0)
            elif c == 'Z':
                clifford_circuit.z(0)
            elif c == 'H':
                clifford_circuit.h(0)
            elif c == 'S':
                clifford_circuit.s(0)
        return clifford_circuit

    def run(self, dag):
        """Run the pass."""
        # iterate over all operations
        for node in dag.op_nodes():
            # replace U with clifford
            if node.op.name == 'u':
                clifford_str = np.random.choice(self.clifford_list)
                # calculate the replacement
                replacement = self.create_clifford_circuit(clifford_str)

                # replace the node with our new decomposition
                dag.substitute_node_with_dag(node, circuit_to_dag(replacement))

        return dag
    
    def __call__(self, circuit, property_set=None):
        result = self.run(circuit_to_dag(circuit))

        result_circuit = circuit

        if isinstance(property_set, dict):  # this includes (dict, PropertySet)
            property_set.clear()
            property_set.update(self.property_set)

        if isinstance(result, DAGCircuit):
            result_circuit = dag_to_circuit(result)
        elif result is None:
            result_circuit = circuit.copy()

        if self.property_set["layout"]:
            result_circuit._layout = self.property_set["layout"]
        if self.property_set["clbit_write_latency"] is not None:
            result_circuit._clbit_write_latency = self.property_set["clbit_write_latency"]
        if self.property_set["conditional_latency"] is not None:
            result_circuit._conditional_latency = self.property_set["conditional_latency"]

        return result_circuit


def add_miti_gates_to_circuit(circuit):
    dag = circuit_to_dag(circuit)
    new_dag = DAGCircuit()
    for qreg in dag.qregs.values():
        new_dag.add_qreg(qreg)
    for creg in dag.cregs.values():
        new_dag.add_creg(creg)

    canonical_register = dag.qregs['q']
    trivial_layout = Layout.generate_trivial_layout(canonical_register)
    current_layout = trivial_layout.copy()
    order = current_layout.reorder_bits(new_dag.qubits)

    for i, layer in enumerate(dag.serial_layers()):
        subdag = layer['graph']
        for node in subdag.op_nodes():
            node_qargs = node.qargs
            if len(node.op.params) == 3 and i > 2 * circuit.num_qubits:  # node.name != 'barrier' # i < circuit.num_qubits * 2:  # len(node_qargs) > 1 and np.random.rand() > 1.0:
                mitigate_layer = DAGCircuit()
                mitigate_layer.add_qreg(canonical_register)
                
                for qubit in node_qargs:
                    mitigate_layer.apply_operation_back(IGate(label='miti'), qargs=[qubit], cargs=[])
                
                new_dag.compose(mitigate_layer, qubits=order)
                new_dag.compose(subdag, qubits=order)
                # new_dag.compose(mitigate_layer, qubits=order)
                
                # new_dag.compose(mitigate_layer, qubits=order)
            else:
                new_dag.compose(subdag, qubits=order)

    # mitigate_layer = DAGCircuit()
    # mitigate_layer.add_qreg(canonical_register)
    # for qubit in new_dag.qubits:
    #     mitigate_layer.apply_operation_back(IGate(label='miti'), qargs=[qubit], cargs=[])
    # new_dag.compose(mitigate_layer, qubits=new_dag.qubits)
    new_circuit = dag_to_circuit(new_dag)
    return new_circuit


def add_miti_gates_to_circuit2(circuit):
    dag = circuit_to_dag(circuit)
    new_dag = DAGCircuit()
    for qreg in dag.qregs.values():
        new_dag.add_qreg(qreg)
    for creg in dag.cregs.values():
        new_dag.add_creg(creg)

    canonical_register = dag.qregs['q']
    trivial_layout = Layout.generate_trivial_layout(canonical_register)
    current_layout = trivial_layout.copy()
    order = current_layout.reorder_bits(new_dag.qubits)

    for layer in dag.serial_layers():
        subdag = layer['graph']
        for node in subdag.op_nodes():
            if node.name != 'barrier' and len(node.qargs) > 1 and np.random.rand() > 1.0:
                node_qargs = node.qargs
                mitigate_layer = DAGCircuit()
                mitigate_layer.add_qreg(canonical_register)
                
                for qubit in node_qargs:
                    mitigate_layer.apply_operation_back(IGate(label='miti'), qargs=[qubit], cargs=[])
                
                new_dag.compose(mitigate_layer, qubits=order)
                new_dag.compose(subdag, qubits=order)
                # new_dag.compose(mitigate_layer, qubits=order)
            else:
                new_dag.compose(subdag, qubits=order)
    mitigate_layer = DAGCircuit()
    mitigate_layer.add_qreg(canonical_register)
    for qubit in new_dag.qubits:
        if qubit.index > 1: break
        mitigate_layer.apply_operation_back(IGate(label='miti'), qargs=[qubit], cargs=[])
    new_dag.compose(mitigate_layer, qubits=new_dag.qubits)
    new_circuit = dag_to_circuit(new_dag)

    return new_circuit


if __name__ == '__main__':
    from circuit_lib import random_circuit, DQCp, swaptest
    rand_circuit = random_circuit(4, 10, 2)
    # print(rand_circuit)
    # rand_circuit = swaptest().decompose().decompose()
    new_circuit = add_miti_gates_to_circuit(rand_circuit)
    print(new_circuit)