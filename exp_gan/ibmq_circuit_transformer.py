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
            IGate(), XGate(), YGate(), ZGate()
            # ops.I, ops.X, ops.Y, ops.Z,
            # GateRX(), GateRY(), GateRZ(), GateRYZ(),
            # GateRZX(), GateRXY(), GatePiX(), GatePiY(),
            # GatePiZ(), GatePiYZ(), GatePiZX(), GatePiXY()
        ]

    def run(self, dag, idx):
        """Run the pass."""
        # iterate over all operations
        for node in dag.op_nodes():
            # if we hit a RYY or RZZ gate replace it
            if node.op.name == 'id':
                cur_idx = idx % len(self.basis_ops)
                gate = self.basis_ops[cur_idx]
                # calculate the replacement
                qr = QuantumRegister(1)
                replacement = QuantumCircuit(qr)
                replacement.append(gate, qr)

                # replace the node with our new decomposition
                dag.substitute_node_with_dag(node, circuit_to_dag(replacement))
                idx //= len(self.basis_ops)

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

    for layer in dag.serial_layers():
        subdag = layer['graph']
        for node in subdag.op_nodes():
            node_qargs = node.qargs
            if len(node_qargs) > 1 and np.random.rand() > 0.8:
                mitigate_layer = DAGCircuit()
                mitigate_layer.add_qreg(canonical_register)
                
                for qubit in node_qargs:
                    mitigate_layer.apply_operation_back(IGate(), qargs=[qubit], cargs=[])
                
                new_dag.compose(mitigate_layer, qubits=order)
                new_dag.compose(subdag, qubits=order)
                # new_dag.compose(mitigate_layer, qubits=order)
            else:
                new_dag.compose(subdag, qubits=order)

    mitigate_layer = DAGCircuit()
    mitigate_layer.add_qreg(canonical_register)
    for qubit in new_dag.qubits:
        mitigate_layer.apply_operation_back(IGate(), qargs=[qubit], cargs=[])
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