import cirq
import cirq.ops as ops
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import TransformationPass


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
        from qiskit.converters import circuit_to_dag, dag_to_circuit
        from qiskit.dagcircuit.dagcircuit import DAGCircuit

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
            ops.I, ops.X, ops.Y, ops.Z,
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
                replacement = QuantumCircuit(1)
                replacement.unitary(cirq.unitary(gate), 0, label=str(gate))

                # replace the node with our new decomposition
                dag.substitute_node_with_dag(node, circuit_to_dag(replacement))
                idx //= len(self.basis_ops)

        return dag
    
    def __call__(self, circuit, idx, property_set=None):
        from qiskit.converters import circuit_to_dag, dag_to_circuit
        from qiskit.dagcircuit.dagcircuit import DAGCircuit

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