from itertools import product
import pickle
import numpy as np
import qiskit
from qiskit import Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info.operators import Pauli
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import QasmSimulator
from qiskit.opflow import PauliOp
from qiskit.opflow.gradients import Gradient
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA, SLSQP


class VQETrainer:

    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers

    def get_circuit(self, barriers=True):
        qc = QuantumCircuit(self.num_qubits)
        params = []
        # initial Euler Rotation Layer
        for i in range(self.num_qubits):
            for _ in range(2):  # two new parameters
                params.append(Parameter(f'p{len(params):02}'))
            # rotation with the two new parameters. Don't need the first
            # z rotation
            qc.u(params[-2], params[-1], 0, i)
        if barriers:
            qc.barrier()
        for l in range(self.num_layers):
            # entangling layer
            for i in range(self.num_qubits - 1):
                qc.cnot(i, i + 1)
            if barriers:
                qc.barrier()
            for i in range(self.num_qubits):
                for _ in range(3):
                    params.append(Parameter(f'p{len(params):02}'))
                qc.u(params[-3], params[-2], params[-1], i)
            if barriers:
                qc.barrier()
        return qc

    def get_hamitonian_ising(self, g=None):
        operators = []
        op_str = 'I' * self.num_qubits
        for i in range(self.num_qubits - 1):
            tmp_op = op_str[:i] + 'ZZ' + op_str[i + 2:]
            operators.append(PauliOp(Pauli(tmp_op), -1.0))
        if g is None:
            g = np.random.uniform(-2, 2)
        for i in range(self.num_qubits):
            tmp_op = op_str[:i] + 'X' + op_str[i + 1:]
            operators.append(PauliOp(Pauli(tmp_op), -g))
        hamitonian = sum(operators)
        return hamitonian

    def train(self, circuit, hamitonian, num_iters=1000, num_shots=1, save_path=None):
        # backend = Aer.get_backend('qasm_simulator')
        # qinstance = QuantumInstance(backend=backend,
        #                             shots=num_shots)
        qinstance = QuantumInstance(QasmSimulator(method='matrix_product_state'), shots=num_shots)
        # optimizer = SPSA(maxiter=num_iters)
        optimizer = SLSQP(maxiter=num_iters)
        vqe = VQE(ansatz=circuit,
                  # gradient=Gradient(grad_method='lin_comb'),
                  optimizer=optimizer,
                  quantum_instance=qinstance,
                  include_custom=True
                  )
        result_vqe = vqe.compute_minimum_eigenvalue(operator=hamitonian)
        npme = NumPyMinimumEigensolver()
        result_np = npme.compute_minimum_eigenvalue(operator=hamitonian)
        ref_value = result_np.eigenvalue.real
        print('VQE result: {},\t Calculation result: {}'.format(result_vqe.optimal_value, ref_value))
        if save_path is not None:
            bind_ansatz = vqe.ansatz.bind_parameters(result_vqe.optimal_point)
            with open(save_path, 'wb') as f:
                pickle.dump(bind_ansatz, f)
            print(f'Trained ansatz saved to {save_path}.')
        return vqe.ansatz.bind_parameters(result_vqe.optimal_point)


if __name__ == '__main__':
    trainer = VQETrainer(6, 3)
    circuit = trainer.get_circuit()
    for i, x in enumerate(np.arange(-2.0, 2.0, 0.25)):
        H = trainer.get_hamitonian_ising(x)
        trainer.train(circuit, H, save_path=f'../environments/circuits/vqe_{x}.pkl')



    # H = trainer.get_hamitonian_ising(1.0)
    # # trainer.train(circuit, H, save_path='tmp.pkl')
    # # from Ising_random_ground_state_generation import exact_E
    # # print(exact_E(4, 1.0, 0.3))
    # with open('tmp.pkl', 'rb') as f:
    #     circuit = pickle.load(f)

    # qinstance = QuantumInstance(QasmSimulator(method='matrix_product_state'), shots=1)
    # operator = H.to_matrix()
    # eigvals, U = np.linalg.eigh(operator)
    # circuit.unitary(np.linalg.inv(U), qubits=range(circuit.num_qubits))
    # circuit.measure_all()
    # backend = Aer.get_backend('aer_simulator')
    # shots = 1024
    
    # results = backend.run(circuit, shots=shots).result()
    # counts = results.get_counts()
    # expectation = 0
    # for bitstring, count in counts.items():
    #     expectation += (
    #         eigvals[int(bitstring[0 : circuit.num_qubits], 2)] * count / shots
    #     )
    # print(expectation)