import pickle
import functools
import numpy as np
from qiskit import Aer, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliOp
from qiskit.utils import QuantumInstance
from qiskit.circuit import ParameterVector
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from LBEM_for_VQE.util import main_fn
from LBEM_for_VQE.expval_calc_q_optim import truncate_training_set, expval_calc, q_optimize, test
from LBEM_for_VQE.generate_training_set import get_circuits_dict, insert_pauli

from ibmq_circuit_transformer import TransformToClifford, TransformCircWithIndex, add_miti_gates_to_circuit2


class LBEMTrainer:

    def __init__(self, noise_model):
        self.em_instance = QuantumInstance(backend=AerSimulator(noise_model=noise_model), shots=10000)
        sv_backend = Aer.get_backend('aer_simulator_statevector')
        self.ef_instance = QuantumInstance(backend=sv_backend)
        self.group_pauli_op = None
        self.q = None

    def fit(self, circuit, num_train=10, num_pauli=100):
        self.group_pauli_op, [ansatz,num_par_gates] = main_fn('H2', 0.774, 6, 3, 'custom')
        trunc_T, trunc_P = truncate_training_set(num_par_gates, num_pauli, num_train, exhaustive=False)
        print(len(trunc_T), len(trunc_P))
        circuit_list = get_circuits_dict(ansatz, trunc_T, trunc_P, num_par_gates)
        print('Training circuits generated')
        print(len(circuit_list))
        com_ef, com_em = expval_calc(self.group_pauli_op, circuit_list, self.em_instance, self.ef_instance)
        print('All expectation values calculated')
        self.q = q_optimize(self.group_pauli_op, circuit_list, com_em, com_ef)
        print('q optimized')
        with open('q_data.pkl', 'wb') as f:
            pickle.dump((self.q, self.group_pauli_op), f)
    
    def predict(self, circuit):
        if self.q is None:
            try:
                with open('q_data.pkl', 'rb') as f:
                    self.q, self.group_pauli_op = pickle.load(f)
            except Exception as e:
                raise e
        ef_expval, em_expval, n_expval = test(circuit, self.group_pauli_op, self.q, self.ef_instance, self.em_instance)
        return em_expval


class LBEMTrainer2:

    def __init__(self, noise_model):
        self.noisy_backend = AerSimulator(noise_model=noise_model)
        self.ideal_backend = AerSimulator()
        self.group_pauli_op = None
        self.clifford_tsfm = TransformToClifford()
        self.reg = None

    def fit(self, circuit, observable, num_train=1000):
        noisy_data = []
        ground_truth = []
        num = 0
        pbar = tqdm(total=num_train)
        while num < num_train:
            clifford_circuit = self.clifford_tsfm(circuit)
            gt = self.simulate_ideal(clifford_circuit, observable)
            if abs(gt) < 0.3: continue
            ground_truth.append(gt)
            clifford_circuit_with_miti_gates = add_miti_gates_to_circuit2(clifford_circuit)
            noisy_data.append(self.noisy_measure(clifford_circuit_with_miti_gates, observable))
            num += 1
            pbar.update(1)
        noisy_data = np.array(noisy_data)
        ground_truth = np.array(ground_truth).reshape(-1, 1)

        self.reg = LinearRegression()
        self.reg.fit(noisy_data, ground_truth)
        with open('q_data.pkl', 'wb') as f:
            pickle.dump(self.reg, f)
    
    def predict(self, circuit, observable):
        if self.reg is None:
            try:
                with open('q_data.pkl', 'rb') as f:
                    self.reg = pickle.load(f)
            except Exception as e:
                raise e
        circuit = add_miti_gates_to_circuit2(circuit)
        noisy_data = self.noisy_measure(circuit, observable)
        noisy_data = np.array(noisy_data).reshape(1, -1)
        em_result = self.reg.predict(noisy_data)
        return em_result[0][0]
    
    def simulate_ideal(self, circuit, observable):
        state_vector = Statevector(circuit)
        return state_vector.expectation_value(observable).real
    
    def noisy_measure(self, miti_circuit, observable, shots_per_sim=100):
        """ Build mitigation circuits and measure them. """
        tsfm = TransformCircWithIndex()
        miti_circuit.save_density_matrix()
        num_miti_gates = self.count_mitigate_gates(miti_circuit)
        table_size = len(tsfm.basis_ops) ** num_miti_gates
        # print(mitigation_circuit)
        # print(table_size)
        results = []
        for i in range(table_size):
            circuit = tsfm(miti_circuit, i)
            t_circuit = transpile(circuit, self.noisy_backend)
            result_noisy = self.noisy_backend.run(t_circuit, shots=shots_per_sim).result()
            density_matrix = result_noisy.data()['density_matrix']
            results.append(density_matrix.expectation_value(observable).real)

        return np.array(results)
    
    def count_mitigate_gates(self, circuit):
        num = 0
        for item in circuit:
            num += item[0].label == 'miti'
        return num

if __name__ == '__main__':
    # from circuit_lib import random_circuit
    # circuit = random_circuit(6, 2, 2)

    noise_model = NoiseModel()
    error_1 = noise.depolarizing_error(0.01, 1)  # single qubit gates
    error_2 = noise.depolarizing_error(0.01, 2)
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'i', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cy', 'cz', 'ch', 'crz', 'swap', 'cu1', 'cu3', 'rzz'])

    with open('../environments/circuits_test/vqe_1.0.pkl', 'rb') as f:
        circuit = pickle.load(f)

    paulis = [Pauli(x).to_matrix() for x in ('I', 'X', 'Y', 'Z')]
    rand_obs = [paulis[-1] for _ in range(2)]  # Pauli ZZ
    # rand_idx = np.random.randint(num_qubits - 1)
    rand_idx = 0
    selected_qubits = list(range(rand_idx, rand_idx + 2))
    obs = [np.eye(2) for i in range(rand_idx)] + rand_obs +\
            [np.eye(2) for i in range(rand_idx + len(selected_qubits), 6)]
    obs_kron = functools.reduce(np.kron, obs[::-1])
    trainer = LBEMTrainer2(noise_model)
    trainer.fit(circuit, obs_kron, 1200)
    print(trainer.predict(circuit, obs_kron))
    
    state_vector = Statevector(circuit)
    exp = state_vector.expectation_value(obs_kron).real
    print(exp)
    
