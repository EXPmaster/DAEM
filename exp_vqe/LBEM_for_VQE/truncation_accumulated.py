from util import *
from expval_calc_q_optim import *
from generate_training_set import *

from qiskit.utils import QuantumInstance 
import qiskit.providers.aer.noise as noise
from qiskit import Aer

def experiment(trunc_T, trunc_P, em_instance, ef_instance, angles):
    group_pauli_op, [ansatz,num_par_gates] = main('LiH', 1.4, 6, 2, 'num_particle_preserving')
    group_pauli_op = group_pauli_op['grouped_paulis']
    print('Ansatz & qubit hamiltonian created')
    circuit_list = get_circuits_dict(ansatz, trunc_T, trunc_P, num_par_gates)
    print('Training circuits generated')
    com_ef, com_em = expval_calc(group_pauli_op, circuit_list, em_instance, ef_instance)
    print('All expectation values calculated')
    q = q_optimize(group_pauli_op, circuit_list, com_em, com_ef)
    print('Optimum q(P) found')
    ef_expval, em_expval, n_expval = test(ansatz, angles, group_pauli_op, q, ef_instance, em_instance)
    return ef_expval, em_expval, n_expval

# Error probabilities
prob_1 = 0.01  # 1-qubit gate
prob_2 = 0.1   # 2-qubit gate

# Depolarizing quantum errors
error_1 = noise.depolarizing_error(prob_1, 1)
error_2 = noise.depolarizing_error(prob_2, 2)

# Add errors to noise model
noise_model = noise.NoiseModel()
noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

qasm_backend = Aer.get_backend('qasm_simulator')
em_instance = QuantumInstance(backend = qasm_backend, noise_model= noise_model, shots = 10000)
sv_backend = Aer.get_backend('aer_simulator_statevector')
ef_instance = QuantumInstance(backend = sv_backend)

seed = 100
np.random.seed(seed)

P = [2, 5, 10, 20, 50, 100, 200]
angles = 2*np.pi*np.random.random(size = 8)

efs = []
ems = []
ns = []

group_pauli_op, [ansatz,num_par_gates] = main('LiH', 1.4, 6, 2, 'num_particle_preserving')
total_trunc_T, total_trunc_P = truncate_training_set(num_par_gates, P[-1], 3*P[-1], s = seed)

filename = './truncation_experiment.txt'
for p in P:
    ef_expval, em_expval, n_expval = experiment(total_trunc_T[:3*p], total_trunc_P[:p], em_instance, ef_instance, angles)
    results = 'p: {} | ef {}, em {}, noisy {}'.format(p, ef_expval, em_expval, n_expval)
    print(results)
    efs.append(ef_expval)
    ems.append(em_expval)
    ns.append(n_expval)
    with open(filename, 'a') as of:
        of.write(results + '\n')
