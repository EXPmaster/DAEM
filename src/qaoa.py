import pickle
import functools
import os
import rustworkx as rx
# from rustworkx.visualization import mpl_draw
# General imports
import numpy as np
from tqdm import tqdm

# Pre-defined ansatz circuit, operator class and visualization tools
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Operator, random_statevector, Statevector
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
# from qiskit.visualization import plot_distribution
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import IGate

from qiskit.providers.aer import AerSimulator

# SciPy minimizer routine
from scipy.optimize import minimize


def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    cost = estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
    return cost


def get_training_circuit(circuit, init_state=None):
    """
    Generate training circuit containing identity gates.
    """
    train_circ = QuantumCircuit(circuit.num_qubits)
    if init_state is not None:
        train_circ.initialize(init_state, range(train_circ.num_qubits))
    for operation, gate_qubits, _ in circuit.data:
        if operation.name in ['u', 'u3']:
            train_circ.append(IGate(), [q.index for q in gate_qubits])
        elif operation.name == 'measure':
            continue
        else:
            train_circ.append(operation, [q.index for q in gate_qubits])
    train_circ.measure_all()
    return train_circ


def get_random_state(num_qubits):
    # circ = QuantumCircuit(num_qubits)
    # for i in range(num_qubits):
    #     if np.random.rand() < 0.5:
    #         circ.h(i)
    # return Statevector(circ)
    pauli_x = np.array([[0., 1.], [1., 0.]])
    xs = functools.reduce(np.kron, [pauli_x] * num_qubits)
    eigvals, eigvecs = np.linalg.eigh(xs)
    eigvec_ones = eigvecs[:, np.where(eigvals == 1)].squeeze()
    dim_eigenspace = eigvec_ones.shape[1]
    while True:
        selected_indices = np.random.choice(dim_eigenspace, size=np.random.randint(1, min(4, dim_eigenspace)), replace=False)
        selected_eigenvecs = eigvec_ones[:, selected_indices].T
        rand_uniform = np.random.rand(selected_eigenvecs.shape[0])
        rand_uniform = np.sqrt(rand_uniform / np.sum(rand_uniform))
        rand_statevector = rand_uniform @ selected_eigenvecs
        yield Statevector(rand_statevector)


def get_random_graphs(num_graphs, num_nodes, p=0.5):
    return [rx.undirected_gnp_random_graph(num_nodes, p) for _ in range(num_graphs)]


def counts_to_probabilities(counts, num_nodes):
    ret_list = np.zeros(2 ** num_nodes)
    for key, val in counts.items():
        ret_list[int(key, 2)] = val
    ret_list /= np.sum(ret_list)
    return ret_list


def train_qaoa(graph, num_nodes):
    estimator = Estimator()
    # sampler = Sampler(options={"shots": int(1e3)})
    # Problem to Hamiltonian operator
    ham_list = []
    for edge in graph.edge_list():
        pauli_list = ['I' for _ in range(num_nodes)]
        for node in edge:
            pauli_list[num_nodes - 1 - node] = 'Z'
        ham_list.append((''.join(pauli_list), 1))
    hamiltonian = SparsePauliOp.from_list(ham_list)
    # QAOA ansatz circuit
    ansatz = QAOAAnsatz(hamiltonian, reps=4)
    # print(ansatz.decompose(reps=3)) # .draw("mpl")
    x0 = 2 * np.pi * np.random.rand(ansatz.num_parameters)
    res = minimize(cost_func, x0, args=(ansatz, hamiltonian, estimator), method="COBYLA")
    # Assign solution parameters to ansatz
    qc = ansatz.assign_parameters(res.x)
    # Add measurements to output circuit
    qc.measure_all()
    # samp_dist = sampler.run(qc, shots=int(1e3)).result().quasi_dists[0]
    return qc


def generate_train_val(num_nodes, num_samples, circuits, noise_model, save_path, shots=1024):
    ideal_backend = AerSimulator()
    noisy_backends = [AerSimulator(noise_model=m) for m in noise_model]
    trainset = []
    valset = []
    for param, circuit in tqdm(enumerate(circuits)):
        for i in range(num_samples):
            init_state = next(get_random_state(num_nodes))
            circuit = transpile(circuit, basis_gates=['cx', 'u', 'id'], optimization_level=0)
            train_circ = get_training_circuit(circuit, init_state=init_state)
            train_circ = transpile(train_circ, basis_gates=['initialize', 'cx', 'u', 'id'], optimization_level=0)
            results_train_ideal = ideal_backend.run(train_circ, shots=shots).result()
            ideal_probs = counts_to_probabilities(results_train_ideal.get_counts(), num_nodes=num_nodes)
            noisy_probs = []
            for noisy_backend in noisy_backends:
                results_train_noisy = noisy_backend.run(train_circ, shots=shots).result()
                noisy_probs.append(counts_to_probabilities(results_train_noisy.get_counts(), num_nodes=num_nodes))
            if i < 70:
                trainset.append([param, [np.eye(2)] * 2, [0, 1], 0.1, noisy_probs, ideal_probs])
            else:
                valset.append([param, [np.eye(2)] * 2, [0, 1], 0.1, noisy_probs, ideal_probs])
        
    with open(os.path.join(save_path, 'trainset.pkl'), 'wb') as f:
        pickle.dump(trainset, f)
    with open(os.path.join(save_path, 'valset.pkl'), 'wb') as f:
        pickle.dump(valset, f)


def generate_test(num_nodes, circuits, noise_model, save_path, shots=1024):
    ideal_backend = AerSimulator()
    noisy_backends = [AerSimulator(noise_model=m) for m in noise_model]
    dataset = []
    for param, circuit in tqdm(enumerate(circuits)):
        circuit = transpile(circuit, basis_gates=['cx', 'u', 'id'], optimization_level=0)
        results_train_ideal = ideal_backend.run(circuit, shots=shots).result()
        ideal_probs = counts_to_probabilities(results_train_ideal.get_counts(), num_nodes=num_nodes)
        noisy_probs = []
        for noisy_backend in noisy_backends:
            results_train_noisy = noisy_backend.run(circuit, shots=shots).result()
            noisy_probs.append(counts_to_probabilities(results_train_noisy.get_counts(), num_nodes=num_nodes))
        dataset.append([param, [np.eye(2)] * 2, [0, 1], 0.1, noisy_probs, ideal_probs])
    
    with open(os.path.join(save_path, 'testset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    # num_nodes = 6
    # num_graphs = 20
    # save_root = '../data_mitigate/qaoa_6q_dep_distr'
    # if not os.path.exists(save_root):
    #     os.makedirs(save_root)
    # # graphs = get_random_graphs(num_graphs, num_nodes, p=0.7)
    # # # The edge syntax is (start, end, weight)

    # # with open(os.path.join(save_root, 'graphs.pkl'), 'wb') as f:
    # #     pickle.dump(graphs, f)

    # noise_models = []
    # for i in np.round(np.linspace(0.015, 0.025, 5), 3):
    #     noise_model = NoiseModel()
    #     error_1 = noise.depolarizing_error(i, 1)
    #     error_2 = noise.depolarizing_error(i, 2)
    #     noise_model.add_all_qubit_quantum_error(error_1, ['u', 'rx', 'ry', 'rz', 'i', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'])
    #     noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    #     noise_models.append(noise_model)
    
    # # print('Training QAOA circuits...')
    # # circuits = [train_qaoa(G, num_nodes) for G in graphs]
    # # with open(os.path.join(save_root, 'circuits.pkl'), 'wb') as f:
    # #     pickle.dump(circuits, f)
    
    # with open(os.path.join(save_root, 'circuits.pkl'), 'rb') as f:
    #     circuits = pickle.load(f)
    # print('Generating datasets...')
    # generate_train_val(num_nodes, 100, circuits, noise_models, save_root, shots=10000)
    # generate_test(num_nodes, circuits, noise_models, save_root, shots=10000)
    with open('../data_mitigate/qaoa_6q_dep_distr/graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)
    from rustworkx.visualization import mpl_draw
    import matplotlib.pyplot as plt
    mpl_draw(graphs[16], node_size=600, linewidths=2, with_labels=True)
    plt.savefig(f'../figures/qaoa_graph.pdf', bbox_inches='tight')
    plt.close()