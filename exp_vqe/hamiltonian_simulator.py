from contextlib import contextmanager, redirect_stderr
from abc import ABC, abstractmethod
import os
import sys
import functools
import itertools
import numpy as np
import scipy
import torch
# import warnings
# warnings.simplefilter("ignore", UserWarning)
from qiskit import QuantumCircuit
from qiskit.opflow import PauliSumOp, PauliOp
from qiskit.quantum_info import Pauli, SparsePauliOp
from tqdm import tqdm

from circuit_parser import CircuitParser


class HamiltonianSimulator(ABC):

    def __init__(self, noise_scale, **kwargs):
        self.noise_scale = noise_scale
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self, inputs, init_rho=None, simulate_identity=False, verbose=False):
        assert self.noise_scale > 0, 'Noise scale must be positive.'
        if isinstance(inputs, QuantumCircuit):
            hamiltonians = CircuitParser().parse(inputs)
        else:
            hamiltonians = inputs
        self.init_function()
        if init_rho is None:
            dim = hamiltonians[0].system.to_matrix().shape[0]
            initial_state = np.zeros(dim)
            initial_state[0] = 1.0
            rho = np.outer(initial_state, initial_state.conj())
        else:
            rho = torch.from_numpy(init_rho).cuda()
        idx = 0
        for hamiltonian in tqdm(hamiltonians, disable=not verbose, file=sys.__stdout__):
            system = hamiltonian.system.to_matrix()
            # apply noise
            rho = self.forward_function(rho, hamiltonian)
            # ideal evolution
            if not simulate_identity or (simulate_identity and len(hamiltonian.system) == 3):
            # if init_rho is None or (init_rho is not None and idx < len(hamiltonians) - 3):
                evolution_operator = torch.matrix_exp(torch.from_numpy(-1j * system).cuda())
                rho = evolution_operator @ rho @ evolution_operator.conj().T
            idx += 1
        return rho.cpu().numpy()

    def init_function(self):
        """ Initialization run before simulation."""
        pass

    @abstractmethod
    def forward_function(self, rho, hamiltonian):
        """ Apply noise to the system."""


class IdealSimulator(HamiltonianSimulator):

    def __init__(self):
        super().__init__(1.0)

    def forward_function(self, rho, hamiltonian):
        return rho


class NonMarkovianSimulator(HamiltonianSimulator):
    """Simulate the non-Markovian dephasing noise.
    Note, the lamb shift is not considered for simplicity."""

    def __init__(self, noise_scale, alpha=0.0005, zeta=6, cutoff=5):
        super().__init__(noise_scale, alpha=alpha, zeta=zeta, cutoff=cutoff)
        self.basis_change_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def init_function(self):
        J_omega.__defaults__ = (self.cutoff, self.alpha, self.zeta)
    
    def forward_function(self, rho, hamiltonian):
        pauliops = hamiltonian.system.to_list()

        for pauli_string, coeff in pauliops:
            parameter = 2 * coeff.real
            noise_operator = [1.]
            basis_change = [1.]
            bias = [1.]
            
            for p in pauli_string:
                if p == 'Z':
                    decay_rate = np.exp(-2 * z_decay_fn(self.noise_scale * abs(parameter) / (2 * np.pi), self.cutoff, self.alpha, self.zeta))
                    decay_mask = np.array([[1., decay_rate], [decay_rate, 1.]])
                    basis_change = np.kron(basis_change, np.eye(2))
                    bias = np.kron(bias, np.zeros((2, 2)))
                elif p == 'I':
                    decay_mask = np.ones((2, 2))
                    basis_change = np.kron(basis_change, np.eye(2))
                    bias = np.kron(bias, np.zeros((2, 2)))
                elif p == 'X':
                    decay_rate = np.exp(-zeta_t(self.noise_scale, parameter))
                    energy_change = np.exp(-eta_t(self.noise_scale, parameter))
                    decay_mask = np.array([[energy_change, decay_rate],
                                            [decay_rate, energy_change]])
                    basis_change = np.kron(basis_change, self.basis_change_matrix)
                    bias = np.kron(bias, np.array([[energy_change * xi_t(self.noise_scale, parameter), 0.],
                                                  [0., 1 - 2 * energy_change * xi_t(self.noise_scale, parameter)]]))
                else:
                    raise ValueError('Invalid Pauli string.')
                noise_operator = np.kron(noise_operator, decay_mask)
            
            rho = noise_operator * (basis_change @ rho @ basis_change) + bias
            rho = basis_change @ rho @ basis_change
        return rho


class DepolarizeSimulator(HamiltonianSimulator):

    def __init__(self, noise_scale):
        super().__init__(noise_scale)

    def forward_function(self, rho, hamiltonian):
        pauliops = hamiltonian.system.to_list()

        for pauli_string, _ in pauliops:
            noise_operator = []
            
            for p in pauli_string:
                if p == 'I':
                    noise_operator.append(['I'])
                else:
                    noise_operator.append(['I', 'X', 'Y', 'Z'])
            noise_strings = itertools.product(*noise_operator)
            noise_operators = [Pauli(''.join(op)) for op in noise_strings]
            probabilities = np.ones(len(noise_operators)) * self.noise_scale / len(noise_operators)
            probabilities[0] = 1 - (len(noise_operators) - 1) * self.noise_scale / len(noise_operators)
            rho = sum([prob * op.to_matrix() @ rho @ op.to_matrix()
                        for prob, op in zip(probabilities, noise_operators)])
        return rho


class DephaseSimulator(HamiltonianSimulator):

    def __init__(self, noise_scale, alpha=1.0):
        super().__init__(noise_scale, alpha=alpha)

    def forward_function(self, rho, hamiltonian):
        pauliops = hamiltonian.system.to_list()

        for pauli_string, _ in pauliops:
            noise_operator = torch.tensor([1.]).cuda()
            
            for p in pauli_string:
                if p == 'I':
                    decay_mask = torch.ones((2, 2)).cuda()
                else:
                    decay_rate = np.exp(-2 * self.alpha * self.noise_scale)
                    decay_mask = torch.tensor([[1.0, decay_rate],
                                            [decay_rate, 1.0]]).cuda()
                noise_operator = torch.kron(noise_operator, decay_mask)

            rho = noise_operator * rho
        return rho


class AmpdampSimulator(HamiltonianSimulator):

    def __init__(self, noise_scale):
        super().__init__(noise_scale)
        self.noise_map = {
            'I': np.eye(2),
            'E0': np.diag([1., np.sqrt(1 - noise_scale)]),
            'E1': np.array([[0., np.sqrt(noise_scale)], [0., 0.]])
        }

    def forward_function(self, rho, hamiltonian):
        pauliops = hamiltonian.system.to_list()
        for pauli_string, _ in pauliops:
            noise_operator = []

            for p in pauli_string:
                if p == 'I':
                    noise_operator.append(['I'])
                else:
                    noise_operator.append(['E0', 'E1'])
            noise_strings = itertools.product(*noise_operator)
            noise_operators = [functools.reduce(np.kron, [self.noise_map[x] for x in op]) for op in noise_strings]
            rho = sum([op @ rho @ op.conj().T
                        for op in noise_operators])
        return rho


@contextmanager
def suppress_stderr():
    """A context manager that redirects stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err:
            yield err


def z_decay_fn(t, omega_c=5, alpha=0.002, zeta=4):
    return 2 * alpha * scipy.special.gamma(zeta - 1) * \
            (1 - (np.cos((zeta - 1) * np.arctan(omega_c * t))) / \
            (1 + (omega_c * t) ** 2) ** ((zeta - 1) / 2))


def J_omega(omega, omega_c=5, alpha=0.002, zeta=4):
    return alpha * omega_c ** (1 - zeta) * omega ** zeta * np.exp(-omega / omega_c)


@functools.lru_cache(maxsize=2 ** 20, typed=False)
def gamma_omega0_t(t, omega_0):
    f = lambda omega: 2 * J_omega(omega) * np.sin((omega_0 - omega) * t) / (omega_0 - omega)
    return scipy.integrate.quad(f, 0, np.inf)[0]


@functools.lru_cache(maxsize=2 ** 20, typed=False)
def eta_t(t, omega_0):
    f = lambda t: gamma_omega0_t(t, omega_0) + gamma_omega0_t(t, -omega_0)
    return scipy.integrate.quad(f, 0, t)[0]


@functools.lru_cache(maxsize=2 ** 20, typed=False)
def xi_t(t, omega_0):
    f = lambda t: gamma_omega0_t(t, -omega_0) * np.exp(eta_t(t, omega_0))
    return scipy.integrate.quad(f, 0, t)[0]


def zeta_t(t, omega_0):
    return 0.5 * eta_t(t, omega_0)


def cnots(dim):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    circ = QuantumCircuit(int(np.log2(dim)))
    for i in range(circ.num_qubits - 1):
        circ.cx(i, i + 1)
    for i in range(circ.num_qubits - 1):
        circ.cx(i, i + 1)
    return Operator(circ).data


if __name__ == '__main__':
    import pickle
    from qiskit.quantum_info import random_density_matrix, DensityMatrix, state_fidelity, Statevector, Operator, Pauli
    from qiskit.opflow import PauliOp
    with open('../environments/circuits/autoencoder_6l/ae_1.pkl', 'rb') as f:
        circuit = pickle.load(f)
    # circuit = transpile(circ, basis_gates=['cx', 'u'])

    parser = CircuitParser()
    hs = parser.construct_train(circuit, train_num=1)
    dim = hs[0][0].system.to_matrix().shape[0]
    rho = random_density_matrix(2 ** 6).data
    backend = IdealSimulator()
    op = cnots(dim)
    result = backend.run(hs[0], init_rho=rho)
    rho_out = op @ rho @ op.conj().T
    print(state_fidelity(DensityMatrix(rho_out), DensityMatrix(result)))

    state = Statevector(circuit)
    hs = parser.parse(circuit)
    result = backend.run(hs)
    print(state_fidelity(DensityMatrix(result), state))
