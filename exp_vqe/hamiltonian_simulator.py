from contextlib import contextmanager, redirect_stderr
import os
import sys
import functools
import numpy as np
import scipy
# import warnings
# warnings.simplefilter("ignore", UserWarning)
import oqupy
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Pauli, SparsePauliOp
from tqdm import tqdm


class HamiltonianSimulator:

    def __init__(self, noise_scale, alpha=0.002, zeta=4, cutoff=5):
        self.noise_scale = noise_scale
        self.alpha = alpha / 2
        self.zeta = zeta
        self.cutoff = cutoff

    def run(self, hamiltonians, verbose=False):
        assert self.noise_scale > 0, 'Noise scale must be positive.'
        bath_correlation = oqupy.PowerLawSD(
            alpha=self.alpha,
            zeta=self.zeta,
            cutoff=self.cutoff,
            cutoff_type='exponential'
        )
        dim = hamiltonians[0].system.to_matrix().shape[0]
        initial_state = np.zeros(dim)
        initial_state[0] = 1.0
        rho = np.outer(initial_state, initial_state.conj())
        # tempo_parameters = oqupy.TempoParameters(dt=0.001, dkmax=3, epsrel=10**(-4))
        for hamiltonian in tqdm(hamiltonians, disable=not verbose, file=sys.__stdout__):
            system = oqupy.System(hamiltonian.system.to_matrix() / self.noise_scale)
            bath = oqupy.Bath(hamiltonian.bath.to_matrix(), bath_correlation)
            with suppress_stderr():
                dynamics = oqupy.tempo_compute(
                    system=system,
                    bath=bath,
                    initial_state=rho,
                    start_time=0.0,
                    end_time=1.0 * self.noise_scale,
                    # parameters=tempo_parameters,
                    progress_type='silent'
                )
            rho = dynamics.states[-1]
        return rho


class AnalyticSimulator:

    def __init__(self, noise_scale, alpha=0.002, zeta=4, cutoff=5):
        self.noise_scale = noise_scale
        self.alpha = alpha
        self.zeta = zeta
        self.cutoff = cutoff
        self.basis_change_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def run(self, hamiltonians, verbose=False):
        assert self.noise_scale > 0, 'Noise scale must be positive.'
        J_omega.__defaults__ = (self.cutoff, self.alpha, self.zeta)
        dim = hamiltonians[0].system.to_matrix().shape[0]
        initial_state = np.zeros(dim)
        initial_state[0] = 1.0
        rho = np.outer(initial_state, initial_state.conj())
        for hamiltonian in tqdm(hamiltonians, disable=not verbose, file=sys.__stdout__):
            system = hamiltonian.system.to_matrix()
            # ideal evolution
            evolution_operator = scipy.linalg.expm(-1j * system)
            rho = evolution_operator @ rho @ evolution_operator.conj().T

            # apply noise
            if isinstance(hamiltonian.system, PauliSumOp):
                pauliops = hamiltonian.system.primitive
            else:
                pauliops = hamiltonian.system.oplist

            for pauliop in pauliops:
                if isinstance(pauliop, SparsePauliOp):
                    assert len(pauliop.paulis) == 1
                    pauli_string = pauliop.paulis[0]
                    parameter = 2 * pauliop.coeffs[0]
                else:
                    pauli_string = pauliop.primitive
                    parameter = 2 * pauliop.coeff
                noise_operator = [1.]
                basis_change = [1.]
                bias = [1.]
                
                for p in pauli_string:
                    p = p.to_label()
                    if p == 'Z':
                        decay_rate = np.exp(-2 * z_decay_fn(self.noise_scale, self.cutoff, self.alpha, self.zeta))
                        decay_mask = np.array([[1., decay_rate], [decay_rate, 1.]])
                        basis_change = np.kron(np.eye(2), basis_change)
                        bias = np.kron(np.zeros((2, 2)), bias)
                    elif p == 'I':
                        decay_mask = np.ones((2, 2))
                        basis_change = np.kron(np.eye(2), basis_change)
                        bias = np.kron(np.zeros((2, 2)), bias)
                    elif p == 'X':
                        decay_rate = np.exp(-zeta_t(self.noise_scale, parameter))
                        energy_change = np.exp(-eta_t(self.noise_scale, parameter))
                        decay_mask = np.array([[energy_change, decay_rate],
                                                [decay_rate, energy_change]])
                        basis_change = np.kron(self.basis_change_matrix, basis_change)
                        bias = np.kron(np.array([[energy_change * xi_t(self.noise_scale, parameter), 0.],
                                                 [0., 1 - 2 * energy_change * xi_t(self.noise_scale, parameter)]]), bias)
                    else:
                        raise ValueError('Invalid Pauli string.')
                    noise_operator = np.kron(decay_mask, noise_operator)
                
                rho = noise_operator * (basis_change @ rho @ basis_change) + bias
                rho = basis_change @ rho @ basis_change
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


@functools.lru_cache(maxsize=2 ** 10, typed=False)
def gamma_omega0_t(t, omega_0):
    f = lambda omega: 2 * J_omega(omega) * np.sin((omega_0 - omega) * t) / (omega_0 - omega)
    return scipy.integrate.quad(f, 0, np.inf)[0]


@functools.lru_cache(maxsize=2 ** 10, typed=False)
def eta_t(t, omega_0):
    f = lambda t: gamma_omega0_t(t, omega_0) + gamma_omega0_t(t, -omega_0)
    return scipy.integrate.quad(f, 0, t)[0]


@functools.lru_cache(maxsize=2 ** 10, typed=False)
def xi_t(t, omega_0):
    f = lambda t: gamma_omega0_t(t, -omega_0) * np.exp(eta_t(t, omega_0))
    return scipy.integrate.quad(f, 0, t)[0]


def zeta_t(t, omega_0):
    return 0.5 * eta_t(t, omega_0)