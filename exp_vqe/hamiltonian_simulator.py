import functools
import oqupy
import numpy as np

import warnings
warnings.filterwarnings('ignore')


class HamiltonianSimulator:

    def __init__(self, num_qubits, alpha=0.00625, zeta=6, cutoff=3):
        self.num_qubits = num_qubits
        self.bath_correlation = oqupy.PowerLawSD(
            alpha=alpha,
            zeta=1,
            cutoff=cutoff,
            cutoff_type='exponential'
        )

    def run(self, noise_scale, hamiltonians):
        initial_state = np.zeros(2 ** self.num_qubits)
        initial_state[0] = 1.0
        rho = np.outer(initial_state, initial_state.conj())
        for hamiltonian in hamiltonians:
            system = oqupy.System(hamiltonian.system / noise_scale)
            bath = oqupy.Bath(hamiltonian.bath, self.bath_correlation)
            dynamics = oqupy.tempo_compute(
                system=system,
                bath=bath,
                initial_state=rho,
                start_time=0.0,
                end_time=1.0 * noise_scale,
                # parameters=tempo_parameters
            )
            rho = dynamics.states[-1]
        return rho