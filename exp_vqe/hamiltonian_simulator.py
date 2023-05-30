from contextlib import contextmanager, redirect_stderr
import os
import sys
import functools
import numpy as np
# import warnings
# warnings.simplefilter("ignore", UserWarning)
import oqupy
from tqdm import tqdm


@contextmanager
def suppress_stderr():
    """A context manager that redirects stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err:
            yield err


class HamiltonianSimulator:

    def __init__(self, noise_scale, alpha=0.00625, zeta=6, cutoff=3):
        self.noise_scale = noise_scale
        self.bath_correlation = oqupy.PowerLawSD(
            alpha=alpha,
            zeta=1,
            cutoff=cutoff,
            cutoff_type='exponential'
        )

    def run(self, hamiltonians, verbose=False):
        dim = hamiltonians[0].system.to_matrix().shape[0]
        initial_state = np.zeros(dim)
        initial_state[0] = 1.0
        rho = np.outer(initial_state, initial_state.conj())
        
        for hamiltonian in tqdm(hamiltonians, disable=not verbose, file=sys.__stdout__):
            system = oqupy.System(hamiltonian.system.to_matrix() / self.noise_scale)
            bath = oqupy.Bath(hamiltonian.bath.to_matrix(), self.bath_correlation)
            with suppress_stderr():
                dynamics = oqupy.tempo_compute(
                    system=system,
                    bath=bath,
                    initial_state=rho,
                    start_time=0.0,
                    end_time=1.0 * self.noise_scale,
                    # parameters=tempo_parameters
                    progress_type='silent'
                )
            rho = dynamics.states[-1]
        return rho