import functools
import oqupy
import numpy as np
import scipy
import qiskit
import matplotlib.pyplot as plt


def matrix_sqrt(matrix):
    """Compute the square root of a matrix."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.diag(np.sqrt(eigvals))
    return np.dot(np.dot(eigvecs, eigvals), eigvecs.conj().T)


def test_multi_qubit_evolution():
    sigma_x = oqupy.operators.sigma("x")
    sigma_y = oqupy.operators.sigma("y")
    sigma_z = oqupy.operators.sigma("z")
    zero_state = np.array([0.35, 0.65])
    one_state = np.array([0.6, 0.4])
    zero_state = zero_state / np.linalg.norm(zero_state)
    one_state = one_state / np.linalg.norm(one_state)
    initial_state = np.kron(zero_state, one_state)
    initial_rho = np.outer(initial_state, initial_state.conj())

    rz_gate = np.array([[np.exp(-1j * np.pi / 4), 0], [0, np.exp(1j * np.pi / 4)]])
    rz_gate = np.kron(rz_gate, rz_gate)
    out_rho = rz_gate @ initial_state

    # up_density_matrix = np.kron(oqupy.operators.spin_dm("z+"), oqupy.operators.spin_dm("z+"))
    # down_density_matrix = oqupy.operators.spin_dm("z-")
    Omega = np.pi
    omega_cutoff = 3.0
    alpha = 0.0125 / 2

    correlations = oqupy.PowerLawSD(alpha=alpha,
                                    zeta=6,
                                    cutoff=omega_cutoff,
                                    cutoff_type='exponential')
    sigma_zs = np.kron(sigma_z, np.eye(2)) + np.kron(np.eye(2), sigma_z)  # functools.reduce(np.kron, [sigma_z] * 2)
    bath = oqupy.Bath(sigma_zs, correlations)
    # tempo_parameters = oqupy.TempoParameters(dt=0.005, dkmax=30, epsrel=10**(-4))

    observable = np.kron(sigma_z, sigma_y)

    # system_h = 0.25 * Omega * (np.kron(sigma_z, np.eye(2)) + np.kron(np.eye(2), sigma_z))
    # system = oqupy.System(system_h)
    # dynamics = oqupy.tempo_compute(system=system,
    #                             bath=bath,
    #                             initial_state=initial_rho,
    #                             start_time=0.0,
    #                             end_time=1.0,
    #                             parameters=tempo_parameters)
    # output = dynamics.states[-1]
    # print(output)
    # print(np.outer(out_rho, out_rho.conj()))
    # print(out_rho.conj() @ output @ out_rho)
    # assert False
    
    print('ideal:', out_rho.conj() @ observable @ out_rho)
    expectations = []
    ks = np.linspace(0.05, 0.4, 20)
    for k in ks:
        system_h = 0.25 * Omega * (np.kron(sigma_z, np.eye(2)) + np.kron(np.eye(2), sigma_z)) / k
        system = oqupy.System(system_h)
        # bath = oqupy.Bath(sigma_zs, correlations)
        # tempo_parameters = oqupy.tempo.guess_tempo_parameters(bath, 0, 1.0 * k, system)
        dynamics = oqupy.tempo_compute(system=system,
                                    bath=bath,
                                    initial_state=initial_rho,
                                    start_time=0.0,
                                    end_time=1.0 * k,
                                    # parameters=tempo_parameters,
                                    progress_type='silent')
        output = dynamics.states[-1]
        expectations.append(np.trace(observable @ output).real)
    
    print(expectations)
    fig = plt.figure()
    plt.plot(ks, expectations)
    plt.xlabel('Evolution time')
    plt.ylabel('Expectation value')
    plt.savefig('../imgs/results_diff_scales_test.png')


def test_cnot_evolution():
    sigma_x = oqupy.operators.sigma("x")
    sigma_y = oqupy.operators.sigma("y")
    sigma_z = oqupy.operators.sigma("z")
    plus_state = np.array([1, 1]) / np.sqrt(2)
    zero_state = np.array([1., 0])
    one_state = np.array([0, 1.])
    initial_state = np.kron(plus_state, zero_state)
    initial_rho = np.outer(initial_state, initial_state.conj())

    cnot_gate = []
    out_rho = (np.kron(zero_state, zero_state) + np.kron(one_state, one_state)) / np.sqrt(2)

    Omega = np.pi
    omega_cutoff = 5.0
    alpha = 0.01

    correlations = oqupy.PowerLawSD(alpha=alpha,
                                    zeta=1,
                                    cutoff=omega_cutoff,
                                    cutoff_type='exponential')
    sigma_zs = np.kron(sigma_z, np.eye(2)) + np.kron(np.eye(2), sigma_z)  # functools.reduce(np.kron, [sigma_z] * 2)
    bath = oqupy.Bath(sigma_zs, correlations)
    # tempo_parameters = oqupy.TempoParameters(dt=0.01, dkmax=3.0, epsrel=10**(-4))

    observable = np.kron(sigma_x, sigma_x)
    
    k = 0.3
    system_h = 0.25 * Omega * (-np.kron(sigma_z, np.eye(2)) + np.kron(sigma_z, sigma_x) - np.kron(np.eye(2), sigma_x)) / k
    system = oqupy.System(system_h)
    dynamics = oqupy.tempo_compute(system=system,
                                bath=bath,
                                initial_state=initial_rho,
                                start_time=0.0,
                                end_time=1.0 * k,
                                # parameters=tempo_parameters
                                )
    output = dynamics.states[-1]
    print(output)
    print(np.outer(out_rho, out_rho.conj()))
    print(out_rho.conj() @ output @ out_rho)


def check_correctness_single_qubit():
    sigma_x = oqupy.operators.sigma("x")
    sigma_y = oqupy.operators.sigma("y")
    sigma_z = oqupy.operators.sigma("z")

    up_density_matrix = oqupy.operators.spin_dm("z+")
    initial_rho = qiskit.quantum_info.random_density_matrix(2, seed=256).data
    # initial_rho = up_density_matrix
    omega_cutoff = 5.0
    alpha = 0.3
    zeta = 2

    correlations = oqupy.PowerLawSD(alpha=alpha / 2,
                                    zeta=zeta,
                                    cutoff=omega_cutoff,
                                    cutoff_type='exponential',
                                    temperature=0.0)

    bath = oqupy.Bath(sigma_z, correlations)
    # tempo_parameters = oqupy.TempoParameters(dt=0.1, dkmax=3, epsrel=10**(-4))
    observable = sigma_x
    system_h = 0.5 * sigma_z  # np.eye(2)
    system = oqupy.System(system_h)
    dynamics = oqupy.tempo_compute(system=system,
                                bath=bath,
                                initial_state=initial_rho,
                                start_time=0.0,
                                end_time=3.0,
                                # parameters=tempo_parameters
                                )
    ts, exp_simulate = dynamics.expectations(observable, real=True)
    decay_fn = lambda alpha, s, omega_c, t: 2 * alpha * scipy.special.gamma(s - 1) * \
                                            (1 - (np.cos((s - 1) * np.arctan(omega_c * t))) / \
                                            (1 + (omega_c * t) ** 2) ** ((s - 1) / 2))

    exp_analytic = []
    for t in ts:
        decay_rate = np.exp(-2 * decay_fn(alpha, zeta, omega_cutoff, t) - 1j * t)
        decay_mask = np.array([[1, decay_rate], [decay_rate.conj(), 1]])
        decoherent_rho = decay_mask * initial_rho
        exp_analytic.append(np.trace(observable @ decoherent_rho).real)
        
    # print(exp_analytic)
    # print(exp_simulate)
    fig = plt.figure()
    plt.plot(ts, exp_simulate, label='Simulated')
    plt.plot(ts, exp_analytic, label='Analytic')
    plt.xlabel('Time')
    plt.ylabel('Expectation value')
    plt.legend()
    plt.savefig('../imgs/single_qubit_decay.png')
    

if __name__ == '__main__':
    # test_multi_qubit_evolution()
    # test_cnot_evolution()
    check_correctness_single_qubit()
    