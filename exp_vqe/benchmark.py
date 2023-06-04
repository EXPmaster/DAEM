import argparse
import random
import pickle
import functools
import os
from vqe import VQETrainer
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from qiskit import Aer, QuantumCircuit, transpile, execute
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import DensityMatrix, state_fidelity
from qiskit.opflow import PauliOp
from tqdm import tqdm
from torchquantum import switch_little_big_endian_state

from model import *
from utils import AverageMeter, abs_deviation, gen_rand_obs
from my_envs import IBMQEnv
from datasets import MitigateDataset, StateGenerator
from cdr_trainer import CDRTrainer
from lbem_trainer import LBEMTrainer2
from zne_trainer import ZNETrainer


@torch.no_grad()
def evaluation():
    paulis = [Pauli(x).to_matrix() for x in ('I', 'X', 'Y', 'Z')]

    backend = Aer.get_backend('aer_simulator')
    noise_model = NoiseModel()
    error_1 = noise.depolarizing_error(0.01, 1)  # single qubit gates
    error_2 = noise.depolarizing_error(0.01, 2)
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'i', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cy', 'cz', 'ch', 'crz', 'swap', 'cu1', 'cu3', 'rzz'])
    noise_backend = AerSimulator(noise_model=noise_model)

    eval_results = []
    # load GAN model
    ckpt = torch.load(args.weight_path, map_location=args.device)
    model_g = Generator(args.num_mitigates).to(args.device)
    model_s = SurrogateModel(args.num_mitigates).to(args.device)
    model_g.load_state_dict(ckpt['model_g'])
    model_s.load_state_dict(ckpt['model_s'])
    model_g.eval()
    model_s.eval()

    # rand_obs = [gen_rand_obs() for i in range(args.num_obs)]
    rand_obs = [paulis[-1] for _ in range(args.num_obs)]  # Pauli ZZ
    # rand_idx = np.random.randint(num_qubits - 1)
    rand_idx = 0
    selected_qubits = list(range(rand_idx, rand_idx + args.num_obs))
    obs = [np.eye(2) for i in range(rand_idx)] + rand_obs +\
            [np.eye(2) for i in range(rand_idx + len(selected_qubits), 6)]
    obs_kron = functools.reduce(np.kron, obs[::-1])
    obsvb = np.array(rand_obs)
    # train CDR
    with open('../environments/circuits_test/vqe_0.1.pkl', 'rb') as f:
        circuit = pickle.load(f)
    cdr_model = CDRTrainer(noise_model)
    cdr_model.fit(circuit, PauliOp(Pauli('IIIIZZ')))
    # ZNE
    zne_model = ZNETrainer()
    # LBEM
    lbem_model = LBEMTrainer2(noise_model)

    for circ_name in tqdm(os.listdir(args.test_root)):
        param = float(circ_name.replace('.pkl', '').split('_')[-1])
        if param < 0.6 or param > 1.35: continue
        circ_path = os.path.join(args.test_root, circ_name)
        with open(circ_path, 'rb') as f:
            circuit = pickle.load(f)
        
        circuit_ideal = circuit.copy()
        circuit_ideal.save_statevector()
        results = backend.run(transpile(circuit_ideal, backend, optimization_level=0)).result()
        state_vec = results.get_statevector(circuit)

        circuit_noisy = circuit.copy()
        circuit_noisy.save_density_matrix()
        results = noise_backend.run(transpile(circuit_noisy, noise_backend, optimization_level=0)).result()
        density_matrix = results.data()['density_matrix']

        num_qubits = circuit.num_qubits
        mitigated_diff_gan = []
        raw_diff = []
        mitigated_diff_cdr = []
        mitigated_diff_zne = []
        mitigated_diff_lbem = []

        for i in range(args.test_num):
            
            meas_ideal = state_vec.expectation_value(obs_kron).real
            meas_noisy = density_matrix.expectation_value(obs_kron).real
            raw_diff.append(abs(meas_ideal - meas_noisy))
            
            # GAN prediction
            obs = torch.tensor(obsvb, dtype=torch.cfloat)[None].to(args.device)
            params = torch.FloatTensor([param])[None].to(args.device)
            pos = torch.tensor(selected_qubits)[None].to(args.device)
            prs = model_g(params, obs, pos)
            predicts = model_s(params, prs, obs, pos).item()
            mitigated_diff_gan.append(abs(meas_ideal - predicts))
            
            # CDR prediction
            cdr_predicts = cdr_model.predict(np.array(meas_noisy).reshape(-1, 1))
            mitigated_diff_cdr.append(abs(cdr_predicts - meas_ideal))

            # ZNE prediction
            zne_predicts = zne_model.fit_and_predict(circuit, PauliOp(Pauli('IIIIZZ')))
            mitigated_diff_zne.append(abs(zne_predicts - meas_ideal))

            # LBEM prediction
            lbem_predicts = lbem_model.predict(circuit, PauliOp(Pauli('IIIIZZ')))
            mitigated_diff_lbem.append(abs(lbem_predicts - meas_ideal))


        eval_results.append((param, np.mean(raw_diff), np.mean(mitigated_diff_gan),
                             np.mean(mitigated_diff_cdr), np.mean(mitigated_diff_zne),
                             np.mean(mitigated_diff_lbem)))

    eval_results = sorted(eval_results, key=lambda x: x[0])
    eval_results = np.array(eval_results)
    params = eval_results[:, 0].ravel()
    raw_results = eval_results[:, 1].ravel()
    miti_results_gan = eval_results[:, 2].ravel()
    miti_results_cdr = eval_results[:, 3].ravel()
    miti_results_zne = eval_results[:, 4].ravel()
    miti_results_lbem = eval_results[:, 5].ravel()

    fig = plt.figure()
    plt.plot(params, raw_results)
    plt.plot(params, miti_results_gan)
    # plt.plot(params, miti_results_cdr)
    plt.plot(params, miti_results_zne)
    plt.plot(params, miti_results_lbem)
    # plt.xscale('log')
    # plt.legend(['w/o mitigation', 'GAN mitigation', 'CDR mitigation', 'ZNE mitigation',
    #             'LBEM mitigation'])
    plt.legend(['w/o mitigation', 'GAN mitigation', 'ZNE mitigation',
                'LBEM mitigation'])
    plt.xlabel('Coeff of Ising Model')
    plt.ylabel('Mean Absolute Error')
    plt.savefig('../imgs/mitigate_vs_raw.svg')


@torch.no_grad()
def evaluate_testset():
    backend = Aer.get_backend('aer_simulator')
    paulis = [Pauli(x).to_matrix() for x in ('X', 'Y', 'Z')]
    envs = {}
    for env_name in os.listdir(args.env_path):
        param = float(env_name.replace('.pkl', '').split('_')[-1])
        env_path = os.path.join(args.env_path, env_name)
        envs[param] = IBMQEnv.load(env_path)

    eval_results = []
    # load GAN model
    ckpt = torch.load(args.weight_path, map_location=args.device)
    model_g = SuperviseModel(args.num_mitigates)
    model_g.load_state_dict(ckpt['model_g'], strict=False)
    # model_g.load_envs(args, force=True)
    model_g.to(args.device)
    model_g.eval()

    with open(args.testset, 'rb') as f:
        testset = pickle.load(f)

    random.shuffle(testset)
    # For all

    all_results = {}

    for params, obs, rand_obs_string, pos, scale, exp_noisy, exp_ideal in tqdm(testset):
        if params not in all_results:
            all_results[params] = [[], [], [], []]  # [raw, gan, cdr, zne]

        observable = PauliOp(Pauli(rand_obs_string))
        # noise_backend = envs[params].backends[scale]

        # # CDR
        # cdr_model = CDRTrainer(envs[0.4].backends[0.05])
        # cdr_model.fit(envs[0.4].circuit, observable)
        # ZNE
        zne_model = ZNETrainer(envs[params])

        # state_vec = envs[params].simulate_ideal()
        # density_matrix = envs[params].simulate_noisy(scale)
        meas_ideal = exp_ideal
        meas_noisy = exp_noisy
        all_results[params][0].append(abs(meas_ideal - meas_noisy))
        
        # Model prediction
        obs = torch.tensor(obs, dtype=torch.cfloat)[None].to(args.device)
        param = torch.FloatTensor([params])[None].to(args.device)
        pos = torch.tensor(pos)[None].to(args.device)
        scale = torch.FloatTensor([0.])[None].to(args.device)
        predicts = model_g(param, obs, pos, scale).squeeze().item()
        # prs = model_g(param, obs, pos, scale)
        # predicts = model_g.expectation_from_prs(param_cvt, obs_kron, pos, prs).squeeze().item()
        all_results[params][1].append(abs(meas_ideal - predicts))
        
        # # CDR prediction
        # cdr_predicts = cdr_model.predict(np.array(meas_noisy).reshape(-1, 1))
        # all_results[params][2].append(abs(cdr_predicts - meas_ideal))

        # ZNE prediction
        zne_predicts = zne_model.fit_and_predict(observable)[-1]
        all_results[params][3].append(abs(zne_predicts - meas_ideal))

    parameters = []
    diffs_raw = []
    diffs_sup = []
    # diffs_cdr = []
    diffs_zne = []

    all_results = dict(sorted(all_results.items(), key=lambda x: x[0]))
    for key, val in all_results.items():
        parameters.append(key)
        diffs_raw.append(np.mean(val[0]))
        diffs_sup.append(np.mean(val[1]))
        # diffs_cdr.append(np.mean(val[2]))
        diffs_zne.append(np.mean(val[3]))

    fig = plt.figure()
    plt.plot(parameters, diffs_raw)
    plt.plot(parameters, diffs_sup)
    # plt.plot(parameters, diffs_cdr)
    plt.plot(parameters, diffs_zne)
    # plt.xscale('log')
    plt.legend(['w/o mitigation', 'Supervise mitigation', 'ZNE mitigation'])
    plt.xlabel('Coeff of Ising Model')
    plt.ylabel('Mean Absolute Error')
    plt.savefig('../imgs/comp_exp_phasedamp_vqe4l.png')


@torch.no_grad()
def evaluate_different_noise_scale():
    backend = Aer.get_backend('aer_simulator')
    paulis = [Pauli(x).to_matrix() for x in ('I', 'X', 'Y', 'Z')]
    envs = {}
    for env_name in os.listdir(args.env_path):
        param = float(env_name.replace('.pkl', '').split('_')[-1])
        env_path = os.path.join(args.env_path, env_name)
        envs[param] = IBMQEnv.load(env_path)

    # load GAN model
    ckpt = torch.load(args.weight_path, map_location=args.device)
    model_g = SuperviseModel(args.num_mitigates)
    # model_g = Generator(args.num_mitigates)
    model_g.load_state_dict(ckpt['model_g'], strict=False)
    # model_g.load_envs(args, force=True)
    model_g.to(args.device)
    model_g.eval()

    params = 0.6
    observable = PauliOp(Pauli('XXII'))
    circuit = envs[params].circuit
    scales = np.arange(0, 0.2, 0.001)
    results_sim = []
    results_pred = []
    for scale in scales:
        scale = round(scale, 3)
        noise_backend = envs[params].backends[scale]
        circuit_noisy = circuit.copy()
        circuit_noisy.save_density_matrix()
        # results = noise_backend.run(transpile(circuit_noisy, noise_backend, optimization_level=0)).result()
        # density_matrix = results.data()['density_matrix']
        density_matrix = noise_backend.run(circuit_noisy)
        num_qubits = circuit.num_qubits
        meas_noisy = density_matrix.expectation_value(observable).real

        obs_kron = np.kron(paulis[1], paulis[1])
        obs_kron = torch.tensor(obs_kron, dtype=torch.cfloat)[None].to(args.device)
        # obs = torch.tensor([paulis[0], paulis[0]], dtype=torch.cfloat)[None].to(args.device)
        obs = [np.eye(2) for _ in range(num_qubits)]
        obs[0] = paulis[1]
        obs[1] = paulis[1]
        obs = torch.tensor(obs, dtype=torch.cfloat)[None].to(args.device)
        param = torch.FloatTensor([params])[None].to(args.device)
        pos = torch.tensor([0, 1])[None].to(args.device)
        scale = torch.FloatTensor([scale])[None].to(args.device)
        param_cvt = torch.tensor([np.rint((params - 0.4) * 10)], dtype=int)[None].to(args.device)

        # prs = model_g(param, obs, pos, scale)
        # predicts = model_g.expectation_from_prs(param_cvt, obs_kron, pos, prs)
        # noise = torch.randn(1, 64, dtype=torch.float, device=args.device)
        # predicts = model_g(noise, param, obs, pos, scale)
        predicts = model_g(param, obs, pos, scale)
        results_sim.append(meas_noisy)
        results_pred.append(predicts.cpu())
    
    zne_model = ZNETrainer()
    model_params = zne_model.fit_and_predict(circuit, observable, envs[params].backends)
    
    results_zne = []
    for scale in scales:
        results_zne.append(model_params[0] * scale ** 2 + model_params[1] * scale + model_params[2])

    fig = plt.figure()
    plt.plot(scales, results_sim)
    plt.plot(scales, results_pred)
    plt.plot(scales, results_zne)
    # plt.xscale('log')
    plt.legend(['simulation', 'Supervise prediction', 'ZNE prediction'])
    plt.xlabel('Noise scale')
    plt.ylabel('Expectation of observable')
    plt.savefig('../imgs/results_diff_scales_gd_phasedamp.png')


def operator_2_norm(R):
    """
    Calculate the operator 2-norm.

    Args:
        R (array): The operator whose norm we want to calculate.

    Returns:
        Scalar corresponding to the norm.
    """
    return np.sqrt(np.trace(R.conj().T @ R))


@torch.no_grad()
def evaluate_ae():
    from qiskit.circuit.library.standard_gates import HGate, SdgGate, IGate
    backend = Aer.get_backend('aer_simulator')
    paulis = ['X', 'Y', 'Z']
    NUM_HIDDEN = 4
    envs = {}
    env_name = os.listdir(args.env_path)[0]
    env_path = os.path.join(args.env_path, env_name)
    envs = IBMQEnv.load(env_path)

    encoder = envs.circuit
    decoder = encoder.inverse()
    states = StateGenerator('cat', encoder.num_qubits, shuffle=False).dataset

    eval_results = []
    # load GAN model
    ckpt = torch.load(args.weight_path, map_location=args.device)
    model_g = SuperviseModel(args.num_mitigates, mitigate_prob=True)
    model_g.load_state_dict(ckpt['model_g'], strict=False)
    # model_g.load_envs(args, force=True)
    model_g.to(args.device)
    model_g.eval()

    all_results = {}

    for params, state in tqdm(enumerate(states)):
        if params not in all_results:
            all_results[params] = [[], [], []]  # [raw, supervise, ideal]
        
        input_state = switch_little_big_endian_state(state)
        # Construct original AE circuit for ideal and noisy simulation
        circuit = QuantumCircuit(encoder.num_qubits)
        circuit.initialize(input_state, range(encoder.num_qubits))
        circuit = circuit.compose(encoder)
        # for i in range(NUM_HIDDEN, encoder.num_qubits):
        #     circuit.reset(i)
        # circuit = circuit.compose(decoder)
        
        noise_backend = envs.backends[0.05]

        circuit_ideal = circuit.copy()
        circuit_ideal.save_statevector()
        results = backend.run(transpile(circuit_ideal, backend, optimization_level=0)).result()
        state_vec = results.get_statevector(circuit).data

        circuit_noisy = circuit.copy()
        circuit_noisy.save_density_matrix()
        results = backend.run(transpile(circuit_noisy, backend, optimization_level=0)).result()
        # results = noise_backend.run(transpile(circuit_noisy, noise_backend, optimization_level=0)).result()
        density_matrix = results.data()['density_matrix'].data

        num_qubits = circuit.num_qubits
        
        # all_results[params][0].append((input_state.conj() @ density_matrix @ input_state).real)
        # all_results[params][2].append(np.abs(np.dot(input_state, state_vec.conj())) ** 2)

        shadow = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
        hadamard = HGate().to_matrix()
        sdg = SdgGate().to_matrix()
        identity = IGate().to_matrix()
        zero_state = np.array([[1., 0.], [0., 0.]])
        one_state = np.array([[0., 0.], [0., 1.]])
        bit_states = [zero_state, one_state]
        unitaries = {'X': hadamard, 'Y': hadamard @ sdg, 'Z': identity}
        num_samples = 100_000
        for itr in range(num_samples):
            rho_snapshot = [1]
            for idx in range(encoder.num_qubits):
                rand_pauli = np.random.choice(paulis)
                tmp_circuit = QuantumCircuit(num_qubits, 1)
                tmp_circuit = tmp_circuit.compose(circuit)
                if rand_pauli == 'X':
                    tmp_circuit.h(idx)
                elif rand_pauli == 'Y':
                    tmp_circuit.sdg(idx)
                    tmp_circuit.h(idx)
                tmp_circuit.measure(idx, 0)
                U = unitaries[rand_pauli]
                results = backend.run(transpile(tmp_circuit, backend, optimization_level=0), shots=1).result().get_counts()
                out_state_idx = 0 if '0' in results else 1
                out_state = bit_states[out_state_idx]
                local_rho = 3 * (U.conj().T @ out_state @ U) - identity
                rho_snapshot = np.kron(local_rho, rho_snapshot)
            shadow += rho_snapshot
        shadow /= num_samples
        # shadow /= np.trace(shadow)
        
        # Model prediction
        param = torch.FloatTensor([params])[None].to(args.device)

        predicts = []
        neural_shadow = model_g.construct_shadow(param, num_samples=100_000)
        print(operator_2_norm(shadow - neural_shadow))
        # print(state_vec.conj().T @ shadow @ state_vec)
        print(np.trace(shadow.conj().T @ neural_shadow))
        assert False
        shadow_rho.reset(range(NUM_HIDDEN, encoder.num_qubits))
        decoder_circuit = transpile(decoder, noise_backend, optimization_level=0)
        out_density = shadow_rho.evolve(decoder_circuit).data

        all_results[params][1].append((input_state.conj() @ out_density @ input_state).real)
        print(all_results)
        assert False
        
    parameters = []
    diffs_raw = []
    diffs_gan = []
    diffs_ideal = []

    all_results = dict(sorted(all_results.items(), key=lambda x: x[0]))
    for key, val in all_results.items():
        parameters.append(key)
        diffs_raw.append(np.mean(val[0]))
        diffs_gan.append(np.mean(val[1]))
        diffs_ideal.append(np.mean(val[2]))

    fig = plt.figure()
    plt.plot(parameters, diffs_raw)
    plt.plot(parameters, diffs_gan)
    # plt.plot(parameters, diffs_cdr)
    plt.plot(parameters, diffs_ideal)
    # plt.xscale('log')
    plt.legend(['w/o mitigation', 'Supervise mitigation', 'Ideal'])
    plt.xlabel('Coeff of Ising Model')
    plt.ylabel('Mean Absolute Error')
    plt.savefig('../imgs/comp_exp_gd_phasedamp.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-path', default='../environments/noise_models/phase_damping/vqe_envs_train_4l', type=str)
    parser.add_argument('--weight-path', default='../runs/env_vqe4l_noef_pd_2023-06-04-00-09/gan_model.pt', type=str)
    parser.add_argument('--testset', default='../data_mitigate/phasedamp/testset_vqe4l.pkl', type=str)
    parser.add_argument('--test-num', default=1, type=int, help='number of data to test')
    parser.add_argument('--num-mitigates', default=4, type=int, help='number of mitigation gates')
    parser.add_argument('--num-obs', default=2, type=int, help='number of observables')
    parser.add_argument('--gpus', default='0', type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # evaluate_ae()
    evaluate_testset()
    # evaluate_different_noise_scale()

