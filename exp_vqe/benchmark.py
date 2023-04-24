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
from qiskit.opflow import PauliOp
from tqdm import tqdm

from model import *
from utils import AverageMeter, abs_deviation, gen_rand_obs
from my_envs import IBMQEnv
from datasets import MitigateDataset
from cdr_trainer import CDRTrainer
from lbem_trainer import LBEMTrainer2
from zne_trainer import ZNETrainer


def run_single_test():
    trainer = VQETrainer(6, 3)
    circuit0 = trainer.get_circuit()
    g = 0.3
    
    H = trainer.get_hamitonian_ising(g)
    circuit = trainer.train(circuit0, H)
    circuit.save_statevector()
    backend = Aer.get_backend('aer_simulator')
    results = backend.run(circuit, shots=1024).result()
    state_vec = results.get_statevector(circuit)
    rand_obs = [gen_rand_obs() for i in range(args.num_obs)]
    num_qubits = circuit.num_qubits
    rand_idx = np.random.randint(num_qubits - 1)
    selected_qubits = list(range(rand_idx, rand_idx + args.num_obs))  # [rand_idx, rand_idx + 1]
    obs = [np.eye(2) for i in range(rand_idx)] + rand_obs +\
            [np.eye(2) for i in range(rand_idx + len(selected_qubits), num_qubits)]
    obs_kron = functools.reduce(np.kron, obs[::-1])
    obs = np.array(rand_obs)
    meas_ideal = state_vec.expectation_value(obs_kron).real
    obs = torch.tensor(obs, dtype=torch.cfloat)[None].to(args.device)

    params = torch.FloatTensor([g])[None].to(args.device)
    pos = torch.tensor(selected_qubits)[None].to(args.device)

    ckpt = torch.load(args.weight_path, map_location=args.device)
    model_g = Generator(args.num_mitigates).to(args.device)
    model_s = SurrogateModel(args.num_mitigates).to(args.device)
    model_g.load_state_dict(ckpt['model_g'])
    model_s.load_state_dict(ckpt['model_s'])
    model_g.requires_grad_(False)
    model_s.requires_grad_(False)
    model_g.eval()
    model_s.eval()

    prs = model_g(params, obs, pos)
    predicts = model_s(params, prs, obs, pos)

    print(predicts)
    print(meas_ideal)


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
def eval_testset():
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
    model_g = Generator(args.num_mitigates)
    model_g.load_state_dict(ckpt['model_g'], strict=False)
    model_g.load_envs(args, force=True)
    model_g.to(args.device)
    model_g.eval()

    with open(args.testset, 'rb') as f:
        testset = pickle.load(f)

    mitigated_diff_gan = []
    raw_diff = []
    mitigated_diff_cdr = []
    mitigated_diff_zne = []
    mitigated_diff_lbem = []

    for params, obs, rand_obs_string, pos, scale, exp_noisy, exp_ideal in tqdm(testset):
        circuit = envs[params].circuit
        observable = PauliOp(Pauli(rand_obs_string))
        noise_backend = envs[params].backends[scale]

        # CDR
        cdr_model = CDRTrainer(envs[0.4].backends[0.05])
        cdr_model.fit(envs[0.4].circuit, observable)
        # ZNE
        zne_model = ZNETrainer()
        # # LBEM
        # lbem_model = LBEMTrainer2(noise_model)


        circuit_ideal = circuit.copy()
        circuit_ideal.save_statevector()
        results = backend.run(transpile(circuit_ideal, backend, optimization_level=0)).result()
        state_vec = results.get_statevector(circuit)

        circuit_noisy = circuit.copy()
        circuit_noisy.save_density_matrix()
        results = noise_backend.run(transpile(circuit_noisy, noise_backend, optimization_level=0)).result()
        density_matrix = results.data()['density_matrix']

        num_qubits = circuit.num_qubits
        
        meas_ideal = state_vec.expectation_value(observable).real
        meas_noisy = density_matrix.expectation_value(observable).real
        raw_diff.append(abs(meas_ideal - meas_noisy))
        
        # GAN prediction
        obs_kron = np.kron(obs[0], obs[1])
        obs_kron = torch.tensor(obs_kron, dtype=torch.cfloat)[None].to(args.device)
        obs = torch.tensor(obs, dtype=torch.cfloat)[None].to(args.device)
        param = torch.FloatTensor([params])[None].to(args.device)
        pos = torch.tensor(pos)[None].to(args.device)
        scale = torch.FloatTensor([0.])[None].to(args.device)
        param_cvt = torch.tensor([np.rint((params - 0.4) * 5)], dtype=int)[None].to(args.device)
        predicts = []
        for _ in range(100):
            noise = torch.randn(1, 64, dtype=torch.float, device=args.device)
            prs = model_g(noise, param, obs, pos, scale)
            preds = model_g.expectation_from_prs(param_cvt, obs_kron, pos, prs)
            predicts.append(preds)
        predicts = torch.stack(predicts).mean(0).item()
        mitigated_diff_gan.append(abs(meas_ideal - predicts))
        
        # CDR prediction
        cdr_predicts = cdr_model.predict(np.array(meas_noisy).reshape(-1, 1))
        mitigated_diff_cdr.append(abs(cdr_predicts - meas_ideal))

        # ZNE prediction
        zne_predicts = zne_model.fit_and_predict(circuit, observable)
        mitigated_diff_zne.append(abs(zne_predicts - meas_ideal))

    print(np.mean(raw_diff), np.mean(mitigated_diff_gan), np.mean(mitigated_diff_cdr), np.mean(mitigated_diff_zne))


@torch.no_grad()
def evaluate_arbitrary():
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
    model_g.load_envs(args, force=True)
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

        circuit = envs[params].circuit
        observable = PauliOp(Pauli(rand_obs_string))
        noise_backend = envs[params].backends[scale]

        # CDR
        cdr_model = CDRTrainer(envs[0.4].backends[0.05])
        cdr_model.fit(envs[0.4].circuit, observable)
        # ZNE
        zne_model = ZNETrainer()

        circuit_ideal = circuit.copy()
        circuit_ideal.save_statevector()
        results = backend.run(transpile(circuit_ideal, backend, optimization_level=0)).result()
        state_vec = results.get_statevector(circuit)

        circuit_noisy = circuit.copy()
        circuit_noisy.save_density_matrix()
        results = noise_backend.run(transpile(circuit_noisy, noise_backend, optimization_level=0)).result()
        density_matrix = results.data()['density_matrix']

        num_qubits = circuit.num_qubits
        
        meas_ideal = state_vec.expectation_value(observable).real
        meas_noisy = density_matrix.expectation_value(observable).real
        all_results[params][0].append(abs(meas_ideal - meas_noisy))
        
        # GAN prediction
        # obs_kron = np.kron(obs[0], obs[1])
        obs_kron = np.kron(obs[pos[0]], obs[pos[1]])
        obs_kron = torch.tensor(obs_kron, dtype=torch.cfloat)[None].to(args.device)
        obs = torch.tensor(obs, dtype=torch.cfloat)[None].to(args.device)
        param = torch.FloatTensor([params])[None].to(args.device)
        pos = torch.tensor(pos)[None].to(args.device)
        scale = torch.FloatTensor([0.])[None].to(args.device)
        param_cvt = torch.tensor([np.rint((params - 0.4) * 10)], dtype=int)[None].to(args.device)
        predicts = []
        # for _ in range(100):
        #     # print(rand_obs_string)
        #     noise = torch.randn(1, 64, dtype=torch.float, device=args.device)
        #     prs = model_g(noise, param, obs, pos, scale)
        #     # arr = prs.cpu().numpy().ravel()
        #     # fig = plt.figure()
        #     # plt.bar(np.arange(len(arr)), arr)
        #     # plt.savefig('../imgs/quasiprob.png')

        #     # assert False
        #     preds = model_g.expectation_from_prs(param_cvt, obs_kron, pos, prs)
        #     predicts.append(preds)
        # predicts = torch.stack(predicts).mean(0).item()
        predicts = model_g(param, obs, pos, scale).squeeze().item()
        # prs = model_g(param, obs, pos, scale)
        # predicts = model_g.expectation_from_prs(param_cvt, obs_kron, pos, prs).squeeze().item()
        all_results[params][1].append(abs(meas_ideal - predicts))
        
        # CDR prediction
        cdr_predicts = cdr_model.predict(np.array(meas_noisy).reshape(-1, 1))
        all_results[params][2].append(abs(cdr_predicts - meas_ideal))

        # ZNE prediction
        zne_predicts = zne_model.fit_and_predict(circuit, observable, envs[params].backends)[-1]
        all_results[params][3].append(abs(zne_predicts - meas_ideal))

    parameters = []
    diffs_raw = []
    diffs_gan = []
    diffs_cdr = []
    diffs_zne = []

    all_results = dict(sorted(all_results.items(), key=lambda x: x[0]))
    for key, val in all_results.items():
        parameters.append(key)
        diffs_raw.append(np.mean(val[0]))
        diffs_gan.append(np.mean(val[1]))
        diffs_cdr.append(np.mean(val[2]))
        diffs_zne.append(np.mean(val[3]))

    fig = plt.figure()
    plt.plot(parameters, diffs_raw)
    plt.plot(parameters, diffs_gan)
    plt.plot(parameters, diffs_cdr)
    plt.plot(parameters, diffs_zne)
    # plt.xscale('log')
    plt.legend(['w/o mitigation', 'Supervise mitigation', 'CDR mitigation', 'ZNE mitigation'])
    plt.xlabel('Coeff of Ising Model')
    plt.ylabel('Mean Absolute Error')
    plt.savefig('../imgs/comp_exp_gd_ampdamp.png')


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
    model_g.load_envs(args, force=True)
    model_g.to(args.device)
    model_g.eval()

    params = 0.7
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
        results = noise_backend.run(transpile(circuit_noisy, noise_backend, optimization_level=0)).result()
        density_matrix = results.data()['density_matrix']
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
    plt.savefig('../imgs/results_diff_scales_gd_ampdamping.png')


# @torch.no_grad()
# def evaluate_different_noise_scale():
#     backend = Aer.get_backend('aer_simulator')
#     paulis = [Pauli(x).to_matrix() for x in ('I', 'X', 'Y', 'Z')]
#     envs = {}
#     for env_name in os.listdir(args.env_path):
#         param = float(env_name.replace('.pkl', '').split('_')[-1])
#         env_path = os.path.join(args.env_path, env_name)
#         envs[param] = IBMQEnv.load(env_path)

#     # load GAN model
#     ckpt = torch.load(args.weight_path, map_location=args.device)
#     model_g = Generator(args.num_mitigates)
#     model_g.load_state_dict(ckpt['model_g'], strict=False)
#     model_g.load_envs(args, force=True)
#     model_g.to(args.device)
#     model_g.eval()

#     params = 0.6
#     observable = PauliOp(Pauli('XXII'))
#     circuit = envs[params].circuit
#     scales = np.arange(0, 0.2, 0.001)
#     results_sim = []
#     results_pred = []
#     for scale in scales:
#         scale = round(scale, 3)
#         noise_backend = envs[params].backends[scale]
#         circuit_noisy = circuit.copy()
#         circuit_noisy.save_density_matrix()
#         results = noise_backend.run(transpile(circuit_noisy, noise_backend, optimization_level=0)).result()
#         density_matrix = results.data()['density_matrix']
#         num_qubits = circuit.num_qubits
#         meas_noisy = density_matrix.expectation_value(observable).real

#         obs_kron = np.kron(paulis[0], paulis[0])
#         obs_kron = torch.tensor(obs_kron, dtype=torch.cfloat)[None].to(args.device)
#         obs = torch.tensor([paulis[0], paulis[0]], dtype=torch.cfloat)[None].to(args.device)
#         param = torch.FloatTensor([params])[None].to(args.device)
#         pos = torch.tensor([0, 1])[None].to(args.device)
#         scale = torch.FloatTensor([scale])[None].to(args.device)
#         param_cvt = torch.tensor([np.rint((params - 0.4) * 5)], dtype=int)[None].to(args.device)
#         predicts = []
#         for _ in range(100):
#             noise = torch.randn(1, 64, dtype=torch.float, device=args.device)
#             prs = model_g(noise, param, obs, pos, scale)
#             preds = model_g.expectation_from_prs(param_cvt, obs_kron, pos, prs)
#             # if abs(scale - 0.05) < 1e-6:
#             #     arr = prs.cpu().numpy().ravel()
#             #     fig = plt.figure()
#             #     plt.bar(np.arange(len(arr)), arr)
#             #     plt.savefig('../imgs/quasiprob_ampdamp.png')
#             #     print(preds)
#             #     print(meas_noisy)
#             #     rho = model_g.rhos[param_cvt.flatten()]  # .squeeze(1)
#             #     rho = rho[torch.arange(param_cvt.shape[0]), pos[:, 0]]
#             #     meas_results = torch.matmul(rho, obs_kron.unsqueeze(1)).diagonal(dim1=-2, dim2=-1).sum(-1).real
#             #     print(meas_results)
#             #     assert False
#             predicts.append(preds)
#         predicts = torch.stack(predicts).mean(0).item()
#         results_sim.append(meas_noisy)
#         results_pred.append(predicts)
    
#     zne_model = ZNETrainer()
#     model_params = zne_model.fit_and_predict(circuit, observable)
    
#     results_zne = []
#     for scale in scales:
#         results_zne.append(model_params[0] * scale ** 2 + model_params[1] * scale + model_params[2])

#     fig = plt.figure()
#     plt.plot(scales, results_sim)
#     plt.plot(scales, results_pred)
#     plt.plot(scales, results_zne)
#     # plt.xscale('log')
#     plt.legend(['simulation', 'GAN prediction', 'ZNE prediction'])
#     plt.xlabel('Noise scale')
#     plt.ylabel('Expectation of observable')
#     plt.savefig('../imgs/results_diff_scales_ampdamping.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-path', default='../environments/gate_dependent_ad/vqe_envs_train_4l', type=str)
    parser.add_argument('--weight-path', default='../runs/env_vqe_noef_ampdamp_gd_2023-04-24-14-20/gan_model.pt', type=str)
    parser.add_argument('--testset', default='../data_mitigate/gate_dependent_ad/testset_train.pkl', type=str)
    parser.add_argument('--test-num', default=1, type=int, help='number of data to test')
    parser.add_argument('--num-mitigates', default=4, type=int, help='number of mitigation gates')
    parser.add_argument('--num-obs', default=2, type=int, help='number of observables')
    parser.add_argument('--gpus', default='0', type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # eval_testset()
    evaluate_arbitrary()
    # evaluate_different_noise_scale()

