import argparse
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
from lbem_trainer import LBEMTrainer
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

    for circ_name in tqdm(os.listdir(args.test_root)):
        param = float(circ_name.replace('.pkl', '').split('_')[-1])
        if param < 0.6 or param > 1.6: continue
        circ_path = os.path.join(args.test_root, circ_name)
        with open(circ_path, 'rb') as f:
            circuit = pickle.load(f)
        
        circuit_ideal = circuit.copy()
        circuit_ideal.save_statevector()
        results = backend.run(transpile(circuit_ideal, backend)).result()
        state_vec = results.get_statevector(circuit)

        circuit_noisy = circuit.copy()
        circuit_noisy.save_density_matrix()
        results = noise_backend.run(transpile(circuit_noisy, noise_backend)).result()
        density_matrix = results.data()['density_matrix']

        num_qubits = circuit.num_qubits
        mitigated_diff_gan = []
        raw_diff = []
        mitigated_diff_cdr = []
        mitigated_diff_zne = []

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
            mitigated_diff_cdr.append(abs(cdr_predicts[0][0] - meas_ideal))

            # ZNE prediction
            zne_predicts = zne_model.fit_and_predict(circuit, PauliOp(Pauli('IIIIZZ')))
            mitigated_diff_zne.append(abs(zne_predicts - meas_ideal))

        eval_results.append((param, np.mean(raw_diff), np.mean(mitigated_diff_gan),
                             np.mean(mitigated_diff_cdr), np.mean(mitigated_diff_zne)))

    eval_results = sorted(eval_results, key=lambda x: x[0])
    eval_results = np.array(eval_results)
    params = eval_results[:, 0].ravel()
    raw_results = eval_results[:, 1].ravel()
    miti_results_gan = eval_results[:, 2].ravel()
    miti_results_cdr = eval_results[:, 3].ravel()
    miti_results_zne = eval_results[:, 4].ravel()
    

    fig = plt.figure()
    plt.plot(params, raw_results)
    plt.plot(params, miti_results_gan)
    plt.plot(params, miti_results_cdr)
    plt.plot(params, miti_results_zne)
    # plt.xscale('log')
    plt.legend(['w/o mitigation', 'GAN mitigation', 'CDR mitigation', 'ZNE mitigation'])
    plt.xlabel('Coeff of Ising Model')
    plt.ylabel('Mean Absolute Error')
    plt.savefig('../imgs/mitigate_vs_raw.svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-root', default='../environments/circuits_test', type=str)
    parser.add_argument('--weight-path', default='../runs/env_vqe/gan_model_best.pt', type=str)
    parser.add_argument('--test-num', default=1, type=int, help='number of data to test')
    parser.add_argument('--num-mitigates', default=6, type=int, help='number of mitigation gates')
    parser.add_argument('--num-obs', default=2, type=int, help='number of observables')
    parser.add_argument('--gpus', default='0', type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    evaluation()

