import argparse
import pickle
import functools
import os
from vqe import VQETrainer
import numpy as np
import torch
from torch.utils.data import DataLoader
from qiskit import Aer

from model import *
from utils import AverageMeter, abs_deviation, gen_rand_obs
from my_envs import IBMQEnv
from datasets import MitigateDataset


def run_test():
    trainer = VQETrainer(6, 3)
    circuit0 = trainer.get_circuit()
    g = -0.25
    
    H = trainer.get_hamitonian_ising(g)
    circuit = trainer.train(circuit0, H)
    circuit.save_statevector()
    backend = Aer.get_backend('aer_simulator')
    results = backend.run(circuit, shots=1024).result()
    state_vec = results.get_statevector(circuit)

    # circuit = trainer.train(circuit0, H)
    # circuit.save_statevector()
    # backend = Aer.get_backend('aer_simulator')
    # results = backend.run(circuit, shots=1024).result()
    # state_vec2 = results.get_statevector(circuit)

    # print(np.abs(state_vec1.inner(state_vec2)) ** 2)
    # with open(f'../environments/circuits/vqe_{g}.pkl', 'rb') as f:
    #     circuit = pickle.load(f)
    # circuit.save_statevector()
    # backend = Aer.get_backend('aer_simulator')
    # results = backend.run(circuit, shots=1024).result()
    # state_vec = results.get_statevector(circuit)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data', default='../data_mitigate/testset_randomcirc.pkl', type=str)
    parser.add_argument('--weight-path', default='../runs/env_vqe/gan_model.pt', type=str)
    parser.add_argument('--num-mitigates', default=6, type=int, help='number of mitigation gates')
    parser.add_argument('--num-obs', default=2, type=int, help='number of observables')
    parser.add_argument('--gpus', default='0', type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    run_test()

