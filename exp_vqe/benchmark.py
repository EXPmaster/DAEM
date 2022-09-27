import argparse
import pickle
import os
from vqe import VQETrainer
import numpy as np
import torch
from torch.utils.data import DataLoader
from qiskit import Aer

from model import *
from utils import AverageMeter, abs_deviation
from my_envs import IBMQEnv
from datasets import MitigateDataset


def run_test():
    trainer = VQETrainer(4, 3)
    circuit0 = trainer.get_circuit()
    g = -0.8
    
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

    rand_matrix = (torch.rand((2, 2), dtype=torch.cfloat) * 2 - 1).numpy()
    rand_obs = (rand_matrix.conj().T + rand_matrix) / 2
    eigen_val = np.linalg.eigvalsh(rand_obs)
    rand_obs = rand_obs / np.max(np.abs(eigen_val))
    meas_ideal = state_vec.expectation_value(np.kron(np.eye(2**3), rand_obs)).real
    rand_obs = torch.from_numpy(rand_obs)[None].to(args.device)

    params = torch.FloatTensor([g])[None].to(args.device)

    ckpt = torch.load(args.weight_path, map_location=args.device)
    model_g = Generator(dim_out=args.num_mitigates).to(args.device)
    model_s = SurrogateModel(dim_in=4 * args.num_mitigates + 9).to(args.device)
    model_g.load_state_dict(ckpt['model_g'])
    model_s.load_state_dict(ckpt['model_s'])
    model_g.requires_grad_(False)
    model_s.requires_grad_(False)
    model_g.eval()
    model_s.eval()

    prs = model_g(params, rand_obs)
    predicts = model_s(params, prs, rand_obs)

    print(predicts)
    print(meas_ideal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data', default='../data_mitigate/testset_randomcirc.pkl', type=str)
    parser.add_argument('--weight-path', default='../runs/env_vqe/gan_model.pt', type=str)
    parser.add_argument('--num-mitigates', default=4, type=int, help='number of mitigation gates')
    parser.add_argument('--gpus', default='0', type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    run_test()

