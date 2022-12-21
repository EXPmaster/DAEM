import os
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
from qiskit.quantum_info.operators import Pauli
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import *
from utils import AverageMeter, abs_deviation


class TestDataset(Dataset):

    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.dataset = pickle.load(f)

    def __getitem__(self, idx):
        param, obs, pos, noise_scale, exp_noisy, exp_ideal = self.dataset[idx]
        param_converted = int((param - 0.4) * 5)
        obs_kron = np.kron(obs[0], obs[1])
        return (
            torch.FloatTensor([param]),
            torch.tensor([param_converted]),
            torch.tensor(obs, dtype=torch.cfloat),
            torch.tensor(obs_kron, dtype=torch.cfloat),
            torch.tensor(pos),
            torch.FloatTensor([noise_scale]),
            torch.FloatTensor([exp_noisy]),
            torch.FloatTensor([exp_ideal])
        )

    def __len__(self):
        return len(self.dataset)


@torch.no_grad()
def test(args):
    testset = TestDataset(args.test_path)
    loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, drop_last=True)
    model_g = Generator(args.num_mitigates)
    state_dict = torch.load(args.weight_path, map_location=args.device)
    model_g.load_state_dict(state_dict['model_g'], strict=False)
    model_g.load_envs(args, force=True)
    model_g.to(args.device)
    model_g.eval()
    metric = AverageMeter()

    print('Testing...')
    for params, params_cvt, obs, obs_kron, pos, scale, noisy_r, gts in tqdm(loader):
        params_cvt = params_cvt.to(args.device)
        params, obs, pos, gts = params.to(args.device), obs.to(args.device), pos.to(args.device), gts.to(args.device)
        scale = scale.to(args.device)
        obs_kron = obs_kron.to(args.device)
        scale = torch.full((len(params), 1), 0.0, dtype=torch.float, device=args.device)
        predicts = []
        for _ in range(args.num_samples):
            noise = torch.randn(len(params), 64, dtype=torch.float, device=args.device)
            prs = model_g(noise, params, obs, pos, scale)
            preds = model_g.expectation_from_prs(params_cvt, obs_kron, pos, prs)
            predicts.append(preds)
        predicts = torch.stack(predicts).mean(0)
        metric.update(abs_deviation(predicts, gts))

    value = metric.getval()
    print('validation absolute deviation: {:.6f}'.format(value))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-path', default='../data_mitigate/testset_vqe4l.pkl', type=str)
    parser.add_argument('--env-path', default='../environments/vqe_envs_test_4l', type=str)
    parser.add_argument('--weight-path', default='../runs/env_vqe_noef/gan_model.pt', type=str)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--num-mitigates', default=12, type=int, help='number of mitigation gates')
    parser.add_argument('--num-samples', default=100, type=int, help='number of samples to be averaged')
    parser.add_argument('--num-ops', default=2, type=int)
    parser.add_argument('--workers', default=4, type=int, help='dataloader worker nums')
    parser.add_argument('--gpus', default='0', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(args.device)

    test(args)
