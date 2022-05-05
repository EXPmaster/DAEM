import argparse
import os
import numpy as np
import torch

from model import *
from utils import AverageMeter, build_dataloader, abs_deviation
from my_envs import IBMQEnv


def infer(args):
	ckpt = torch.load(args.weight_path, map_location=args.device)
	model_g = Generator(num_layers=args.num_layers, num_qubits=args.num_qubits).to(args.device)
	model_s = SurrogateModel(dim_in=4 * args.num_layers * args.num_qubits + 8).to(args.device)
	model_g.load_state_dict(ckpt['model_g'])
	model_s.load_state_dict(ckpt['model_s'])
	model_g.requires_grad_(False)
	model_s.requires_grad_(False)
	model_g.eval()
	model_s.eval()

	env = IBMQEnv.load(args.env_path)
	env.gen_new_circuit_without_id()
	ideal_state = env.simulate_ideal()
	obs = torch.diag(torch.tensor([1., -1.], dtype=torch.cfloat, device=args.device))[None]
	result = model_s(model_g(obs), obs)
	print(result)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weight-path', default='../runs/env_ibmq/gan_model1.pt', type=str)
	parser.add_argument('--env-path', default='../environments/ibmq1.pkl', type=str)
	parser.add_argument('--logdir', default='../runs/env_ibmq', type=str, help='path to save logs and models')
	parser.add_argument('--batch-size', default=128, type=int)
	parser.add_argument('--num-layers', default=4, type=int, help='depth of the circuit')
	parser.add_argument('--num-qubits', default=2, type=int, help='number of qubits')
	parser.add_argument('--workers', default=8, type=int, help='dataloader worker nums')
	parser.add_argument('--gpus', default='0', type=str)
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
	args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	infer(args)

