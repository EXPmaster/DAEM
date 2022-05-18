import argparse
import os
import numpy as np
import torch

from model import *
from utils import AverageMeter, build_dataloader, abs_deviation
from my_envs import IBMQEnv

@torch.no_grad()
def infer_gan(args):
	ckpt = torch.load(args.weight_path, map_location=args.device)
	model_g = Generator(dim_out=args.num_mitigates).to(args.device)
	model_s = SurrogateModel(dim_in=4 * args.num_mitigates + 8).to(args.device)
	model_g.load_state_dict(ckpt['model_g'])
	model_s.load_state_dict(ckpt['model_s'])
	model_g.requires_grad_(False)
	model_s.requires_grad_(False)
	model_g.eval()
	model_s.eval()

	obs = torch.diag(torch.tensor([1., -1.], dtype=torch.cfloat, device=args.device))[None]
	# obs = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.cfloat,device=args.device)[None]
	# obs = torch.tensor([[0., -1.j], [1.j, 0.]], dtype=torch.cfloat,device=args.device)[None]
	result = model_s(model_g(obs), obs)
	ideal_result = 0.5
	print(f'predicted result: {result.item()}')
	print(f'ideal result: {ideal_result}')
	print(abs(result.item() - ideal_result))

@torch.no_grad()
def infer_supervised(args):
	ckpt = torch.load(args.weight_path, map_location=args.device)
	model_g = MitigateModel(dim_in=8, dim_out=args.num_mitigates).to(args.device)
	model_s = SurrogateModel(dim_in=4 * args.num_mitigates + 8).to(args.device)
	model_g.load_state_dict(ckpt['model_g'])
	model_s.load_state_dict(ckpt['model_s'])
	model_g.requires_grad_(False)
	model_s.requires_grad_(False)
	model_g.eval()
	model_s.eval()

	obs = torch.diag(torch.tensor([1., -1.], dtype=torch.cfloat, device=args.device))[None]
	# obs = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.cfloat,device=args.device)[None]
	# obs = torch.tensor([[0., -1.j], [1.j, 0.]], dtype=torch.cfloat,device=args.device)[None]
	result = model_s(model_g(obs, 0), obs)
	ideal_result = 0.5
	print(f'predicted result: {result.item()}')
	print(f'ideal result: {ideal_result}')
	print(abs(result.item() - ideal_result))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weight-path', default='../runs/env_swaptest/gan_model.pt', type=str)
	parser.add_argument('--batch-size', default=128, type=int)
	parser.add_argument('--num-mitigates', default=5, type=int, help='number of mitigation gates')
	parser.add_argument('--workers', default=8, type=int, help='dataloader worker nums')
	parser.add_argument('--gpus', default='0', type=str)
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
	args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# infer_supervised(args)
	infer_gan(args)

