import os
import time
import pickle
import argparse

import numpy as np
import torch
from torch import autograd
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from qiskit.quantum_info.operators import Pauli

from model import *
from utils import AverageMeter, build_dataloader, abs_deviation
from datasets import MitigateDataset


def main(args):
    trainset, testset, train_loader, test_loader = build_dataloader(args, eval(args.dataset))
    print(len(trainset))
    train_loader = iter(train_loader)
    loss_fn = nn.BCELoss()
    # loss_fn = nn.MSELoss()
    # model_s.load_state_dict(torch.load(args.weight_path, map_location=args.device))
    model_g = Generator(args.num_mitigates)
    model_g.load_envs(args, force=True)
    model_g.to(args.device)
    model_d = Discriminator(args.num_mitigates).to(args.device)
    # optimizer_g = optim.Adam(model_g.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    # optimizer_d = optim.Adam(model_d.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizer_g = optim.RMSprop(model_g.parameters(), lr=args.lr_g)
    optimizer_d = optim.RMSprop(model_d.parameters(), lr=args.lr_d)
    print('Start training...')
    scheduler_g = StepLR(optimizer_g, 30, gamma=0.5)
    scheduler_d = StepLR(optimizer_d, 30, gamma=0.5)

    best_metric = args.thres
    for epoch in range(args.epochs):
        print(f'=> Epoch {epoch}')
        train(epoch, args, train_loader, model_g, model_d, loss_fn, optimizer_g, optimizer_d)
        # if epoch % 10 == 0:
        metric = validate(epoch, args, test_loader, model_g, loss_fn)
        scheduler_g.step()
        scheduler_d.step()
        if metric < best_metric:
            print('Saving model...')
            best_metric = metric
            ckpt = {
                'model_g': model_g.state_dict(),
                'model_d': model_d.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict()
            }
            torch.save(ckpt, os.path.join(args.logdir, args.save_name))
    with open(os.path.join(args.logdir, 'metric_gan.txt'), 'a+') as f:
        f.write('{} {:.6f}\n'.format(len(trainset), best_metric))


def gen_rand_obs_torch(args):
    rand_matrix = (torch.rand((args.batch_size, args.num_ops, 2, 2), dtype=torch.cfloat) * 2.0 - 1.0).to(args.device)
    rand_hermitian = (rand_matrix.conj().mT + rand_matrix) / 2
    eigen_vals = torch.linalg.eigvalsh(rand_hermitian)
    rand_obs = rand_hermitian / eigen_vals.abs().max(1, keepdim=True)[0][:, :, None]
    return rand_obs


def rand_pauli_torch_generator(args):
    paulis = torch.tensor(np.array([Pauli('I').to_matrix(),
                           Pauli('X').to_matrix(),
                           Pauli('Y').to_matrix(),
                           Pauli('Z').to_matrix()]), dtype=torch.cfloat)
    def gen_fn(num_ops=1):
        rand_indices = torch.randint(0, 4, size=(args.batch_size, num_ops))
        return paulis[rand_indices]
    return gen_fn


# def train(epoch, args, loader, model_g, model_s, model_d, loss_fn, optimizer_g, optimizer_d):
#     model_g.train()
#     model_s.train()
#     model_d.train()
#     pauli_generator = rand_pauli_torch_generator(args)
#     loss_g_avg = AverageMeter()
#     loss_d_avg = AverageMeter()

#     for itr, (params, obs, pos, scale, exp_noisy, _) in enumerate(loader):
#         # Update D to maximize log(D(x)) + log(1 - D(G(z)))
#         ## real
#         params, obs, pos, scale = params.to(args.device), obs.to(args.device), pos.to(args.device), scale.to(args.device)
#         exp_noisy = exp_noisy.to(args.device)
#         optimizer_d.zero_grad()
#         labels = torch.full((args.batch_size, 1), 1.0, dtype=torch.float, device=args.device)
#         output = model_d(exp_noisy, params, obs, pos, scale)
#         D_ideal = output.mean().item()
#         lossD_real = loss_fn(output, labels)
#         lossD_real.backward()

#         ## fake
#         # rand_matrix = torch.randn((args.batch_size, 2, 2), dtype=torch.cfloat).to(args.device)
#         # rand_hermitian = torch.bmm(rand_matrix.conj().mT, rand_matrix)
        
#         rand_params = (torch.rand((args.batch_size, 1)) * 4 - 2).to(args.device)
#         rand_obs = pauli_generator(args.num_ops).to(args.device)  # gen_rand_obs_torch(args)
#         rand_pos = torch.randint(0, args.num_mitigates - 1, size=(args.batch_size, 1)).to(args.device)
#         rand_pos = torch.cat((rand_pos, rand_pos + 1), 1)
#         # rand_scale = (torch.rand((args.batch_size, 1)) * 0.1).to(args.device)
#         rand_scale = (torch.rand((args.batch_size, 1)) * (0.1 - 0.01) + 0.01).to(args.device)
#         sep_idx = args.batch_size // 3
#         rand_params = torch.cat((rand_params[:sep_idx], params[sep_idx:]), 0)
#         rand_obs = torch.cat((rand_obs[:sep_idx], obs[sep_idx:]), 0)
#         rand_pos = torch.cat((rand_pos[:sep_idx], pos[sep_idx:]), 0)
#         rand_scale = torch.cat((rand_scale[:sep_idx], scale[sep_idx:]), 0)
#         labels.fill_(0.0)
#         fake = model_s(rand_params, model_g(rand_params, rand_obs, rand_pos, rand_scale), rand_obs, rand_pos, rand_scale)
#         output = model_d(fake.detach(), rand_params, rand_obs, rand_pos, rand_scale)
#         D_g_z1 = output.mean().item()
#         lossD_fake1 = loss_fn(output, labels)
#         lossD_fake1.backward()

#         output = model_d(exp_noisy, params, obs, pos, scale)
#         D_noisy = output.mean().item()
#         lossD_fake2 = loss_fn(output, labels)
#         lossD_fake2.backward()
#         optimizer_d.step()
#         lossD = lossD_real + lossD_fake1 + lossD_fake2

#         # Update G to maximize log(D(G(z)))
#         optimizer_g.zero_grad()
#         labels.fill_(1.0)
#         output = model_d(fake, rand_params, rand_obs, rand_pos, rand_scale)
#         D_g_z2 = output.mean().item()
#         lossG = loss_fn(output, labels)
#         lossG.backward()
#         optimizer_g.step()

#         loss_g_avg.update(lossG.item())
#         loss_d_avg.update(lossD.item())
#         if itr % 1000 == 0:
#             # args.writer.add_scalar('Loss/train', loss_accumulator.getval(), epoch)
#             print('Loss_D: {:.4f}\tLoss_G\t{:.4f}\tD(noisy): {:.4f}\tD(ideal): {:.4f}\tD(G(z)): {:.4f} / {:.4f}'.format(lossD, lossG, D_noisy, D_ideal, D_g_z1, D_g_z2))
        
#     args.writer.add_scalar('Loss/loss_G', loss_g_avg.getval(), epoch)
#     args.writer.add_scalar('Loss/loss_D', loss_d_avg.getval(), epoch)


def train(epoch, args, loader, model_g, model_d, loss_fn, optimizer_g, optimizer_d):
    model_g.train()
    model_d.train()
    pauli_generator = rand_pauli_torch_generator(args)
    loss_g_avg = AverageMeter()
    loss_d_avg = AverageMeter()

    for itr in range(1000):
        for p in model_d.parameters():
            p.requires_grad = True 
        for d_itr in range(args.d_steps):
            params, params_cvt, obs, obs_kron, pos, scale, exp_noisy, _ = next(loader)
            # Update D to maximize log(D(x)) + log(1 - D(G(z)))

            ## real
            params_cvt = params_cvt.to(args.device)
            params, obs, pos, scale = params.to(args.device), obs.to(args.device), pos.to(args.device), scale.to(args.device)
            obs_kron = obs_kron.to(args.device)
            exp_noisy = exp_noisy.to(args.device)
            # quasi_prob = quasi_prob.to(args.device)
            optimizer_d.zero_grad()
            labels = torch.full((args.batch_size, 1), 1., dtype=torch.float, device=args.device)
            output_real = model_d(exp_noisy, params, obs, pos, scale)
            # loss_real = -output_real.mean() + args.gamma * value_real.mean()
            loss_real = -output_real.mean()
            loss_real.backward()
            # output = model_d(quasi_prob, params, obs, pos, scale)

            ## fake
            noise = torch.randn(args.batch_size, 64, dtype=torch.float, device=args.device)
            # rand_scale = (torch.rand((args.batch_size, 1)) * 0.3).to(args.device)
            # sep_idx = args.batch_size // 3
            # scale = torch.cat((rand_scale[:sep_idx], scale[sep_idx:]), 0)
            # labels = torch.rand((args.batch_size, 1), dtype=torch.float, device=args.device) * 0.1
            # fake = model_g.expectation_from_prs(params_cvt, obs_kron, pos, model_g(noise, params, obs, pos, scale))
            fake = model_g(noise, params, obs, pos, scale)
            # scale = scale.fill_(0.05)
            output_fake = model_d(fake.detach(), params, obs, pos, scale)
            loss_fake = output_fake.mean()
            loss_fake.backward()

            # Gradient Penalty
            eta = torch.FloatTensor(args.batch_size, 1).uniform_(0, 1).to(args.device)
            interpolated = eta * output_real.data + ((1 - eta) * output_fake.data)
            interpolated = interpolated.to(args.device)
            interpolated.requires_grad_(True)
            # calculate probability of interpolated examples
            prob_interpolated = model_d(interpolated, params, obs, pos, scale)
            # calculate gradients of probabilities with respect to examples
            gradients = autograd.grad(
                outputs=prob_interpolated,
                inputs=interpolated,
                grad_outputs=torch.ones(prob_interpolated.size()).to(args.device),
                create_graph=True,
                retain_graph=True
            )[0]
            grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
            grad_penalty.backward()
            lossD = loss_real + loss_fake
            # torch.nn.utils.clip_grad_value_(model_d.parameters(), 0.01)
            optimizer_d.step()

        # Update G to maximize log(D(G(z)))
        for p in model_d.parameters():
            p.requires_grad = False
        for g_iter in range(args.g_steps):
            optimizer_g.zero_grad()
            noise = torch.randn(args.batch_size, 64, dtype=torch.float, device=args.device)
            scale = (torch.rand((args.batch_size, 1)) * 0.3).to(args.device)
            # fake = model_g.expectation_from_prs(params_cvt, obs_kron, pos, model_g(noise, params, obs, pos, scale))
            fake = model_g(noise, params, obs, pos, scale)
            ptrue = model_d(fake, params, obs, pos, scale)
            lossG = -ptrue.mean()  # + args.gamma * value.mean()
            lossG.backward()
            # torch.nn.utils.clip_grad_value_(model_g.parameters(), 0.01)
            optimizer_g.step()

        if itr % 1000 == 0:
            # args.writer.add_scalar('Loss/train', loss_accumulator.getval(), epoch)
            print('Loss_D: {:.6f}\tLoss_G\t{:.6f}'.format(lossD.item(), lossG.item()))


@torch.no_grad()
def validate(epoch, args, loader, model_g, loss_fn):
    print('Validating...')
    model_g.eval()
    metric = AverageMeter()
    for itr, (params, params_cvt, obs, obs_kron, pos, scale, exp_noisy, gts) in enumerate(loader):
        params_cvt = params_cvt.to(args.device)
        params, obs, pos, gts = params.to(args.device), obs.to(args.device), pos.to(args.device), gts.to(args.device)
        obs_kron = obs_kron.to(args.device)
        exp_noisy = exp_noisy.to(args.device)
        scale = scale.to(args.device)
        # scale = torch.full((len(params), 1), 0.0, dtype=torch.float, device=args.device)
        # predicts = []
        # for _ in range(args.num_samples):
        noise = torch.randn(len(params), 64, dtype=torch.float, device=args.device)
        # prs = model_g(noise, params, obs, pos, scale)
        # preds = model_g.expectation_from_prs(params_cvt, obs_kron, pos, prs)
        preds = model_g(noise, params, obs, pos, scale)
        # predicts.append(preds)
        # predicts = torch.stack(predicts).mean(0)
        metric.update(abs_deviation(preds, exp_noisy))

    value = metric.getval()
    # args.writer.add_scalar('Loss/val', losses.getval(), epoch)
    args.writer.add_scalar('Abs_deviation/val', value, epoch)
    print('validation absolute deviation: {:.6f}'.format(value))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MitigateDataset', type=str, help='[MitigateDataset]')
    parser.add_argument('--train-path', default='../data_mitigate/gate_dependent_ad/dataset_vqe4l.pkl', type=str)
    parser.add_argument('--test-path', default='../data_mitigate/gate_dependent_ad/dataset_vqe4l.pkl', type=str)
    parser.add_argument('--env-path', default='../environments/gate_dependent_ad/vqe_envs_train_4l', type=str)
    parser.add_argument('--logdir', default='../runs', type=str, help='path to save logs and models')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-mitigates', default=4, type=int, help='number of mitigation gates')
    parser.add_argument('--num-samples', default=30, type=int, help='number of samples to be averaged')
    parser.add_argument('--num-ops', default=2, type=int)
    parser.add_argument('--workers', default=4, type=int, help='dataloader worker nums')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--g_steps', default=1, type=float, help='steps for generator updates')
    parser.add_argument('--d_steps', default=5, type=float, help='steps for discriminator updates')
    parser.add_argument('--lr-g', default=5e-5, type=float, help='learning rate for generator')
    parser.add_argument('--lr-d', default=5e-5, type=float, help='learning rate for discriminator')
    parser.add_argument('--gamma', default=0.0, type=float, help='constraint for discriminator')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', default=0.9, type=float, help='beta2 for Adam optimizer')
    parser.add_argument('--thres', default=0.12, type=float, help='threshold to saving model')
    parser.add_argument('--save-name', default='gan_model.pt', type=str, help='model file name')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(args.device)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    args.logdir = os.path.join(args.logdir, f'env_vqe_gan_ampdamp_{time_str}')
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    args.writer = SummaryWriter(log_dir=args.logdir)
    main(args)
    args.writer.flush()
    args.writer.close()
