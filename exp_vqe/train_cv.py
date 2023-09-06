import os
import time
import pickle
import argparse

import numpy as np
import torch
from torch import autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from qiskit.quantum_info.operators import Pauli

from model import CvModel 
from utils import AverageMeter, build_dataloader2, abs_deviation
from datasets import MitigateDataset, CvDataset


def main(args):
    trainset, testset, train_loader, test_loader = build_dataloader2(args, eval(args.dataset))
    print(len(trainset))
    if args.miti_prob:
        # loss_fn = nn.CrossEntropyLoss()
        # loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=False)
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.MSELoss()
    model = CvModel()
    model.to(args.device)
    # optimizer_g = optim.Adam(model_g.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    print('Start training...')
    # scheduler = StepLR(optimizer, 30, gamma=0.5)

    best_metric = args.thres
    for epoch in range(args.epochs):
        print(f'=> Epoch {epoch}')
        train(epoch, args, train_loader, model, loss_fn, optimizer)
        # if epoch % 10 == 0:
        metric = validate(epoch, args, test_loader, model, loss_fn)
        # scheduler.step()
        model_save_flag = False
        if args.miti_prob and metric > 0.9:
            if metric > best_metric:
                model_save_flag = True
        elif not args.miti_prob and metric < best_metric:
            model_save_flag = True
        if model_save_flag:
            print('Saving model...')
            best_metric = metric
            ckpt = {
                'model_g': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(args.logdir, args.save_name))
    with open(os.path.join(args.logdir, 'metric_gan.txt'), 'a+') as f:
        f.write('{} {:.6f}\n'.format(len(trainset), best_metric))


def train(epoch, args, loader, model, loss_fn, optimizer):
    model.train()
    for itr, (params, exp_noisy, gts) in enumerate(loader):
        params = params.to(args.device)
        gts = gts.to(args.device)
        exp_noisy = exp_noisy.to(args.device)
        optimizer.zero_grad()
        fake = model(params, exp_noisy).flatten(1)
        # fake = F.log_softmax(fake, dim=1)
        loss_real = loss_fn(fake, gts.flatten(1))
        loss_real.backward()
        optimizer.step()

        if itr == 0:
            # args.writer.add_scalar('Loss/train', loss_accumulator.getval(), epoch)
            # print('Loss_D: {:.6f}\tLoss_G\t{:.6f}'.format(loss_real.item(), loss_gradient.item()))
            print('Loss_D: {:.6f}\tLoss_G\t{:.6f}'.format(loss_real.item(), 0.))


@torch.no_grad()
def validate(epoch, args, loader, model, loss_fn):
    print('Validating...')
    model.eval()
    distance = nn.CosineSimilarity()
    metric = AverageMeter()
    unmitigated_metric = AverageMeter()
    for itr, (params, exp_noisy, gts) in enumerate(loader):
        params = params.to(args.device)
        gts = gts.to(args.device)
        exp_noisy = exp_noisy.to(args.device)
        preds = model(params, exp_noisy)
        if args.miti_prob:
            diff = distance(preds.flatten(1), gts.flatten(1)).mean().item()
            unmitigated_metric.update(distance(exp_noisy[:, 0].flatten(1), gts.flatten(1)).mean().item())
        else:
            diff = abs_deviation(preds, gts)
        metric.update(diff)

    value = metric.getval()
    # args.writer.add_scalar('Loss/val', losses.getval(), epoch)
    args.writer.add_scalar('Abs_deviation/val', value, epoch)
    print('validation metric: {:.6f}'.format(value))
    print('unmitigated metric: {:.6f}'.format(unmitigated_metric.getval()))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CvDataset', type=str, help='[CvDataset]')
    parser.add_argument('--miti-prob', action='store_true', default=True, help='mitigate probability distribution or expectation')
    parser.add_argument('--train-path', default='../data_mitigate/continuous_variable/new_train_coherent.pkl', type=str)
    parser.add_argument('--test-path', default='../data_mitigate/continuous_variable/new_test_coherent.pkl', type=str)
    parser.add_argument('--logdir', default='../runs', type=str, help='path to save logs and models')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--workers', default=4, type=int, help='dataloader worker nums')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--p_steps', default=5, type=float, help='steps for penalty updates')
    parser.add_argument('--gamma', default=4.0, type=float, help='constraint for discriminator')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', default=0.99, type=float, help='beta2 for Adam optimizer')
    parser.add_argument('--thres', default=0.12, type=float, help='threshold to saving model')
    parser.add_argument('--save-name', default='gan_model.pt', type=str, help='model file name')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(args.device)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    args.logdir = os.path.join(args.logdir, f'env_cv_new_photonloss_{time_str}')
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    args.writer = SummaryWriter(log_dir=args.logdir)
    main(args)
    args.writer.flush()
    args.writer.close()
