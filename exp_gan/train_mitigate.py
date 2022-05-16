import os
import pickle
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import AverageMeter, build_dataloader, abs_deviation
from datasets import MitigateDataset


def main(args):
    trainset, testset, train_loader, test_loader = build_dataloader(args, MitigateDataset)
    loss_fn = nn.MSELoss()
    model_s = SurrogateModel(dim_in=4 * args.num_mitigates + 8).to(args.device)
    model_s.load_state_dict(torch.load(args.weight_path, map_location=args.device))
    model_g = MitigateModel(dim_in=8, dim_out=args.num_mitigates).to(args.device)
    optimizer = optim.Adam([{'params': model_g.parameters()},
                            {'params': model_s.parameters(), 'lr': 1e-6}], lr=args.lr)
    print('Start training...')

    best_metric = 1.0
    for epoch in range(args.epochs):
        print(f'=> Epoch {epoch}')
        train(epoch, args, train_loader, model_g, model_s, loss_fn, optimizer)
        metric = validate(epoch, args, test_loader, model_g, model_s, loss_fn)
        if metric < best_metric:
            print('Saving model...')
            best_metric = metric
            ckpt = {
                'model_g': model_g.state_dict(),
                'model_s': model_s.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(ckpt, os.path.join(args.logdir, args.save_name))

    with open(os.path.join(args.logdir, 'metric_supervise.txt'), 'a+') as f:
        f.write('{} {:.6f}\n'.format(len(trainset), best_metric))


def train(epoch, args, loader, model_g, model_s, loss_fn, optimizer):
    model_g.train()
    model_s.train()
    loss_accumulator = AverageMeter()
    for itr, (obs, exp_noisy, gts) in enumerate(loader):
        obs, exp_noisy, gts = obs.to(args.device), exp_noisy.to(args.device), gts.to(args.device)
        optimizer.zero_grad()
        prs = model_g(obs, exp_noisy)
        predicts = model_s(prs, obs)
        loss = loss_fn(predicts, gts)
        loss_accumulator.update(loss)
        loss.backward()
        optimizer.step()

    # args.writer.add_scalar('Loss/train', loss_accumulator.getval(), epoch)
    print('training loss: {:.6f}'.format(loss_accumulator.getval()))


@torch.no_grad()
def validate(epoch, args, loader, model_g, model_s, loss_fn):
    model_g.eval()
    model_s.eval()
    metric = AverageMeter()
    for itr, (obs, exp_noisy, gts) in enumerate(loader):
        obs, exp_noisy, gts = obs.to(args.device), exp_noisy.to(args.device), gts.to(args.device)
        prs = model_g(obs, exp_noisy)
        predicts = model_s(prs, obs)
        metric.update(abs_deviation(predicts, gts))

    value = metric.getval()
    # args.writer.add_scalar('Loss/val', losses.getval(), epoch)
    # args.writer.add_scalar('Abs_deviation/val', value, epoch)
    print('validation absolute deviation: {:.6f}'.format(value))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', default='../data_mitigate/trainset_2.pkl', type=str)
    parser.add_argument('--test-path', default='../data_mitigate/testset_2.pkl', type=str)
    parser.add_argument('--weight-path', default='../runs/env_ibmq/model_surrogate.pt', type=str)
    parser.add_argument('--logdir', default='../runs/env_ibmq_random', type=str, help='path to save logs and models')
    parser.add_argument('--model-type', default='SurrogateModel', type=str, help='what model to use: [SurrogateModel]')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--num-mitigates', default=8, type=int, help='number of mitigation gates')
    parser.add_argument('--workers', default=8, type=int, help='dataloader worker nums')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nosave', default=False, action='store_true', help='not to save model')
    parser.add_argument('--save-name', default='mitigation_model.pt', type=str, help='model file name')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # args.writer = SummaryWriter(log_dir=args.logdir)
    main(args)
    # args.writer.flush()
    # args.writer.close()
