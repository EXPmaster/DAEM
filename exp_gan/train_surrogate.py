import os
import pickle
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import AverageMeter, build_dataloader, abs_deviation
from datasets import SurrogateGenerator


def main(args):
    loader = SurrogateGenerator(args.env_path, args.batch_size)
    loss_fn = nn.MSELoss()
    print(f'Model type: {args.model_type}.')
    model = SurrogateModel(dim_in=4 * loader.num_miti_gates + 8).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print('Start training...')

    best_metric = 0.075
    for epoch in range(args.epochs):
        print(f'=> Epoch {epoch}')
        train(epoch, args, loader, model, loss_fn, optimizer)
        metric = validate(epoch, args, loader, model, loss_fn)

        if metric < best_metric:
            print('Saving model...')
            best_metric = metric
            torch.save(model.state_dict(), os.path.join(args.logdir, 'model_surrogate.pt'))


def train(epoch, args, loader, model, loss_fn, optimizer):
    model.train()
    loss_accumulator = AverageMeter()
    for itr, (prs, obs, gts) in enumerate(loader):
        prs, obs, gts = prs.to(args.device), obs.to(args.device), gts.to(args.device)
        optimizer.zero_grad()
        predicts = model(prs, obs)
        loss = loss_fn(predicts, gts)
        loss_accumulator.update(loss)
        loss.backward()
        optimizer.step()

    # args.writer.add_scalar('Loss/train', loss_accumulator.getval(), epoch)
    print('training loss: {:.6f}'.format(loss_accumulator.getval()))


@torch.no_grad()
def validate(epoch, args, loader, model, loss_fn):
    model.eval()
    metric = AverageMeter()
    loader.cur_itr = 30
    for itr,(prs, obs, gts) in enumerate(loader):
        prs, obs, gts = prs.to(args.device), obs.to(args.device), gts.to(args.device)
        predicts = model(prs, obs)
        metric.update(abs_deviation(predicts, gts))

    value = metric.getval()
    # args.writer.add_scalar('Loss/val', losses.getval(), epoch)
    # args.writer.add_scalar('Abs_deviation/val', value, epoch)
    print('validation absolute deviation: {:.6f}'.format(value))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train-path', default='../data_surrogate/env1_data.pkl', type=str)
    # parser.add_argument('--test-path', default='../data_surrogate/env1_data.pkl', type=str)
    parser.add_argument('--env-path', default='../environments/ibmq_random.pkl', type=str)
    parser.add_argument('--logdir', default='../runs/env_ibmq_random', type=str, help='path to save logs and models')
    parser.add_argument('--model-type', default='SurrogateModel', type=str, help='what model to use: [SurrogateModel]')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--workers', default=8, type=int, help='dataloader worker nums')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # args.writer = SummaryWriter(log_dir=args.logdir)
    main(args)
    # args.writer.flush()
    # args.writer.close()
