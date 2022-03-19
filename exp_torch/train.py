import os
import pickle
import argparse
import torch
import torch.optim as optim
import torch.nn as nn

from model import QuantumModel, QuantumModelv2
from utils import build_dataloader, AverageMeter, abs_deviation
from torch.utils.tensorboard import SummaryWriter


def main(args):
    trainset, testset, train_loader, test_loader = build_dataloader(args)
    loss_fn = nn.MSELoss()
    print(f'Model type: {args.model_type}.')
    model = eval(args.model_type)(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print('testing initial model...')
    validate(0, args, test_loader, model, loss_fn)
    print('Start training...')

    best_metric = 1.0
    for epoch in range(args.epochs):
        print(f'=> Epoch {epoch}')
        train(epoch, args, train_loader, model, loss_fn, optimizer)
        metric = validate(epoch, args, test_loader, model, loss_fn)

        if metric < best_metric:
            print('Saving model...')
            best_metric = metric
            torch.save(model, os.path.join(args.logdir, 'best.pt'))


def train(epoch, args, loader, model, loss_fn, optimizer):
    model.train()
    loss_accumulator = AverageMeter()
    for itr, (moment_real, moment_imag, ops_real, ops_imag, e_ideal, e_noisy, delta) in enumerate(loader):
        moment_real, moment_imag = moment_real.to(args.device), moment_imag.to(args.device)
        ops_real, ops_imag, delta = ops_real.to(args.device), ops_imag.to(args.device), delta.to(args.device)
        optimizer.zero_grad()
        predicts = model(moment_real, moment_imag, ops_real, ops_imag)
        loss = loss_fn(predicts, delta)
        loss_accumulator.update(loss)
        loss.backward()
        optimizer.step()

    args.writer.add_scalar('Loss/train', loss_accumulator.getval(), epoch)
    print('training loss: {:.6f}'.format(loss_accumulator.getval()))


@torch.no_grad()
def validate(epoch, args, loader, model, loss_fn):
    model.eval()
    metric = AverageMeter()
    losses = AverageMeter()
    for itr, (moment_real, moment_imag, ops_real, ops_imag, e_ideal, e_noisy, delta) in enumerate(loader):
        moment_real, moment_imag = moment_real.to(args.device), moment_imag.to(args.device)
        ops_real, ops_imag, e_ideal, e_noisy, delta = ops_real.to(args.device), ops_imag.to(args.device), e_ideal.to(args.device), e_noisy.to(args.device), delta.to(args.device)
        predicts = model(moment_real, moment_imag, ops_real, ops_imag)
        loss = loss_fn(predicts, delta)
        mitigate_results = e_noisy + predicts
        metric.update(abs_deviation(mitigate_results, e_ideal))
        losses.update(loss)

    value = metric.getval()
    args.writer.add_scalar('Loss/val', losses.getval(), epoch)
    args.writer.add_scalar('Abs_deviation/val', value, epoch)
    print('validation absolute deviation: {:.6f}'.format(value))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', default='data2/trainset_1.pkl', type=str)
    parser.add_argument('--test-path', default='data2/testset_1.pkl', type=str)
    parser.add_argument('--logdir', default='runs/v1', type=str, help='path to save logs and models')
    parser.add_argument('--model-type', default='QuantumModelv2', type=str, help='what model to use: [QuantumModel, QuantumModelv2]')
    parser.add_argument('--model-path', default='weights', type=str, help='duplicated')
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--num-layers', default=5, type=int, help='depth of the circuit')
    parser.add_argument('--num-qubits', default=4, type=int, help='number of qubits')
    parser.add_argument('--workers', default=8, type=int, help='dataloader worker nums')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.writer = SummaryWriter(log_dir=args.logdir)
    main(args)
    args.writer.flush()
    args.writer.close()
