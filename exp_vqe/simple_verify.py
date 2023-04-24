import os
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils import AverageMeter, abs_deviation


COEFF = torch.rand(3)
print(COEFF)


class SimpleDataset(Dataset):
    
    def __init__(self, stype='train'):
        if stype == 'train':
            x = torch.arange(0.5, 2., 0.01)
        else:
            x = torch.arange(0.0, 2., 0.001)
        print(f'{stype}set len: {len(x)}')
        ones = torch.full_like(x, 1.)
        xs = torch.stack([ones, x, (x) ** 2], 1)
        self.dataset = x
        self.labels = xs @ COEFF[:, None]

    def __getitem__(self, idx):
        return self.dataset[idx][None], self.labels[idx]

    def __len__(self):
        return len(self.dataset)


class SimpleG(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.ebd = nn.Linear(1, 20)
        self.net = nn.Sequential(
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
    
    def forward(self, noise, x):
        x = self.ebd(x)
        # x = torch.cat([noise, x], 1)
        return self.net(x)


class SimpleD(nn.Module):

    def __init__(self):
        super().__init__()
        self.ebd_data = nn.Linear(1, 10)
        self.ebd_condition = nn.Linear(1, 10)
        self.net = nn.Sequential(
            nn.Linear(20, 30),
            nn.LeakyReLU(0.1),
            nn.Linear(30, 30),
             nn.LeakyReLU(0.1),
            nn.Linear(30, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, condition):
        x = self.ebd_data(x)
        cd = self.ebd_condition(condition)
        return self.net(torch.cat([x, cd], 1))
    

def main(args):
    trainset = SimpleDataset('train')
    testset = SimpleDataset('test')
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    loss_fn = nn.BCELoss()
    model_g = SimpleG().to(args.device)
    model_d = SimpleD().to(args.device)
    optimizer_g = optim.Adam(model_g.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_d = optim.Adam(model_d.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    print('Start training...')
    scheduler_g = StepLR(optimizer_g, 50, gamma=0.5)
    scheduler_d = StepLR(optimizer_d, 50, gamma=0.5)

    for epoch in range(args.epochs):
        print(f'=> Epoch {epoch}')
        train(epoch, args, train_loader, model_g, model_d, loss_fn, optimizer_g, optimizer_d)
        metric = validate(epoch, args, test_loader, model_g, loss_fn)
        # scheduler_g.step()
        # scheduler_d.step()


def train(epoch, args, loader, model_g, model_d, loss_fn, optimizer_g, optimizer_d):
    model_g.train()
    model_d.train()
    loss_g_avg = AverageMeter()
    loss_d_avg = AverageMeter()

    for itr, (data, gts) in enumerate(loader):
        # Update D to maximize log(D(x)) + log(1 - D(G(z)))
        ## real
        data = data.to(args.device)
        gts = gts.to(args.device)
        optimizer_d.zero_grad()
        labels = torch.full((len(data), 1), 1., dtype=torch.float, device=args.device)
        output = model_d(gts, data)
        D_ideal = output.mean().item()
        lossD_real = loss_fn(output, labels)
        lossD_real.backward()

        ## fake
        noise = torch.randn(len(data), 10, dtype=torch.float, device=args.device)
        labels.fill_(0.0)
        data = torch.rand(len(data), 1, dtype=torch.float, device=args.device) * 2
        fake = model_g(noise, data)
        output = model_d(fake.detach(), data)
        D_g_z1 = output.mean().item()
        lossD_fake1 = loss_fn(output, labels)
        lossD_fake1.backward()
        # torch.nn.utils.clip_grad_norm_(model_d.parameters(), 1.)
        optimizer_d.step()
        lossD = lossD_real + lossD_fake1

        # Update G to maximize log(D(G(z)))
        optimizer_g.zero_grad()
        labels.fill_(1.0)
        output = model_d(fake, data)
        D_g_z2 = output.mean().item()
        lossG = loss_fn(output, labels)
        lossG.backward()
        # torch.nn.utils.clip_grad_norm_(model_g.parameters(), 1.)
        optimizer_g.step()

        loss_g_avg.update(lossG.item())
        loss_d_avg.update(lossD.item())
        # print('Loss_D: {:.4f}\tLoss_G\t{:.4f}\tD(ideal): {:.4f}\tD(G(z)): {:.4f} / {:.4f}'.format(lossD, lossG, D_ideal, D_g_z1, D_g_z2))


@torch.no_grad()
def validate(epoch, args, loader, model_g, loss_fn):
    print('Validating...')
    model_g.eval()
    metric = AverageMeter()
    model_predicts = []
    ground_truth = []
    for itr, (data, gts) in enumerate(loader):
        data = data.to(args.device)
        gts = gts.to(args.device)
        predicts = []
        for _ in range(args.num_samples):
            noise = torch.randn(len(data), 10, dtype=torch.float, device=args.device)
            preds = model_g(noise, data)
            predicts.append(preds)
        predicts = torch.stack(predicts).mean(0)
        
        metric.update(abs_deviation(predicts, gts))
        model_predicts.extend(preds.squeeze().tolist())
        ground_truth.extend(gts.squeeze().tolist())
    if epoch % 10 == 0:
        for i in range(len(model_predicts)):
            args.writer.add_scalars(f'runs/run_epoch_{epoch}', {
                                        'y_true': ground_truth[i],
                                        'y_pred': model_predicts[i]}, i)
    value = metric.getval()
    print('validation absolute deviation: {:.6f}'.format(value))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--num-samples', default=100, type=int, help='number of samples to be averaged')
    parser.add_argument('--workers', default=4, type=int, help='dataloader worker nums')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--lr-g', default=5e-4, type=float, help='learning rate for generator')
    parser.add_argument('--lr-d', default=5e-4, type=float, help='learning rate for discriminator')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', default=0.99, type=float, help='beta2 for Adam optimizer')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(args.device)
    args.writer = SummaryWriter(log_dir='../runs/simple_test/exp1')
    main(args)
    args.writer.close()
