import os
import pickle
import argparse
import torch
from torch.utils.data import DataLoader

from model import QuantumModel, QuantumModelv2
from utils import AverageMeter, abs_deviation, model_summary
from dataset import QuantumDataset


def test(args):
    model = torch.load(args.model_path, map_location=args.device)
    model.eval()
    model.requires_grad_(False)
    if args.summary:
        model_summary(model)
    loader = DataLoader(QuantumDataset(args.test_path), batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    metric = AverageMeter()
    with torch.no_grad():
        for itr, (moments, ops, e_ideal, e_noisy, delta) in enumerate(loader):
            moments = moments.to(args.device)
            ops, e_ideal, e_noisy = ops.to(args.device), e_ideal.to(args.device), e_noisy.to(args.device)
            predicts = model(moments, ops)
            mitigate_results = e_noisy + predicts
            metric.update(abs_deviation(mitigate_results, e_ideal))

    value = metric.getval()
    print('validation absolute deviation: {:.6f}'.format(value))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-path', default='data2/dataset_0.pkl', type=str)
    parser.add_argument('--model-path', default='runs/v1_4/best.pt', type=str)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--summary', default=False, action='store_true', help='print out model structure & params')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test(args)