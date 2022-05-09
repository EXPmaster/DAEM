import torch
import sys
sys.path.append('exp_gan')
from exp_gan.datasets import MitigateDataset
from exp_torch.utils import AverageMeter


if __name__ == '__main__':
    data_path = 'data_mitigate/testset_3.pkl'
    dataset = MitigateDataset(data_path)
    print(len(dataset))

    deviation = AverageMeter()
    for obs, exp_noisy, exp_ideal in dataset:
        deviation.update(torch.abs(exp_noisy - exp_ideal).item())

    print('deviation in original dataset: {:.6f}'.format(deviation.getval()))
    
