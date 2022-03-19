import torch
from exp_torch.dataset import QuantumDataset
from exp_torch.utils import AverageMeter


if __name__ == '__main__':
    data_path = 'data2/dataset_0.pkl'
    dataset = QuantumDataset(data_path)
    print(len(dataset))

    deviation = AverageMeter()
    for _, _, _, _, _, _, delta in dataset:
        deviation.update(torch.abs(delta).item())

    print('deviation in original dataset: {:.6f}'.format(deviation.getval()))
    
