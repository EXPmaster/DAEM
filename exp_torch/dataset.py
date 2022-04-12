import torch
from torch.utils.data import Dataset
import pickle
import numpy as np


class QuantumDataset(Dataset):

    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.dataset = pickle.load(f)

    def __getitem__(self, idx):
        total_ops, moment_ops, observable, e_ideal, e_noisy = self.dataset[idx]
        # real_moments = torch.FloatTensor(np.real(moment_ops))
        # imag_moments = torch.FloatTensor(np.imag(moment_ops))
        # real_all = torch.FloatTensor(np.real(total_ops))
        # imag_all = torch.FloatTensor(np.imag(total_ops))
        moments = torch.tensor(moment_ops, dtype=torch.cfloat)
        total = torch.tensor(total_ops, dtype=torch.cfloat)
        observable = torch.tensor(observable, dtype=torch.cfloat)
        return moments, total, observable, \
                torch.FloatTensor([e_ideal]), torch.FloatTensor([e_noisy]), torch.FloatTensor([e_ideal-e_noisy])

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dataset = QuantumDataset('data2/dataset_0.pkl')
    print(next(iter(dataset)))