import torch
from torch.utils.data import Dataset
import pickle
import numpy as np


class SurrogateDataset(Dataset):

    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        self.probabilities = data_dict['probability']
        self.observables = data_dict['observable']
        self.ground_truth = data_dict['meas_result']

    def __getitem__(self, idx):
        prs, obs, gts = self.probabilities[idx], self.observables[idx], self.ground_truth[idx]
        obs = np.array([[1.0, 0.0], [0.0, -1.0]])
        return torch.FloatTensor(prs), torch.tensor(obs, dtype=torch.cfloat), torch.FloatTensor([gts])

    def __len__(self):
        return len(self.probabilities)


if __name__ == '__main__':
    dataset = SurrogateDataset('../data_surrogate/env1_data.pkl')
    print(next(iter(dataset)))