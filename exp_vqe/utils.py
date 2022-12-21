import torch
import numpy as np
from torch.utils.data import DataLoader
from qiskit.quantum_info.operators import Pauli


class ConfigDict(dict):

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    
    def __setattr__(self, name, value):
        self[name] = value
    
    def __delattr__(self, name):
        del self[name]


def build_dataloader(args, dataset):
    trainset = dataset(args.train_path)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    testset = dataset(args.test_path)
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, drop_last=True)

    return trainset, testset, train_loader, test_loader


def gen_rand_obs(dim=(2, 2)):
    rand_matrix = (torch.rand(dim, dtype=torch.cfloat) * 2 - 1).numpy()
    rand_obs = (rand_matrix.conj().T + rand_matrix) / 2
    eigen_val = np.linalg.eigvalsh(rand_obs)
    rand_obs = rand_obs / np.max(np.abs(eigen_val))
    return rand_obs


def gen_rand_pauli(dim=(2, 2)):
    paulis = ['I', 'X', 'Y', 'Z']
    pauli_str = np.random.choice(paulis)
    return Pauli(pauli_str).to_matrix()


def model_summary(model):
    from torchsummary import summary
    ch = model.num_layers if hasattr(model, 'num_layers') else 5
    wh = model.param_dim if hasattr(model, 'num_layers') else int(model.param_dim ** 0.5)
    summary(model, [(ch, wh, wh), (ch, wh, wh), (wh, wh), (wh, wh)], batch_size=1)


def abs_deviation(x1, x2):
    """Calculate mean absolute deviation"""
    assert x1.shape == x2.shape
    return torch.abs(x1 - x2).mean().item()


def partial_trace(rho, keep, dims, optimize=False):
    """Calculate the partial trace

    ρ_a = Tr_b(ρ)

    Parameters
    ----------
    ρ : 2D array
        Matrix to trace
    keep : array
        An array of indices of the spaces to keep after
        being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D,
        keep = [0,2]
    dims : array
        An array of the dimensions of each space.
        For instance, if the space is A x B x C x D,
        dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    ρ_a : 2D array
        Traced matrix
    """
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim + i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims, 2))
    rho_a = np.einsum(rho_a, idx1 + idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = self._sum / self._count if self._count != 0 else 0

    def getval(self):
        return self._avg

    def __str__(self):
        if not hasattr(self, 'val'):
            return 'None.'
        return str(self.getval())
