import torch
from torch.utils.data import DataLoader


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
    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    testset = dataset(args.test_path)
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    return trainset, testset, train_loader, test_loader


def model_summary(model):
    from torchsummary import summary
    ch = model.num_layers if hasattr(model, 'num_layers') else 5
    wh = model.param_dim if hasattr(model, 'num_layers') else int(model.param_dim ** 0.5)
    summary(model, [(ch, wh, wh), (ch, wh, wh), (wh, wh), (wh, wh)], batch_size=1)


def abs_deviation(x1, x2):
    """Calculate mean absolute deviation"""
    assert x1.shape == x2.shape
    return torch.abs(x1 - x2).mean().item()


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
