import torch
import torch.nn as nn
import torch.nn.functional as F


def vec2density(x):
    return torch.bmm(x[:, :, None], x[:, None]) / torch.bmm(x[:, None], x[:, :, None])


def to_statevec(x):
    return x / torch.sqrt(torch.sum(torch.abs(x) ** 2, 0, keepdim=True))


def measure(o, rho):
    return torch.trace(o, rho)


class MishComplex(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, input):
        return F.mish(input.real, inplace=self.inplace).cfloat() + 1j * F.mish(input.imag, inplace=self.inplace).cfloat()


class ToStateVec(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return to_statevec(input)


if __name__ == '__main__':
    mc = MishComplex()
    a = torch.randn((4, 3), dtype=torch.cfloat, requires_grad=True)
    b = mc(a)
    c = a.sum()
    c.backward()
    print(a)
    print(torch.abs(a))
    print(torch.sum(torch.abs(a), 0, keepdim=True))