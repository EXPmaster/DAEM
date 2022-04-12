from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MishComplex, ToStateVec


class QuantumModel(nn.Module):

    def __init__(self, args):
        # parameter number: ~ 852096
        super().__init__()
        self.param_dim = (2 ** args.num_qubits) ** 2
        self.input_real_ebd = nn.Linear(self.param_dim, self.param_dim)
        self.input_imag_ebd = nn.Linear(self.param_dim, self.param_dim)

        self.head = nn.Sequential(
            nn.Linear(self.param_dim * 2, self.param_dim),
            nn.Mish(inplace=True),
            nn.Linear(self.param_dim, self.param_dim),
            nn.Mish(inplace=True),
            nn.Linear(self.param_dim, 1)
        )

    def forward(self, x, u):
        r_ebd = self.input_real_ebd(torch.flatten(u.real, 1))
        i_ebd = self.input_imag_ebd(torch.flatten(u.imag, 1))
        concat_ebd = torch.cat((r_ebd, i_ebd), -1)
        return self.head(concat_ebd)


class QuantumModelv2(nn.Module):

    def __init__(self, args):
        # parameter number: ~ 165504
        super().__init__()
        self.num_layers = args.num_layers
        self.param_dim = 2 ** args.num_qubits
        self.head_input_dim = 2 * self.param_dim ** 2
        self.modellist_real = nn.ParameterList([nn.Parameter(torch.eye(self.param_dim, dtype=torch.cfloat)) for _ in range(self.num_layers)])
        # self.modellist_real = nn.ParameterList([nn.Parameter(torch.randn((self.param_dim, self.param_dim), dtype=torch.cfloat)) for _ in range(self.num_layers)])

        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, self.head_input_dim // 2),
            nn.Mish(inplace=True),
            nn.Linear(self.head_input_dim // 2, self.head_input_dim // 2),
            nn.Mish(inplace=True),
            nn.Linear(self.head_input_dim // 2, 1)
        )

    def forward(self, x, u, o):
        """x: moment operators, u: overall operators, o: observables"""
        # out_r = x_r[:, 0]
        # out_i = x_i[:, 0]
        # for i in range(self.num_layers - 1):
        #     temp_r = self.modellist_real[i] @ out_r
        #     temp_i = self.modellist_imag[i] @ out_i
        #     out_r = x_r[:, i + 1] @ temp_r
        #     out_i = x_i[:, i + 1] @ temp_i
        
        # out_r = self.modellist_real[-1] @ out_r
        # out_i = self.modellist_imag[-1] @ out_i
        out = x[:, 0]
        for i in range(self.num_layers - 1):
            temp = self.modellist_real[i] @ out
            out = x[:, i + 1] @ temp
        
        out = self.modellist_real[-1] @ out
        out = u - out  # o @ (u - out)
        concat_out = torch.cat((out.real, out.imag), -1).flatten(1)
        # concat_out = torch.cat((u_r - out_r, u_i - out_i), -1).flatten(1)
        return self.head(concat_out)


class QuantumModelv3(nn.Module):

    def __init__(self, args):
        # parameter number: ~ 165504
        super().__init__()
        self.num_layers = args.num_layers
        self.param_dim = 2 ** args.num_qubits
        self.head_input_dim = 2 * self.param_dim ** 2
        # self.modellist_real = nn.Parameter(torch.eye(self.param_dim, dtype=torch.cfloat))
        self.modellist_real = nn.ParameterList([nn.Parameter(torch.eye(self.param_dim, dtype=torch.cfloat)) for _ in range(self.num_layers)])

        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, self.head_input_dim // 2),
            nn.Mish(inplace=True),
            nn.Linear(self.head_input_dim // 2, self.head_input_dim // 2),
            nn.Mish(inplace=True),
            nn.Linear(self.head_input_dim // 2, 1)
        )
        self.o_ebd = nn.Linear(8, self.head_input_dim // 2)

    def forward(self, x, u, o):
        """x: moment operators, u: overall operators, o: observables"""
        out = x[:, 0]
        for i in range(self.num_layers - 1):
            temp = self.modellist_real[i] @ out
            out = x[:, i + 1] @ temp
        
        out = self.modellist_real[-1] @ out
        out = u - out
        o_embedding = self.o_ebd(torch.cat((o.real, o.imag), -1).flatten(1))
        concat_out = torch.cat((torch.cat((out.real, out.imag), -1).flatten(1), o_embedding), -1)
        # concat_out = torch.cat((out.real, out.imag, o.real, o.imag), -1).flatten(1)
        return self.head(concat_out)


class QuantumModelv4(nn.Module):

    def __init__(self, args):
        # parameter number: ~ 165504
        super().__init__()
        self.num_layers = args.num_layers
        self.param_dim = 2 ** args.num_qubits
        self.head_input_dim = 4 * self.param_dim ** 2
        # self.modellist_real = nn.Parameter(torch.eye(self.param_dim, dtype=torch.cfloat))
        self.modellist_real = nn.ParameterList([nn.Parameter(torch.eye(self.param_dim, dtype=torch.cfloat)) for _ in range(self.num_layers)])

        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, self.head_input_dim // 2, dtype=torch.cfloat),
            MishComplex(inplace=True),
            nn.Linear(self.head_input_dim // 2, self.head_input_dim // 2, dtype=torch.cfloat),
            MishComplex(inplace=True),
            nn.Linear(self.head_input_dim // 2, self.param_dim, dtype=torch.cfloat),
            ToStateVec()
        )
        self.o_ebd = nn.Linear(8, self.head_input_dim // 2)

    def forward(self, x, u, o):
        """x: moment operators, u: overall operators, o: observables"""
        out = x[:, 0]
        for i in range(self.num_layers - 1):
            temp = self.modellist_real[i] @ out
            out = x[:, i + 1] @ temp
        
        out = self.modellist_real[-1] @ out
        out = u - out
        o_embedding = self.o_ebd(torch.cat((o.real, o.imag), -1).flatten(1))
        concat_out = torch.cat((torch.cat((out.real, out.imag), -1).flatten(1), o_embedding), -1)
        # concat_out = torch.cat((out.real, out.imag, o.real, o.imag), -1).flatten(1)
        return self.head(concat_out)


if __name__ == '__main__':
    # ArgsClass = namedtuple('args', ['num_layers'])
    # args = ArgsClass(6)
    # input1 = torch.randn((12, 6, 16, 16))
    # input2 = torch.randn((12, 6, 16, 16))
    # model = QuantumModelv2(args)
    # model(input1, input2)

    # a = torch.eye(3, dtype=torch.cfloat, requires_grad=True)
    # b = torch.randn((3, 3), dtype=torch.cfloat)
    # d = a @ b
    # d_r = torch.real(d)
    # d_i = torch.imag(d)
    # e = d_r ** 2 + d_i ** 2
    # f = e.sum()
    # f.backward()
    a = torch.randn((4, 5))
    b = a[:, None]
    c = a[:, :, None]
    d = torch.bmm(c, b) / torch.bmm(b, c)
    print(torch.trace(d))

