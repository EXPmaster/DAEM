import os
import time
import pickle
import argparse

import numpy as np
import torch
from torch import autograd
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from qiskit.quantum_info.operators import Pauli
from qiskit.quantum_info import DensityMatrix, pauli_basis
from qiskit.opflow import PauliOp
from my_envs import IBMQEnv

from model import AEModel
from utils import AverageMeter, build_dataloader, abs_deviation
from zne_trainer import ZNETrainer


class AEDataset:

    def __init__(self, data_path, batch_size, shuffle=False):
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        self.dataset = np.array(dataset)
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(dataset)
        self.batch_size = batch_size
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.dataset):
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.dataset)
            raise StopIteration

        start = self.idx
        self.idx += self.batch_size
        # return (
        #     torch.tensor(self.dataset[start: self.idx:, 0], dtype=torch.cfloat),
        #     torch.tensor(self.dataset[start: self.idx:, 1], dtype=torch.cfloat)
        # )
        return (
            torch.FloatTensor(self.dataset[start: self.idx:, 0]),
            torch.FloatTensor(self.dataset[start: self.idx:, 1])
        )


def get_hamiltonian_ising(num_qubits, g=0.7):
    operators = []
    op_str = 'I' * num_qubits
    for i in range(num_qubits - 1):
        tmp_op = op_str[:i] + 'ZZ' + op_str[i + 2:]
        operators.append(PauliOp(Pauli(tmp_op), -1.0))
    for i in range(num_qubits):
        tmp_op = op_str[:i] + 'X' + op_str[i + 1:]
        operators.append(PauliOp(Pauli(tmp_op), -g))
    hamitonian = sum(operators)
    return hamitonian


def transform_to_pauli_basis(matrix):
    """ Vectorize input matrix in Pauli basis.
    Args:
        Matrix: DensityMatrix or Obervable
    Returns:
        np.ndarray: Vectorized matrix, in real numbers.
    """
    ret_list = []
    if isinstance(matrix, DensityMatrix):
        data = matrix.data
    else:
        data = matrix.to_matrix()
    dim = int(np.sqrt(data.shape[0]))
    for basis in pauli_basis(dim, pauli_list=True):
        pauli = Pauli(basis).to_matrix()
        ret_list.append(np.trace(pauli @ data))
    if isinstance(matrix, DensityMatrix):
        return np.array(ret_list).real
    else:
        return np.array(ret_list).real / data.shape[0]


def gen_mitigate_data(env_path):
    env = IBMQEnv.load(env_path)
    circuit = env.circuit
    dataset = []
    for noise_scale in np.round(np.arange(0.05, 0.19, 0.01), 3): # 10
        noisy_state = env.simulate_noisy(noise_scale)
        noisy_vec = transform_to_pauli_basis(noisy_state)
        noisy_state2 = env.simulate_noisy(round(noise_scale + 0.005, 5))
        noisy_vec2 = transform_to_pauli_basis(noisy_state2)
        dataset.append([noisy_vec, noisy_vec2])

    with open('../data_mitigate/ae_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)


def vertual_purification(noisy_state):
    w, v = np.linalg.eigh(noisy_state.data)
    # print(np.allclose(noisy_state.data, v @ np.diag(w) @ v.conj().T))
    k = 12
    # w[:k] = 0
    w = w ** 5
    # v[:, :k] = 0
    data = v @ np.diag(w) @ v.conj().T
    purified_state = data / np.trace(data)
    return purified_state


def main(args, H):
    loss_fn = nn.MSELoss()
    model = AEModel(indim=4 ** 4)
    model.to(args.device)
    # optimizer_g = optim.Adam(model_g.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    print('Start training...')
    H_vec = transform_to_pauli_basis(H)  # H.to_matrix()

    best_metric = args.thres
    for epoch in range(args.epochs):
        print(f'=> Epoch {epoch}')
        train(epoch, args, dataset1, model, loss_fn, optimizer, H_vec)
        # metric = validate(epoch, args, dataset1, model, H_vec)
        # scheduler.step()
        # if metric < best_metric:
        #     # print('Saving model...')
        #     best_metric = metric
   
    # with open(os.path.join(args.logdir, 'metric_gan.txt'), 'a+') as f:
    #     f.write('{} {:.6f}\n'.format(len(trainset), best_metric))
    # print(best_metric)
    model.eval()
    input_state = transform_to_pauli_basis(noisy_state)
    # input_state = torch.tensor(input_state, dtype=torch.cfloat).to(args.device)[None]
    # H = torch.tensor(H_vec, dtype=torch.cfloat).to(args.device)  # [:, None]
    input_state = torch.FloatTensor(input_state).to(args.device)[None]
    H = torch.FloatTensor(H_vec).to(args.device)
    # print(torch.trace(input_state.squeeze() @ H))
    print(input_state.squeeze() @ H)
    predict_energy0 = float('inf') + 0j
    predict_energy = 0.
    n = 0
    while predict_energy.real < predict_energy0.real and n < 10:
        input_state = model(input_state)
        predict_energy0 = predict_energy
        # predict_energy = (torch.trace(input_state.squeeze() @ H)).item()
        predict_energy = (input_state.squeeze() @ H).item()
        print(predict_energy)
        n += 1
    # energy = input_state @ H
    energy = H @ input_state.squeeze()
    print(energy)
    ckpt = {
        'model': model.state_dict(),
    }
    torch.save(ckpt, os.path.join('../runs/AE_env_vqe_ampdamp', f'{energy.item():.4f}.pt'))


def train(epoch, args, loader1, model, loss_fn, optimizer, Hamiltonian):
    model.train()

    for itr, (data_output, data_input) in enumerate(loader1):
        data_input = data_input.to(args.device)
        data_output = data_output.to(args.device)

        optimizer.zero_grad()
        predict = model(data_input)
        loss_real = loss_fn(predict, data_output)
        # loss_real = (predict @ data_output).diagonal(-2, -1).sum(-1).mean()
        loss_real.backward()

        optimizer.step()

        if itr == 0:
            # args.writer.add_scalar('Loss/train', loss_accumulator.getval(), epoch)
            # print('Loss_D: {:.6f}\tLoss_G\t{:.6f}'.format(loss_real.item(), loss_gradient.item()))
            print('Loss_D: {:.6f}\tLoss_G\t{:.6f}'.format(loss_real.item(), 0.))


@torch.no_grad()
def validate(epoch, args, loader, model, Hamiltonian):
    print('Validating...')
    model.eval()
    metric = AverageMeter()
    H = torch.FloatTensor(Hamiltonian).to(args.device)[:, None]
    for data_input in loader:
        data_input = data_input.to(args.device)

        predict = model(data_input)
        predict_value = predict @ H

        metric.update(predict_value.mean())

    value = metric.getval()
    # args.writer.add_scalar('Loss/val', losses.getval(), epoch)
    # args.writer.add_scalar('Abs_deviation/val', value, epoch)
    print('validation energy: {:.6f}'.format(value))
    return value



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MitigateDataset', type=str, help='[MitigateDataset]')
    parser.add_argument('--train-path', default='../data_mitigate/amp_damp/dataset_vqe4l.pkl', type=str)
    parser.add_argument('--test-path', default='../data_mitigate/amp_damp/dataset_vqe4l.pkl', type=str)
    parser.add_argument('--env-path', default='../environments/amp_damping/vqe_envs_train_4l', type=str)
    parser.add_argument('--logdir', default='../runs', type=str, help='path to save logs and models')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--num-ops', default=2, type=int)
    parser.add_argument('--epochs', default=230, type=int)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--gamma', default=4.0, type=float, help='constraint for discriminator')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', default=0.99, type=float, help='beta2 for Adam optimizer')
    parser.add_argument('--thres', default=0.12, type=float, help='threshold to saving model')
    parser.add_argument('--save-name', default='gan_model.pt', type=str, help='model file name')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(args.device)
    
    env_path = '../environments/amp_damping/vqe_envs_train_4l/vqe_0.6.pkl'
    dataset1 = AEDataset('../data_mitigate/ae_dataset.pkl', args.batch_size, shuffle=True)
    # dataset2 = AEDataset('../data_mitigate/ae_dataset.pkl', args.batch_size, shuffle=True)
    env = IBMQEnv.load(env_path)
    ideal_state = env.simulate_noisy(0.0)
    noisy_state = env.simulate_noisy(0.05)
    H = get_hamiltonian_ising(4, g=0.6)
    ideal_exp = ideal_state.expectation_value(H).real
    noisy_exp = noisy_state.expectation_value(H).real
    print(ideal_exp, noisy_exp)

    # rho_vec = transform_to_pauli_basis(noisy_state)
    # H_vec = transform_to_pauli_basis(H)

    # print(np.inner(rho_vec, H_vec))
    # gen_mitigate_data(env_path)
    # assert False
    main(args, H)

    print(ideal_exp)
    zne_model = ZNETrainer()
    print(zne_model.fit_and_predict(env.circuit, H)[-1])

    