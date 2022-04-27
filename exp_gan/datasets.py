import argparse
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import pathos
import cirq

from my_envs import QCircuitEnv


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


class MitigateDataset(Dataset):

    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.dataset = pickle.load(f)

    def __getitem__(self, idx):
        obs, exp_noisy, exp_ideal = self.dataset[idx]
        return torch.tensor(obs, dtype=torch.cfloat), torch.FloatTensor([exp_noisy]), torch.FloatTensor([exp_ideal])

    def __len__(self):
        return len(self.dataset)

    
def gen_mitigation_data(args):
    env = QCircuitEnv.load(args.env_path)
    output_state = cirq.Simulator().simulate(env.circuit).final_state_vector.reshape(-1, 1)
    noisy_circuit = env.circuit.with_noise(cirq.depolarize(p=0.01))
    rho = cirq.DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    num_qubits = len(env.qubits)

    dataset = []
    if args.num_data < 1000:
        dataset = gen_fn(num_qubits, output_state, rho, args.num_data)
    else:
        pool = pathos.multiprocessing.Pool(processes=8)
        data_queue = []
        for i in range(args.num_data // 1000):
            data_queue.append(pool.apply_async(gen_fn, args=(num_qubits, output_state, rho, 1000)))
        pool.close()
        pool.join()
        for item in data_queue:
            dataset += item.get()

    with open(args.out_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'Generation finished. File saved to {args.out_path}')


def gen_fn(num_qubits, output_state, rho, num_samples):
    data_list = []
    for i in range(num_samples):
        rand_matrix = torch.randn((2, 2), dtype=torch.cfloat).numpy()
        rand_obs = rand_matrix.conj().T @ rand_matrix
        obs_all = np.kron(rand_obs, np.eye((num_qubits - 1) ** 2))
        exp_ideal = output_state.conj().T @ obs_all @ output_state
        exp_ideal = round(exp_ideal.real[0][0], 8)
        exp_noisy = round(np.trace(obs_all @ rho).real, 8)
        data_list.append([rand_obs, exp_noisy, exp_ideal])

    return data_list


def gen_mitigation_data_ibmq(args):
    ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-path', default='../environments/env1.pkl', type=str)
    parser.add_argument('--out-path', default='../data_mitigate/env1_mitigate.pkl', type=str)
    parser.add_argument('--num-data', default=100_000, type=int)
    args = parser.parse_args()
    # dataset = SurrogateDataset('../data_surrogate/env1_data.pkl')
    # print(next(iter(dataset)))
    gen_mitigation_data(args)
