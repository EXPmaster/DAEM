import pickle
import pathos
import numpy as np
import cirq
import cirq.ops as ops
from cirq.circuits import InsertStrategy
from utils import AverageMeter, ConfigDict
from basis import *


class QCircuitEnv:

    def __init__(self, args=None, **kwargs):
        self.qubit_gates = [
            [ops.rx, ops.ry, ops.rz, ops.X, ops.Y, ops.Z, ops.H, ops.S, ops.T],
            [ops.CNOT, ops.CZ]
        ]
        self.basis_ops = [
            ops.I, ops.X, ops.Y, ops.Z,
            GateRX(), GateRY(), GateRZ(), GateRYZ(),
            GateRZX(), GateRXY(), GatePiX(), GatePiY(),
            GatePiZ(), GatePiYZ(), GatePiZX(), GatePiXY()
        ]
        self.config = ...
        self.circuit = ...
        self.qubits = ...
        self.simulator = cirq.DensityMatrixSimulator()

        if 'pkl_file' in kwargs:
            pkl_file = kwargs['pkl_file']
            self.config = pkl_file['config']
            self.circuit = pkl_file['circuit']
            self.qubits = pkl_file['qubits']
        else:
            assert args is not None, 'Arguments must be provided.'
            self.config = ConfigDict(
                num_layers=args.num_layers,
                num_qubits=args.num_qubits
            )
            self._gen_new_circuit()

    def _gen_new_circuit(self):
        circuit = cirq.Circuit()
        input_qbits = cirq.NamedQubit.range(self.config.num_qubits, prefix='q')
        for layer in range(self.config.num_layers):
            gate_idx = np.random.choice([0, 0, 1])
            gate_set = self.qubit_gates[gate_idx]
            if gate_idx == 0:
                gates = []
                for qubit in input_qbits:
                    g = np.random.choice(gate_set)
                    if g in [ops.rx, ops.ry, ops.rz]:
                        g = g(np.random.uniform(-np.pi / 2, np.pi / 2))
                    gates.append(g(qubit))
                circuit.append(gates, strategy=InsertStrategy.NEW_THEN_INLINE)
            else:
                gate = np.random.choice(gate_set)
                select_qubit_idx = np.random.choice(self.config.num_qubits, size=2, replace=False)
                circuit.append([gate(*[input_qbits[i] for i in select_qubit_idx])], strategy=InsertStrategy.NEW_THEN_INLINE)
            circuit.append([ops.I(input_qbits[i]) for i in range(self.config.num_qubits)], strategy=InsertStrategy.NEW_THEN_INLINE)
        self.circuit = circuit
        self.qubits = input_qbits

    def _modify_circuit(self, p):
        batch_replace = []
        for index, (i, op) in enumerate(self.circuit.findall_operations(lambda op: op.gate == ops.I)):
            # print(i // 2, index % self.config.num_qubits, op.qubits)
            cur_pr = p[i // 2, index % self.config.num_qubits, :].ravel()
            gate = np.random.choice(self.basis_ops, p=cur_pr)
            batch_replace.append((i, op, gate(*op.qubits)))

        new_circuit = self.circuit.unfreeze(copy=True)
        new_circuit.batch_replace(batch_replace)
        return new_circuit
            
    def step(self, obs_batch, p_batch, nums=10000):
        if len(p_batch.shape) == 4:
            return np.array([self._get_mean_val(obs, p, nums) for obs, p in zip(obs_batch, p_batch)])
        else:
            return self._get_mean_val(obs_batch, p_batch, nums)

    def _apply_step(self, obs, p):
        circuit = self._modify_circuit(p)
        noisy_circuit = circuit.with_noise(cirq.depolarize(p=0.01))
        collector = cirq.PauliSumCollector(noisy_circuit, obs, samples_per_term=1)
        collector.collect(sampler=self.simulator)
        return collector.estimated_energy()
        
    def _get_mean_val(self, obs_operator, p, nums):
        avg = AverageMeter()
        pool = pathos.multiprocessing.Pool(processes=16)
        data_queue = []
        observable = cirq.PauliString(obs_operator(self.qubits[0]))
        # for i in range(nums):
        #     avg.update(self._apply_step(observable, p))
        for i in range(nums):
            data_queue.append(pool.apply_async(self._apply_step, args=(observable, p)))
        pool.close()
        pool.join()
        sum_val = sum([item.get() for item in data_queue])
        return sum_val / nums

    def save(self, path):
        save_data = dict(config=self.config, circuit=self.circuit, qubits=self.qubits)
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f'Environment saved at {path}.')

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print('Environment loaded.')
        return cls(pkl_file=data)


def stable_softmax(x):
    maxval = x.max(-1, keepdims=True)
    x_exp = np.exp(x - maxval)
    return x_exp / x_exp.sum(-1, keepdims=True)


def main(args):
    env = QCircuitEnv.load(args.env_path)
    rand_val = np.random.randn(args.data_num, args.num_layers, args.num_qubits, 16)
    pr = stable_softmax(rand_val)
    observable = [cirq.Z] * args.data_num
    
    results = env.step(observable, pr, nums=10000)
    obs = [cirq.unitary(item) for item in observable]
    save_data = dict(
        probability=pr,
        observable=obs,
        meas_result=results
    )
    with open('../data_surrogate/env2_data.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    print('Done.')


if __name__ == '__main__':
    from collections import namedtuple
    
    ArgsClass = namedtuple('args', ['num_layers', 'num_qubits', 'env_path', 'data_num'])
    args = ArgsClass(8, 5, '../environments/env1.pkl', 1000)
    print(args)
    # env = QCircuitEnv(args)
    # env.save('../environments/env1.pkl')
    main(args)
