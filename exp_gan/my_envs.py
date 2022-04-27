import pickle
import pathos
import numpy as np
import cirq
import cirq.ops as ops
from cirq.circuits import InsertStrategy

from qiskit import IBMQ, Aer, QuantumCircuit, transpile, execute
from qiskit.providers.aer import AerSimulator
import qiskit.opflow as opflow
from qiskit.providers.aer.noise import NoiseModel
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import TransformationPass

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


class IBMQEnv:

    def __init__(self, args=None, **kwargs):
        if 'pkl_file' in kwargs:
            pkl_file = kwargs['pkl_file']
            self.config = pkl_file['config']
            self.circuit = pkl_file['circuit']
        else:
            assert args is not None, 'Arguments must be provided.'
            self.config = ConfigDict(
                num_layers=args.num_layers,
                num_qubits=args.num_qubits,
                backend=args.backend
            )
            self.circuit = self._gen_new_circuit()
        IBMQ.load_account()
        provider = IBMQ.get_provider(
            hub='ibm-q',
            group='open',
            project='main'
        )
        self.backend = provider.get_backend(self.config.backend)
        # self.backend = AerSimulator.from_backend(backend)
        self.transformer = TransformCirc(self.config.num_qubits)
        print(self.circuit)

    @staticmethod
    def _gen_new_circuit():
        circ = QuantumCircuit(2, 1)
        circ.h(0)
        circ.i(0)
        circ.h(1)
        circ.i(1)
        circ.cx(0, 1)
        circ.i(0)
        circ.i(1)
        circ.rz(np.pi/3, 1)
        circ.i(1)
        circ.cx(0, 1)
        circ.i(0)
        circ.i(1)
        circ.h(0)
        circ.i(0)
        circ.barrier()
        circ.measure(0, 0)
        return circ

    def _modify_circuit(self, p):
        return self.transformer(self.circuit, p)

    def step(self, obs_batch, p_batch, nums=10000):
        if len(p_batch.shape) == 4:
            return np.array([self._get_mean_val(obs, p, nums) for obs, p in zip(obs_batch, p_batch)])
        else:
            return self._get_mean_val(obs_batch, p_batch, nums)
    
    @staticmethod
    def measure_z(counts, shots):
        probs = {'0': 0, '1': 0}
        for output in ['0', '1']:
            if output in counts:
                probs[output] = counts[output] / shots
            else:
                probs[output] = 0
        # measure in Z basis
        measure = probs['0'] - probs['1']
        return measure

    @staticmethod
    def measure_obs(counts, obs, shots):
        """measure a specific observable"""
        probs = {'0': 0, '1': 0}
        for output in ['0', '1']:
            if output in counts:
                probs[output] = counts[output] / shots
            else:
                probs[output] = 0
        state_vec = np.array([[np.sqrt(probs['0'])], [np.sqrt(probs['1'])]])
        return (state_vec.T @ obs @ state_vec)[0][0]

    def _apply_step(self, obs, p):
        circuit = self._modify_circuit(p)
        backend = AerSimulator.from_backend(self.backend)
        t_circuit = transpile(circuit, backend)
        result_noisy = backend.run(t_circuit, shots=1).result()
        counts = result_noisy.get_counts()
        return self.measure_obs(counts, obs, shots=1)
        
    def _get_mean_val(self, observable, p, nums):
        avg = AverageMeter()
        pool = pathos.multiprocessing.Pool(processes=16)
        data_queue = []
        for _ in range(nums):
            data_queue.append(pool.apply_async(self._apply_step, args=(observable, p)))
        pool.close()
        pool.join()
        sum_val = sum([item.get() for item in data_queue])
        return sum_val / nums

    def save(self, path):
        save_data = dict(config=self.config, circuit=self.circuit)
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f'Environment saved at {path}.')

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print('Environment loaded.')
        return cls(pkl_file=data)


class TransformCirc(TransformationPass):

    def __init__(self, num_qubits):
        super().__init__()
        self.basis_ops = [
            ops.I, ops.X, ops.Y, ops.Z,
            # GateRX(), GateRY(), GateRZ(), GateRYZ(),
            # GateRZX(), GateRXY(), GatePiX(), GatePiY(),
            # GatePiZ(), GatePiYZ(), GatePiZX(), GatePiXY()
        ]
        self.num_qubits = num_qubits

    def run(self, dag, p):
        """Run the pass."""
        idx = 0
        # iterate over all operations
        for node in dag.op_nodes():
            # if we hit a RYY or RZZ gate replace it
            if node.op.name == 'id':
                cur_pr = p[idx // self.num_qubits, idx % self.num_qubits, :].ravel()
                gate = np.random.choice(self.basis_ops, p=cur_pr)
                # calculate the replacement
                replacement = QuantumCircuit(1)
                replacement.unitary(cirq.unitary(gate), 0, label=str(gate))

                # replace the node with our new decomposition
                dag.substitute_node_with_dag(node, circuit_to_dag(replacement))
                idx += 1

        return dag
    
    def __call__(self, circuit, p, property_set=None):
        from qiskit.converters import circuit_to_dag, dag_to_circuit
        from qiskit.dagcircuit.dagcircuit import DAGCircuit

        result = self.run(circuit_to_dag(circuit), p)

        result_circuit = circuit

        if isinstance(property_set, dict):  # this includes (dict, PropertySet)
            property_set.clear()
            property_set.update(self.property_set)

        if isinstance(result, DAGCircuit):
            result_circuit = dag_to_circuit(result)
        elif result is None:
            result_circuit = circuit.copy()

        if self.property_set["layout"]:
            result_circuit._layout = self.property_set["layout"]
        if self.property_set["clbit_write_latency"] is not None:
            result_circuit._clbit_write_latency = self.property_set["clbit_write_latency"]
        if self.property_set["conditional_latency"] is not None:
            result_circuit._conditional_latency = self.property_set["conditional_latency"]

        return result_circuit


def stable_softmax(x):
    maxval = x.max(-1, keepdims=True)
    x_exp = np.exp(x - maxval)
    return x_exp / x_exp.sum(-1, keepdims=True)


def main(args):
    env = IBMQEnv.load(args.env_path)
    print('Start.')
    rand_val = np.random.randn(args.data_num, args.num_layers, args.num_qubits, 4)
    pr = stable_softmax(rand_val)
    observable = [cirq.unitary(cirq.Z)] * args.data_num
    
    results = env.step(observable, pr, nums=1000)
    print(results)
    # save_data = dict(
    #     probability=pr,
    #     observable=observable,
    #     meas_result=results
    # )
    # with open('../data_surrogate/env2_data.pkl', 'wb') as f:
    #     pickle.dump(save_data, f)

    print('Done.')


if __name__ == '__main__':
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    from collections import namedtuple
    import warnings
    warnings.filterwarnings('ignore')
    
    ArgsClass = namedtuple('args', ['num_layers', 'num_qubits', 'backend', 'env_path', 'data_num'])
    args = ArgsClass(8, 5, 'ibmq_santiago', '../environments/ibmq1.pkl', 1)
    # print(args)
    # env = IBMQEnv(args)
    # env.save('../environments/ibmq1.pkl')
    main(args)
    # print(env.circuit)
    