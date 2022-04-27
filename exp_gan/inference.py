import math
import numpy as np
from qiskit import IBMQ, Aer, QuantumCircuit, transpile, execute
from qiskit.providers.aer import AerSimulator
from qiskit.opflow import I, X, Y, Z
from qiskit.providers.aer.noise import NoiseModel


class QuantumExecuter:

	def __init__(self, backend='ibmq_santiago'):
		IBMQ.load_account()
		provider = IBMQ.get_provider(
			hub='ibm-q',
			group='open',
			project='main'
		)
		self.backend = provider.get_backend(backend)
		self.circuit = self._build_easy_circuit()
		# self.circuit = self._build_swap_test_circuit()
		# backends = provider.backends(filters = lambda x:x.configuration().n_qubits >= 5 and not x.configuration().simulator
        #                      and x.status().operational==True)			 
	
	@staticmethod
	def _build_easy_circuit():
		circ = QuantumCircuit(2, 1)
		circ.h(0)
		circ.h(1)
		circ.cx(0, 1)
		# circ.i(1)
		circ.rz(math.pi/3, 1)
		circ.cx(0, 1)
		circ.h(0)
		circ.barrier()
		circ.measure(0, 0)
		# print(circ)
		# for item in circ:
		# 	print(item[0].name)
		return circ

	@staticmethod
	def _build_swap_test_circuit():
		circ = QuantumCircuit(7, 1)
		circ.h(0)
		circ.h(1)
		circ.cx(1, 2)
		circ.cx(1, 3)
		circ.cswap(0, 1, 4)
		circ.cswap(0, 2, 5)
		circ.cswap(0, 3, 6)
		circ.h(0)
		circ.measure(0, 0)
		return circ

	def modify_circuit(self):
		for node in self.circuit.op_nodes():
			print(node.op.name)

	@staticmethod
	def measure_z(counts, shots):
		probs = {}
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
		probs = {}
		for output in ['0', '1']:
			if output in counts:
				probs[output] = counts[output] / shots
			else:
				probs[output] = 0
		state_vec = np.array([[np.sqrt(probs['0'])], [np.sqrt(probs['1'])]])
		return (state_vec.T @ obs @ state_vec)[0][0]
	
	def run_device(self, shots=10000):
		t_circuit = transpile(self.circuit, self.backend)
		job = self.backend.run(t_circuit, shots=shots)
		results = job.result()
		counts = results.get_counts()
		return self.measure_z(counts, shots)
	
	def run_noisy(self, shots=10000):
		# noise_model = NoiseModel.from_backend(self.backend)
		# coupling_map = self.backend.configuration().coupling_map
		# basis_gates = noise_model.basis_gates
		# result = execute(self.circuit, Aer.get_backend('qasm_simulator'),
		# 				 coupling_map=coupling_map,
		# 				 basis_gates=basis_gates,
		# 				 noise_model=noise_model,
		# 				 shots=shots).result()
		backend = AerSimulator.from_backend(self.backend)
		t_circuit = transpile(self.circuit, backend)
		result_noisy = backend.run(t_circuit, shots=shots).result()
		counts = result_noisy.get_counts()
		print(self.circuit)
		obs = np.array([[1,0],[0,-1]])
		print(self.measure_obs(counts, obs, shots))
		return self.measure_z(counts, shots)
	
	def run_ideal(self, shots=10000):
		backend = Aer.get_backend('aer_simulator')
		job = backend.run(transpile(self.circuit, backend), shots=shots)
		counts = job.result().get_counts()
		return self.measure_z(counts, shots)


if __name__ == '__main__':
	executer = QuantumExecuter()
	shots = 20000
	# print('ideal: ', executer.run_ideal(shots))
	# print('noisy: ', executer.run_noisy(shots))
	# print('device:', executer.run_device(shots))
	executer.modify_circuit()

