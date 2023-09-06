import numpy as np
import torch
from cuquantum import cutensornet as cutn

from mpo_utils import (dm2mpo, mpo2dm, trace, clone, unfold_circuit,
                        apply_gate, measure_expectation, apply_noise)


class MPOCircuitSimulator:
    """
    Simulate Noisy Quantum Circuits with MPO and Kraus Operators.
    """

    def __init__(self, circuit, noise_type=None, dtype=None, device='cuda', options=None):
        self.device = device
        self.dtype = dtype
        if dtype is None:
            self.dtype = torch.complex64
        self.algorithm = {'qr_method': False, 
                'svd_method':{'partition': 'UV', 'abs_cutoff':1e-12}}  # , 'max_extent':256}}
        self.num_qubits = circuit.num_qubits
        self.gate_map = dict(zip(circuit.qubits, range(self.num_qubits)))
        self.circuit = self._parse_circuit(circuit)
        self.kraus_ops = self._get_kraus(noise_type)
        self.options = options
        self.mpo = None

    def _initialize(self, dm):
        """
        Initialize MPO using density matrix.
        """
        if isinstance(dm, np.ndarray):
            dm = torch.from_numpy(dm).to(dtype=self.dtype, device=self.device)
        elif torch.is_tensor(dm):
            dm = dm.to(dtype=self.dtype, device=self.device)
        return dm2mpo(dm, options=self.options)
    
    def _get_zero_state(self):
        """
        Generate the MPO with an initial state of 0.
        """
        state_tensor = torch.zeros(
            (2 ** self.num_qubits, 2 ** self.num_qubits),
            dtype=self.dtype,
            device=self.device
        )
        state_tensor[0, 0] = 1
        return dm2mpo(state_tensor, options=self.options)

    def _get_kraus(self, noise_type):
        if noise_type is None:
            return None
        
        elif noise_type == 'dephasing':
            return lambda p: [
                torch.tensor([[1., 0.], [0., np.sqrt(1 - p)]], dtype=self.dtype, device=self.device),
                torch.tensor([[0., 0.], [0., np.sqrt(p)]], dtype=self.dtype, device=self.device)
            ]

        else:
            raise NotImplementedError(f"Noise of type: '{noise_type}' not supported.")

    def _parse_circuit(self, circuit):
        return unfold_circuit(circuit, dtype=self.dtype, device=self.device)

    def run(
        self,
        observables,
        observe_qubits,
        noise_level=None,
        init_rho=None
    ):
        self.mpo = self._initialize(init_rho) if init_rho is not None else self._get_zero_state()
        noise_operators = None if (self.kraus_ops is None or noise_level is None) else self.kraus_ops(noise_level)
        for idx, (gate, qubits) in enumerate(self.circuit):
            # mapping from qubits to qubit indices
            qubits = [self.gate_map[q] for q in qubits]
            # apply the gate in-place
            apply_gate(self.mpo, gate, qubits, algorithm=self.algorithm, options=self.options)
            if noise_operators is not None and (idx - 2) % 45 == 0:
                apply_noise(self.mpo, noise_operators, (0, *qubits), algorithm=self.algorithm, options=self.options)
                
        return measure_expectation(self.mpo, observables, observe_qubits, options=self.options).item()


if __name__ == '__main__':
    from basic_circuits import swaptest
    from qiskit.quantum_info import DensityMatrix, Operator, Pauli, random_density_matrix, state_fidelity, random_statevector
    handle = cutn.create()
    device = 'cuda'
    options = {'handle': handle}
    dtype = torch.complex64

    pauli_z = torch.tensor([[1., 0.], [0., -1.]], dtype=dtype, device=device)

    num_qubits_per_state = 5
    circuit = swaptest(num_qubits_per_state)
    # state1 = random_density_matrix(2 ** num_qubits_per_state)
    # state2 = random_density_matrix(2 ** num_qubits_per_state)
    state1 = random_statevector(2 ** num_qubits_per_state)
    state2 = random_statevector(2 ** num_qubits_per_state)
    state1_t = torch.from_numpy(state1.data).to(dtype=dtype, device=device)
    state2_t = torch.from_numpy(state2.data).to(dtype=dtype, device=device)
    state1_t = torch.outer(state1_t, state1_t.conj())
    state2_t = torch.outer(state2_t, state2_t.conj())
    ref_state = torch.tensor([[1., 0.], [0., 0.]], dtype=dtype, device=device)
    init_rho = torch.kron(ref_state, torch.kron(state1_t, state2_t))
    print(torch.trace(state1_t), torch.trace(state2_t))

    backend = MPOCircuitSimulator(circuit, 'dephasing', dtype, device, options)
    result = backend.run([pauli_z], (0,), 0.005, init_rho=init_rho)  # 0.005 - 0.02

    print('Qiskit Fidelity:', state_fidelity(state1, state2))
    print('MPO Fidelity:', result)
    
    cutn.destroy(handle)