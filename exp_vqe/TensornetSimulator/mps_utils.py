import numpy as np
import torch
from cuquantum import contract, contract_path, CircuitToEinsum, tensor
from cuquantum import cutensornet as cutn
from cuquantum.cutensornet.experimental import contract_decompose

from basic_circuits import swaptest


class MPSContractionHelper:
    """
    A helper class to compute various quantities for a given MPS using exact contraction.
    
    Interleaved format is used to construct the input args for `cuquantum.contract`. 
    A concrete example on how the modes are populated for a 7-site MPS is provided below:
    
          0     2     4     6     8    10     12    14        
    bra -----A-----B-----C-----D-----E-----F-----G-----
             |     |     |     |     |     |     |     
            1|    3|    5|    7|    9|   11|   13|     
             |     |     |     |     |     |     |     
    ket -----a-----b-----c-----d-----e-----f-----g-----
          15    16    17    18    19    20    21    22
    
    
    The follwing compute quantities are supported:
    
        - the norm of the MPS.
        - the equivalent state vector from the MPS.
        - the expectation value for a given operator.
        - the equivalent state vector after multiplying an MPO to an MPS.
    
    Note that for the nth MPS tensor (rank-3), the modes of the tensor are expected to be `(i,p,j)` 
    where i denotes the bonding mode with the (n-1)th tensor, p denotes the physical mode for the qubit and 
    j denotes the bonding mode with the (n+1)th tensor.
    
    Args:
        num_qubits: The number of qubits for the MPS.
    """
    
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.path_cache = dict()
        self.bra_modes = [(2*i, 2*i+1, 2*i+2) for i in range(num_qubits)]
        offset = 2*num_qubits+1
        self.ket_modes = [(i+offset, 2*i+1, i+1+offset) for i in range(num_qubits)]
    
    def contract_norm(self, mps_tensors, options=None):
        """
        Contract the corresponding tensor network to form the norm of the MPS.

        Args:
            mps_tensors: A list of rank-3 ndarray-like tensor objects. 
                The indices of the ith tensor are expected to be bonding index to the i-1 tensor, 
                the physical mode, and then the bonding index to the i+1th tensor.
            options: Specify the contraction options. 

        Returns:
            The norm of the MPS.
        """
        interleaved_inputs = []
        for i, o in enumerate(mps_tensors):
            interleaved_inputs.extend([o, self.bra_modes[i], o.conj(), self.ket_modes[i]])
        interleaved_inputs.append([]) # output
        return self._contract('norm', interleaved_inputs, options=options).real

    def contract_state_vector(self, mps_tensors, options=None):
        """
        Contract the corresponding tensor network to form the state vector representation of the MPS.

        Args:
            mps_tensors: A list of rank-3 ndarray-like tensor objects. 
                The indices of the ith tensor are expected to be bonding index to the i-1 tensor, 
                the physical mode, and then the bonding index to the i+1th tensor.
            options: Specify the contraction options. 

        Returns:
            An ndarray-like object as the state vector.
        """
        interleaved_inputs = []
        for i, o in enumerate(mps_tensors):
            interleaved_inputs.extend([o, self.bra_modes[i]])
        output_modes = tuple([bra_modes[1] for bra_modes in self.bra_modes])
        interleaved_inputs.append(output_modes) # output
        return self._contract('sv', interleaved_inputs, options=options)
    
    def contract_expectation(self, mps_tensors, operator, qubits, normalize=False, options=None):
        """
        Contract the corresponding tensor network to form the state vector representation of the MPS.

        Args:
            mps_tensors: A list of rank-3 ndarray-like tensor objects. 
                The indices of the ith tensor are expected to be bonding index to the i-1 tensor, 
                the physical mode, and then the bonding index to the i+1th tensor.
            operator: A ndarray-like tensor object. 
                The modes of the operator are expected to be output qubits followed by input qubits, e.g, 
                ``A, B, a, b`` where `a, b` denotes the inputs and `A, B'` denotes the outputs. 
            qubits: A sequence of integers specifying the qubits that the operator is acting on. 
            normalize: Whether to scale the expectation value by the normalization factor.
            options: Specify the contraction options. 

        Returns:
            An ndarray-like object as the state vector.
        """
        
        interleaved_inputs = []
        extra_mode = 3 * self.num_qubits + 2
        operator_modes = [None] * len(qubits) + [self.bra_modes[q][1] for q in qubits]
        qubits = list(qubits)
        for i, o in enumerate(mps_tensors):
            interleaved_inputs.extend([o, self.bra_modes[i]])
            k_modes = self.ket_modes[i]
            if i in qubits:
                k_modes = (k_modes[0], extra_mode, k_modes[2])
                q = qubits.index(i)
                operator_modes[q] = extra_mode # output modes
                extra_mode += 1
            interleaved_inputs.extend([o.conj(), k_modes])
        interleaved_inputs.extend([operator, tuple(operator_modes)])
        interleaved_inputs.append([]) # output
        if normalize:
            norm = self.contract_norm(mps_tensors, options=options)
        else:
            norm = 1
        return self._contract(f'exp{qubits}', interleaved_inputs, options=options) / norm
    
    def contract_mps_mpo_to_state_vector(self, mps_tensors, mpo_tensors, options=None):
        """
        Contract the corresponding tensor network to form the output state vector from applying the MPO to the MPS.

        Args:
            mps_tensors: A list of rank-3 ndarray-like tensor objects. 
                The indices of the ith tensor are expected to be the bonding index to the i-1 tensor, 
                the physical mode, and then the bonding index to the i+1th tensor.
            mpo_tensors: A list of rank-4 ndarray-like tensor objects.
                The indics of the ith tensor are expected to be the bonding index to the i-1 tensor, 
                the output physical mode, the bonding index to the i+1th tensor and then the inputput physical mode.
            options: Specify the contraction options. 

        Returns:
            An ndarray-like object as the output state vector.
        """
        interleaved_inputs = []
        for i, o in enumerate(mps_tensors):
            interleaved_inputs.extend([o, self.bra_modes[i]])
        output_modes = []
        offset = 2 * self.num_qubits + 1
        for i, o in enumerate(mpo_tensors):
            mpo_modes = (2*i+offset, 2*i+offset+1, 2*i+offset+2, 2*i+1)
            output_modes.append(2*i+offset+1)
            interleaved_inputs.extend([o, mpo_modes])
        interleaved_inputs.append(output_modes)
        return self._contract('mps_mpo', interleaved_inputs, options=options)
    
    def _contract(self, key, interleaved_inputs, options=None):
        """
        Perform the contraction task given interleaved inputs. Path will be cached.
        """
        if key not in self.path_cache:
            self.path_cache[key] = contract_path(*interleaved_inputs, options=options)[0]
        path = self.path_cache[key]
        return contract(*interleaved_inputs, options=options, optimize={'path':path})


def state_vector_to_mps(statevec, options=None):
    """
    Generate MPS given a state vector.
    """
    # init_state = statevec.reshape(2, 2, 2, 2, 2)
    num_qubits = int(np.log2(statevec.shape[0]))
    statevec = statevec.reshape(1, *((2,) * num_qubits), 1)
    us = []
    for idx in range(num_qubits - 1):
        u, _, statevec = tensor.decompose(
            "tij...->tia,aj...",
            statevec,
            method=tensor.SVDMethod(partition='V', abs_cutoff=1e-12),
            options=options
        )
        print(statevec.shape)
        us.append(u)
    us.append(statevec)
    # # verifyv
    # contract_info = []
    # for i, item in enumerate(us):
    #     contract_info.extend([item, [2 * i, 2 * i + 1, 2 * i + 2]])
    # contract_info.append([2 * i + 1 for i in range(len(us))])
    # mps_result = contract(*contract_info)
    # print(mps_result.shape)
    # print(torch.allclose(mps_result, init_state, atol=1e-6))
    return us


def get_initial_mps(num_qubits, dtype='complex128'):
    """
    Generate the MPS with an initial state of |00...00> 
    """
    state_tensor = torch.tensor([1, 0], dtype=dtype).reshape(1,2,1)
    mps_tensors = [state_tensor] * num_qubits
    return mps_tensors


def operator_to_mpo(operator, options=None):
    num_qubits = int(np.log2(operator.shape[0]))
    return operator.reshape((2, 2) * num_qubits)


def mps_site_right_swap(
    mps_tensors, 
    i, 
    algorithm=None, 
    options=None
):
    """
    Perform the swap operation between the ith and i+1th MPS tensors.
    """
    # contraction followed by QR decomposition
    a, _, b = contract_decompose('ipj,jqk->iqj,jpk', *mps_tensors[i:i+2], algorithm=algorithm, options=options)
    mps_tensors[i:i+2] = (a, b)
    return mps_tensors


def apply_gate(
    mps_tensors, 
    gate, 
    qubits, 
    algorithm=None, 
    options=None
):
    """
    Apply the gate operand to the MPS tensors in-place.
    
    Args:
        mps_tensors: A list of rank-3 ndarray-like tensor objects. 
            The indices of the ith tensor are expected to be the bonding index to the i-1 tensor, 
            the physical mode, and then the bonding index to the i+1th tensor.
        gate: A ndarray-like tensor object representing the gate operand. 
            The modes of the gate is expected to be output qubits followed by input qubits, e.g, 
            ``A, B, a, b`` where ``a, b`` denotes the inputs and ``A, B`` denotes the outputs. 
        qubits: A sequence of integers denoting the qubits that the gate is applied onto.
        algorithm: The contract and decompose algorithm to use for gate application. 
            Can be either a `dict` or a `ContractDecomposeAlgorithm`.
        options: Specify the contract and decompose options. 
    
    Returns:
        The updated MPS tensors.
    """
    
    n_qubits = len(qubits)
    if n_qubits == 1:
        # single-qubit gate
        i = qubits[0]
        mps_tensors[i] = contract('ipj,qp->iqj', mps_tensors[i], gate, options=options) # in-place update
    elif n_qubits == 2:
        # two-qubit gate
        i, j = qubits
        if i > j:
            # swap qubits order
            return apply_gate(mps_tensors, gate.permute(1,0,3,2), (j, i), algorithm=algorithm, options=options)
        elif i+1 == j:
            # two adjacent qubits
            a, _, b = contract_decompose('ipj,jqk,rspq->irj,jsk', *mps_tensors[i:i+2], gate, algorithm=algorithm, options=options)
            mps_tensors[i:i+2] = (a, b) # in-place update
        else:
            # non-adjacent two-qubit gate
            # step 1: swap i with i+1
            mps_site_right_swap(mps_tensors, i, algorithm=algorithm, options=options)
            # step 2: apply gate to (i+1, j) pair. This amounts to a recursive swap until the two qubits are adjacent
            apply_gate(mps_tensors, gate, (i+1, j), algorithm=algorithm, options=options)
            # step 3: swap back i and i+1
            mps_site_right_swap(mps_tensors, i, algorithm=algorithm, options=options)
    else:
        raise NotImplementedError("Only one- and two-qubit gates supported")
    return mps_tensors


if __name__ == '__main__':
    from qiskit.quantum_info import Statevector, Operator, Pauli, random_statevector, state_fidelity
    from qiskit.circuit.random import random_circuit
    from qiskit import QuantumCircuit
    torch.set_default_device('cuda:0')
    handle = cutn.create()
    options = {'handle': handle}
    dtype = torch.complex128

    pauli_x = torch.tensor([[0., 1.], [1., 0.]], dtype=dtype)
    pauli_y = torch.tensor([[0., -1j], [1j, 0.]], dtype=dtype)
    pauli_z = torch.tensor([[1., 0.], [0., -1.]], dtype=dtype)
    pauli_i = torch.eye(2, dtype=dtype)
    ket0 = torch.tensor([1., 0.], dtype=dtype)

    op_c = pauli_z
    op_c_mpo = operator_to_mpo(op_c, options=options)

    num_qubits_per_state = 6
    # target_state1 = torch.from_numpy(random_statevector(2 ** (num_qubits_per_state)).data).cuda()
    # target_state2 = torch.from_numpy(random_statevector(2 ** (num_qubits_per_state)).data).cuda()
    # init_state = torch.kron(ket0, torch.kron(target_state1, target_state2))
    # init_state = torch.zeros(2 ** (2 * num_qubits_per_state + 1), dtype=dtype)
    # init_state[0] = 1.0
    # mps_tensors = state_vector_to_mps(init_state, options=options)

    mps_tensors = get_initial_mps(2 * num_qubits_per_state + 1, dtype=dtype)

    rand_circuit1 = random_circuit(num_qubits_per_state, 4, max_operands=2)
    rand_circuit2 = rand_circuit1.copy()# random_circuit(num_qubits_per_state, 4, max_operands=2)

    # rand_circuit1 = QuantumCircuit(num_qubits_per_state)
    # rand_circuit1.h(0)
    # for i in range(num_qubits_per_state - 1):
    #     rand_circuit1.cx(0, i + 1)
    # rand_circuit2 = QuantumCircuit(num_qubits_per_state)
    circuit = swaptest(num_qubits_per_state, [rand_circuit1, rand_circuit2])

    # We leverage ``cuquantum.CircuitToEinsum`` to obtain the gate operands.
    myconverter = CircuitToEinsum(circuit, dtype=dtype, backend=torch)
    gates = myconverter.gates
    gate_map = dict(zip(myconverter.qubits, range(num_qubits_per_state * 2 + 1)))

    # We construct an exact MPS with algorithm below. 
    # For two-qubit gates, an SVD is performed with singular values partitioned onto the two MPS sites equally.
    # We also set a cutoff value of 1e-12 to filter out computational noise.
    exact_gate_algorithm = {'qr_method': False, 
                            'svd_method':{'partition': 'UV', 'abs_cutoff':1e-12}}
    # Constructing the final MPS
    for (gate, qubits) in gates:
        # mapping from qubits to qubit indices
        qubits = [gate_map[q] for q in qubits]
        # apply the gate in-place
        apply_gate(mps_tensors, gate, qubits, algorithm=exact_gate_algorithm, options=options)
    print("Final MPS is constructed with the following shapes")
    for i, o in enumerate(mps_tensors):
        print(f"site {i}, shape: {o.shape}")

    
    mps_helper = MPSContractionHelper(2 * num_qubits_per_state + 1)
    # compute the norm of the MPS.
    norm = mps_helper.contract_norm(mps_tensors, options=options)
    print(f"The norm of the MPS: {norm:0.3e}")

    exp_mps = mps_helper.contract_expectation(mps_tensors, op_c_mpo, (0,), options=options, normalize=False)
    print(exp_mps.item())

    state1 = Statevector(rand_circuit1)
    state2 = Statevector(rand_circuit2)
    print(state_fidelity(state1, state2))

    # # comparision with state vector
    # # state vector constructed from the MPS
    # sv_mps = mps_helper.contract_state_vector(mps_tensors, options=options)
    # # reference from CircuitToEinsum

    # # sv_reference = Statevector(swaptest(num_qubits_per_state, init_state.cpu().numpy())).expectation_value(Operator(op_c.cpu().numpy()), [0])
    # # print(sv_reference)
    # # sv_reference = torch.from_numpy(sv_reference.data).cuda()
    # # sv_mps = sv_mps.view(sv_reference.shape)

    # subscripts, operands = myconverter.state_vector()
    # sv_reference = contract(subscripts, *operands, options=options)
    # print(f"State vector difference: {abs(sv_mps-sv_reference).max():0.3e}")
    # assert torch.allclose(sv_mps, sv_reference, atol=1e-2)
    pauli_string = 'Z' + 'I' * 2 * num_qubits_per_state
    expression, operands = myconverter.expectation(pauli_string, lightcone=True)
    expec = contract(expression, *operands).real
    print(f'expectation value for {pauli_string}: {expec}')
    
    cutn.destroy(handle)