import functools
import numpy as np
import torch
from cuquantum import contract, contract_path, tensor
from cuquantum import cutensornet as cutn
from cuquantum.cutensornet.experimental import contract_decompose

from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, ControlledGate, Delay, Gate, Measure
from qiskit.extensions import UnitaryGate


def get_decomposed_gates(circuit, qubit_map=None, gates=None, gate_process_func=None):
    """
    Return the gate sequence for the given circuit. Compound gates/instructions will be decomposed 
    to either standard gates or customized unitary gates.
    """
    if gates is None:
        gates = []
    for operation, gate_qubits, _ in circuit:
        if qubit_map:
            gate_qubits = [qubit_map[q] for q in gate_qubits]
        if isinstance(operation, Gate):
            if 'standard_gate' in str(type(operation)) or isinstance(operation, UnitaryGate):
                if callable(gate_process_func):
                    gates.append(gate_process_func(operation, gate_qubits))
                else:
                    gates.append((operation, gate_qubits))
                continue
        else:
            if isinstance(operation, (Barrier, Delay)):
                # no physical meaning in tensor network simulation
                continue
            elif not isinstance(operation.definition, QuantumCircuit):
                # Instruction as composite gate
                raise ValueError(f'operation type {type(operation)} not supported')
        # for composite gate, must provide a map from the sub circuit to the original circuit
        next_qubit_map = dict(zip(operation.definition.qubits, gate_qubits))
        gates = get_decomposed_gates(operation.definition, qubit_map=next_qubit_map, gates=gates, gate_process_func=gate_process_func)
    return gates
    

def unfold_circuit(circuit, dtype='complex128', device=None):
    """
    Unfold the circuit to obtain the qubits and all gate tensors. All :class:`qiskit.circuit.Gate` and 
    :class:`qiskit.circuit.Instruction` in the circuit will be decomposed into either standard gates or customized unitary gates.
    Barrier and delay operations will be discarded.

    Args:
        circuit: A :class:`qiskit.QuantumCircuit` object. All parameters in the circuit must be binded.
        dtype: Data type for the tensor operands.
        device: CUDA or CPU

    Returns:
        All qubits and gate operations from the input circuit
    """
    
    def gate_process_func(operation, gate_qubits):
        tensor = operation.to_matrix().reshape((2,2)*len(gate_qubits))
        tensor = torch.tensor(tensor, dtype=dtype, device=device)
        # in qiskit notation, qubits are labelled in the inverse order
        return tensor, gate_qubits[::-1]
    
    gates = get_decomposed_gates(circuit, gate_process_func=gate_process_func)
    return gates


def dm2mpo(dm, options=None):
    """
    Transform a batch of density matrices to MPO.
    Suppose the original shape of DM is [batch_size, 2^n, 2^n].
    """
    num_qubits = int(np.log2(dm.shape[1]))
    operators = []
    for i in range(num_qubits - 1):
        dm = dm.reshape(-1, 2, 2 ** (num_qubits - 1 - i), 2, 2 ** (num_qubits - 1 - i), 1)
        o, _, dm = tensor.decompose(
            "mijkln->mika,ajln",
            dm,
            method=tensor.SVDMethod(partition='UV', abs_cutoff=1e-12),
            options=options
        )
        operators.append(o)
    operators.append(dm)
    return operators


def mpo2dm(mpo, options=None):
    contract_info = []
    for k, item in enumerate(mpo):
        contract_info.extend([item, [max(3 * k - 1, 0), 1 + 3 * k, 3 + 3 * k, 2 + 3 * k]])
    contract_info.append([1 + 3 * i for i in range(len(mpo))] + [3 + 3 * i for i in range(len(mpo))])
    dm = contract(*contract_info, options=options).reshape(2 ** len(mpo), -1)
    return dm


def trace(mpo, options=None):
    contract_info = []
    for k, item in enumerate(mpo):
        contract_info.extend([item, [max(3 * k - 1, 0), 1 + 3 * k, 1 + 3 * k, 2 + 3 * k]])
    contract_info.append([0, 2 + 3 * (len(mpo) - 1)])
    return contract(*contract_info, options=options)


def clone(mpo):
    return [item.clone() for item in mpo]


def mpo_add(mpo1, mpo2, options=None):
    """
    Add two MPOs (can restrict dimension by passing algorithm)
    # TODO: better algorithm than MPO -> DM -> MPO
    """
    dm1 = mpo2dm(mpo1, options=options)
    dm2 = mpo2dm(mpo2, options=options)
    dm_sum = dm1 + dm2
    return dm2mpo(dm_sum, options=options)


def mpo_site_right_swap(
    mpo, 
    i,
    options=None
):
    """
    Perform the swap operation between the ith and i+1th MPO tensors.
    """
    # contraction followed by QR decomposition
    a, b = contract_decompose('mikt,tjln->mjlt,tikn', *mpo[i:i+2], options=options)
    mpo[i:i+2] = (a, b)


def apply_gate(mpo, gate, qubits, algorithm=None, options=None):
    num_qubits = len(qubits)
    if num_qubits == 1:
        i = qubits[0]
        mpo[i] = contract('mikn,ai->makn', mpo[i], gate, options=options)
        mpo[i] = contract('mikn,kb->mibn', mpo[i], gate.conj(), options=options)
    elif num_qubits == 2:
        i, j = qubits
        if i > j:
            apply_gate(mpo, gate.permute(1, 0, 3, 2), (j, i), algorithm=algorithm, options=options)
        elif i + 1 == j:
            a, _, b = contract_decompose('mikt,tjln,abij,klcd->mact,tbdn', *mpo[i:i+2], gate, gate.conj(),
                                         algorithm=algorithm, options=options)
            mpo[i: i + 2] = (a, b)
        else:
            # swap i with i + 1 and apply gate to i + 1, then swap back
            mpo_site_right_swap(mpo, i, options=options)
            apply_gate(mpo, gate, (i + 1, j), algorithm=algorithm, options=options)
            mpo_site_right_swap(mpo, i, options=options)
    else:
        raise NotImplementedError("Only one- and two-qubit gates supported")


def measure_expectation(mpo, operators, qubits, normalize=False, options=None):
    tr_mpo = 1.
    mpo = clone(mpo)
    if normalize: tr_mpo = trace(mpo, options=options)
    for i, observable in zip(qubits, operators):
        mpo[i] = contract('mikn,ai->makn', mpo[i], observable, options=options)
    contract_info = []
    return trace(mpo, options=options) / tr_mpo


def expand_bond(mpo, i, ratio):
    """ Expand bond dimension of MPO[i] by ratio. """
    assert 0 <= i < len(mpo), f'MPO index {i} out of range.'
    # if i == 0:
    #     mpo[0] = mpo[0].repeat(1, 1, 1, ratio)
    #     mpo[1] = mpo[1].repeat(ratio, 1, 1, 1)
    # elif i == len(mpo) - 1:
    #     mpo[-1] = mpo[-1].repeat(ratio, 1, 1, 1)
    #     mpo[-2] = mpo[-2].repeat(1, 1, 1, ratio)
    # else:
    #     mpo[i - 1] = mpo[i - 1].repeat(1, 1, 1, ratio)
    #     mpo[i] = mpo[i].repeat(ratio, 1, 1, ratio)
    #     mpo[i + 1] = mpo[i + 1].repeat(ratio, 1, 1, 1)
    for i in range(len(mpo)):
        if i == 0:
            mpo[0] = mpo[0].repeat(1, 1, 1, ratio)
        elif i == len(mpo) - 1:
            mpo[-1] = mpo[-1].repeat(ratio, 1, 1, 1)
        else:
            mpo[i] = mpo[i].repeat(ratio, 1, 1, ratio)

    # mpo[0] /= trace(mpo)


def apply_noise(mpo, kraus, qubits, algorithm=None, options=None, method='add'):
    """
    Apply Kraus operator to MPO. Suppose each Kraus to single qubit.
    """
    mpo_proc = clone(mpo)
    if method == 'add':
        for i in qubits:
            out_mpos = []
            for K in kraus:
                temp_mpo = clone(mpo_proc)
                apply_gate(temp_mpo, K, (i,), algorithm=algorithm, options=options)
                out_mpos.append(temp_mpo)
            mpo_proc = functools.reduce(functools.partial(mpo_add, options=options), out_mpos)
    elif method == 'direct':
        for i in qubits:
            expand_bond(mpo_proc, i, len(kraus))
            print(mpo_proc[i].shape)
            step_size_left = int(np.ceil(mpo_proc[i].shape[0] / len(kraus)))
            step_size_right = int(np.ceil(mpo_proc[i].shape[3] / len(kraus)))
            for id_k, K in enumerate(kraus):
                index_left_l = min(id_k * step_size_left, mpo_proc[i].shape[0] - step_size_left)
                index_left_u = min((id_k + 1) * step_size_left, mpo_proc[i].shape[0])
                index_right_l = min(id_k * step_size_right, mpo_proc[i].shape[3] - step_size_right)
                index_right_u = min((id_k + 1) * step_size_right, mpo_proc[i].shape[3])
                print(index_left_l, index_left_u, index_right_l, index_right_u)
                mpo_proc[i][index_left_l: index_left_u, :, :, index_right_l: index_right_u] = \
                contract(
                    'mikn,ai,kb->mabn',
                    mpo_proc[i][index_left_l: index_left_u, :, :, index_right_l: index_right_u],
                    K,
                    K.conj()
                )
        mpo_proc[0] /= trace(mpo_proc)
            
    for i in range(len(mpo)):
        mpo[i] = mpo_proc[i]


if __name__ == '__main__':
    from qiskit.quantum_info import Statevector, DensityMatrix, Operator, Pauli, state_fidelity, random_density_matrix
    torch.set_default_device('cuda:0')
    dtype = torch.complex64
    handle = cutn.create()
    options = {'handle': handle}

    algorithm = {'qr_method': False, 
                'svd_method':{'partition': 'UV', 'abs_cutoff':1e-12}}

    pauli_x = torch.tensor([[0., 1.], [1., 0.]], dtype=dtype)
    pauli_y = torch.tensor([[0., -1j], [1j, 0.]], dtype=dtype)
    pauli_z = torch.tensor([[1., 0.], [0., -1.]], dtype=dtype)
    cnot = torch.zeros((4, 4), dtype=dtype)
    cnot[0, 0] = 1.0
    cnot[1, 1] = 1.0
    cnot[2, 3] = 1.0
    cnot[3, 2] = 1.0
    pauli_i = torch.eye(2, dtype=dtype)
    ket0 = torch.tensor([1., 0.], dtype=dtype)
    ket1 = torch.tensor([0., 1.], dtype=dtype)

    N = 6
    p = 0.3
    kraus_dephasing = [torch.tensor([[1., 0.], [0., np.sqrt(1 - p)]], dtype=dtype),
                       torch.tensor([[0., 0.], [0., np.sqrt(p)]], dtype=dtype)]
    rho = random_density_matrix(2 ** N)
    rho_torch = torch.from_numpy(rho.data).to(dtype=dtype, device='cuda').repeat(2, 1, 1)
    # rho_torch = torch.kron(torch.kron(ket0, ket0), ket1)
    # rho_torch = torch.outer(rho_torch, rho_torch)
    rho_mpo = dm2mpo(rho_torch, options=options)
    assert False
    # apply_gate(rho_mpo, pauli_x, (1,), algorithm=algorithm, options=options)
    apply_gate(rho_mpo, cnot.reshape(2, 2, 2, 2), [N - 1 - 3, N - 1 - 1], algorithm=algorithm, options=options)
    rho_1 = clone(rho_mpo)
    rho_2 = clone(rho_mpo)
    # apply_noise(rho_1, kraus_dephasing, (N - 1 - 3, N - 1 - 1), algorithm=algorithm, options=options, method='add')
    apply_noise(rho_1, kraus_dephasing, (0, 2, 4, 5), algorithm=algorithm, options=options, method='add')
    dm_1 = DensityMatrix(mpo2dm(rho_1, options=options).cpu().numpy())
    # apply_noise(rho_2, kraus_dephasing, (N - 1 - 3, N - 1 - 1), algorithm=algorithm, options=options, method='direct')
    apply_noise(rho_2, kraus_dephasing, (0, 2, 4, 5), algorithm=algorithm, options=options, method='direct')
    print(trace(rho_2))
    dm_2 = DensityMatrix(mpo2dm(rho_2, options=options).cpu().numpy())
    print(dm_1)
    print(dm_2)
    print(np.allclose(dm_1, dm_2, rtol=1e-3, atol=1e-3))
    
    # print(state_fidelity(dm_1, dm_2))


    assert False
    print(trace(rho_mpo, options=options))
    # rho_out = DensityMatrix(mpo2dm(rho_mpo, options=options).cpu().numpy())

    rho = rho.evolve(Operator(cnot.cpu().numpy()), [1, 3])
    ## NOTE
    # from Qiskit to here:
    # 1) switch two qubit gates' bit order
    # 2) for qibit idx applying gates, idx_here = N - 1 - idx_qiskit
    # print(state_fidelity(rho, rho_out))

    meas_qiskit = rho.expectation_value(Operator(pauli_x.cpu().numpy()), [1])
    meas_mpo = measure_expectation(rho_mpo, [pauli_x], [N - 1 - 1], normalize=False)
    print(f'qiskit: {meas_qiskit}, MPO: {meas_mpo.item()}')
    
    cutn.destroy(handle)