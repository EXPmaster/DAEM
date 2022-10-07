from qiskit import *

default_clifford_list = ['I','X','Y','Z','S','XS','YS','ZS','H','XH','YH','ZH','SH','XSH','YSH','ZSH','HS','XHS','YHS','ZHS','SHS','XSHS','YSHS','ZSHS']
pauli_list = ['I','X','Y','Z']

def create_clifford(str):
    clifford_circuit = QuantumCircuit(1)
    for c in str:
        if c == 'I':
            clifford_circuit.i(0)
        elif c == 'X':
            clifford_circuit.x(0)
        elif c == 'Y':
            clifford_circuit.y(0)
        elif c == 'Z':
            clifford_circuit.z(0)
        elif c == 'H':
            clifford_circuit.h(0)
        elif c == 'S':
            clifford_circuit.s(0)
    return clifford_circuit

def insert_pauli(qc, pauli):
    new_circuit = QuantumCircuit(qc.width())
 
    for args in qc.data:
        if args[0].is_parameterized():
            qubit_idx = args[1][0].index
            new_circuit.pauli(pauli[0], [qubit_idx])
            new_circuit.data.append(args)
            pauli = pauli[1:]
        else:
            new_circuit.data.append(args)
 
    return new_circuit

def get_pauli_comb(n):
    pauli_comb_list = pauli_list.copy()
 
    for _ in range(n-1):
        pauli_comb_list = [pauli1 + pauli2 for pauli1 in pauli_comb_list for pauli2 in pauli_list]

    return pauli_comb_list    

def get_clifford_combo(n):
    clifford_combo_list = default_clifford_list.copy()
    
    for _ in range(n-1):
        clifford_combo_list = [[clifford1, clifford2] for clifford1 in clifford_combo_list for clifford2 in default_clifford_list]
    
    return clifford_combo_list

# e.g.) circuit_list = get_circuits_dict(qc, [['X','Y','X'],['X','Y','Z'],['I','X','X']], ['XXX','YYY'], 3)
def get_circuits_dict(qc, clifford_list=[], pauli_comb_list=[], num_parameterized_gates=-1):
    circuits_list = []
    
    if num_parameterized_gates == -1:
        num_parameterized_gates = qc.num_parameters

    if not clifford_list:
        clifford_list = get_clifford_combo(num_parameterized_gates)

    if not pauli_comb_list:
        pauli_comb_list = get_pauli_comb(num_parameterized_gates)

    for args in clifford_list:
        ef_em_dict = {}
        ef_em_dict['efc'] = QuantumCircuit(qc.width())
 
        em_dict = {}
        for pauli in pauli_comb_list:
            em_dict[pauli] = QuantumCircuit(qc.width())
        ef_em_dict['emc'] = em_dict
 
        circuits_list.append(ef_em_dict)
 
    pauli_idx = 0
    clifford_idx = 0
    for args in qc.data:
        if args[0].is_parameterized():
            qubit_idx = args[1][0].index
            for idx_c, clifford_comb in enumerate(clifford_list):
                for pauli_comb in pauli_comb_list:
                    circuits_list[idx_c]['emc'][pauli_comb].pauli(pauli_comb[pauli_idx], [qubit_idx])
 
                clifford_to_add = create_clifford(clifford_comb[clifford_idx])
                circuits_list[idx_c]['efc'] = circuits_list[idx_c]['efc'].compose(clifford_to_add, [qubit_idx])
                for pauli_comb in pauli_comb_list:
                    circuits_list[idx_c]['emc'][pauli_comb] = circuits_list[idx_c]['emc'][pauli_comb].compose(clifford_to_add, [qubit_idx])
 
            pauli_idx = pauli_idx + 1
            clifford_idx = clifford_idx + 1
        else:
            for idx_c, gate_c in enumerate(clifford_list):
                circuits_list[idx_c]['efc'].data.append(args)
                for pauli_comb in pauli_comb_list:
                    circuits_list[idx_c]['emc'][pauli_comb].data.append(args)

    return circuits_list
