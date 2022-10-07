from math import prod
import qiskit
import numpy as np
from generate_training_set import insert_pauli
from qiskit import QuantumCircuit
from random import sample, seed
from itertools import product

def get_measuring_circuit(basis_list: list) -> QuantumCircuit:
    qc = QuantumCircuit(len(basis_list[0][1]))

    # Find a pauli word that has the most single qubit rotations
    basis = ''
    for qi in range(len(basis_list[0][1])):
        nonI = False
        for term in basis_list:
            if term[1][qi] != 'I':
                nonI = True
                basis += term[1][qi]
                break
        if not nonI:
            basis += 'I'
    
    # Construct the appropriate circuit
    for i,term in enumerate(basis):
        if term == 'X':
            qc.h(i)
        elif term == 'Y':
            qc.rx(np.pi/2,i)
        else:
            continue
    return qc

def pauli_expval(pauliwords: list, circuit: QuantumCircuit, qinstance: qiskit.utils.QuantumInstance, included_measuring_circuit = True) -> list:

    # If the basis rotation is not included, add appropriate rotations. All pauliwords should be commuting.
    if not included_measuring_circuit:
        templist = []
        for pauliword in pauliwords:
            templist.append((1, pauliword))
        measuring_circuit = get_measuring_circuit(templist)
        circuit = circuit.compose(measuring_circuit)
        if not qinstance.is_statevector:
            circuit.measure_all()

    # Run the ciruit!
    res = qinstance.execute(circuit)
    if qinstance.is_statevector:
        sv = res.get_statevector()
        pseudoprobs = sv.probabilities_dict()
    else:
        pseudoprobs = res.get_counts()

    # Calculate the expectation value of each observable
    expvals = []
    for pauliword in pauliwords:
        if not qinstance.is_statevector:
            total_counts = 0
        expval = 0
        for basis in pseudoprobs:
            if not qinstance.is_statevector:
                total_counts += pseudoprobs[basis]
            eigenvalue = 1
            for qubit, pauli in zip(basis[::-1], pauliword):
                if pauli != 'I' and qubit == '0':
                    eigenvalue = eigenvalue * 1
                if pauli != 'I' and qubit == '1':
                    eigenvalue = eigenvalue * (-1)
            expval += eigenvalue * pseudoprobs[basis]
        if not qinstance.is_statevector:
            expval = expval / total_counts
        expvals.append(expval)
    return expvals

def expval_from_counts(pauliword, counts):
    expval = 0
    total_counts = 0
    for basis in counts:
        eigenvalue = 1
        total_counts += counts[basis]
        for qubit, pauli in zip(basis[::-1], pauliword):
            if pauli != 'I' and qubit == '0':
                eigenvalue = eigenvalue * 1
            if pauli != 'I' and qubit == '1':
                eigenvalue = eigenvalue * (-1)
        expval += eigenvalue * counts[basis]
    expval = expval / total_counts
    return expval

def expval_calc(hamiltonian: list, circuits_to_run, em_instance: qiskit.utils.QuantumInstance, ef_instance: qiskit.utils.QuantumInstance):

    # Dictionaries to store com_ef and com_em results
    com_ef = {}
    com_em = {}
    
    # List of all error mitigation circuits (circuits to run on noisy simulator/hardware)
    tot_em_list = []
    batchsize = 64

    # Calculate all com values
    for coi, commuting_operators in enumerate(hamiltonian):

        # Get measurement part of the circuit. This is the same for all commuting operators
        measurement_circuit = get_measuring_circuit(commuting_operators)

        for efcnum, efc_emcs in enumerate(circuits_to_run):
            efc = efc_emcs['efc']
            emcs = efc_emcs['emc']

            # Calculate all error free values
            # For each training circuit, add the measurement circuit and execute to find the statevector
            circuit_to_run = efc.compose(measurement_circuit)

            # Calculate the expectation value of each commuting operator
            pauliwords = []
            for coeff, pauliword in commuting_operators:
                pauliwords.append(pauliword)
            expvals = pauli_expval(pauliwords, circuit_to_run, ef_instance)

            # Save the result in com_ef
            for pauliword, expval in zip(pauliwords, expvals):
                com_ef[(pauliword, efcnum)] = expval

            # For each error mitigation circuit, add the measurement circuit
            for p in emcs:
                emc = emcs[p]
                circuit_to_run = emc.compose(measurement_circuit)
                circuit_to_run.measure_all()
                descriptor = (coi, efcnum, p)
                tot_em_list.append((descriptor, circuit_to_run))
    
    # Divide the total error mitigated circuits into batches and run them.
    batched_list = [tot_em_list[i:i + batchsize] for i in range(0, len(tot_em_list), batchsize)]
    tot_em_counts = []
    for batch in batched_list:
        batch_only_circuits = [i[1] for i in batch]
        res = em_instance.execute(batch_only_circuits)
        counts = res.get_counts()
        for i in range(len(batch)):
            tot_em_counts.append((batch[i][0], counts[i]))
    
    # Calculate the noisy results and save to com_em
    for descriptor, noisy_result in tot_em_counts:
        coi, efcnum, p = descriptor
        commuting_operators = hamiltonian[coi]
        for coef, pauliword in commuting_operators:
            com_em[(pauliword, efcnum, p)] = expval_from_counts(pauliword, noisy_result)

    return com_ef, com_em

def q_optimize(hamiltonian: list, circuits_to_run, com_em: dict, com_ef: dict):

    # Calculate 'a' matrix and 'b' vector
    # Extend the definitions to include a constant q_0 term. a is 5 x 5 because of 4 paulis + q0 term
    extendedP = []
    for p in circuits_to_run[0]['emc']:
        extendedP.append(p)
    extendedP.append('q0')

    pauliwords = []
    for commuting_operators in hamiltonian:
        for coeff, pauliword in commuting_operators:
            pauliwords.append(pauliword)

    N = len(pauliwords)
    T = len(circuits_to_run)

    for pauliword in pauliwords:
        for r in range(T):
            com_em[(pauliword, r, 'q0')] = 1

    a = np.zeros((len(extendedP), len(extendedP)))
    b = np.zeros((len(extendedP)))

    for p1i, p1 in enumerate(extendedP):
        for p2i, p2 in enumerate(extendedP):
            for n in pauliwords:
                for r in range(T):
                    a[p1i][p2i] += com_em[(n, r, p1)]*com_em[(n, r, p2)]
            a[p1i][p2i] = a[p1i][p2i]/(N * T)

    for pi, p in enumerate(extendedP):
        for r in range(T):
            for pauliword in pauliwords:
                b[pi] += com_em[(pauliword, r, p)] * com_ef[(pauliword, r)]
        b[pi] = b[pi] / (N * T)

    # Optimize and find q vector
    q = np.dot(np.linalg.inv(a), b)
    return q, extendedP

def test(ansatz, angles, hamiltonian, q, ef_instance, em_instance):
    boundansatz = ansatz.bind_parameters(angles)

    ef_expval = 0
    em_expval = 0
    n_expval = 0

    noisy_hardware_circuits = []

    for coi, commuting_operators in enumerate(hamiltonian):
        for coef, pauliword in commuting_operators:
            ef_expval += coef * pauli_expval([pauliword], boundansatz, ef_instance, included_measuring_circuit = False)[0]
        measurement_circuit = get_measuring_circuit(commuting_operators)
        noisy = boundansatz.compose(measurement_circuit)
        noisy.measure_all()
        noisy_hardware_circuits.append((('noisy', coi), noisy))
        for p in q[1]:
            if p != 'q0':
                pauli_inserted = insert_pauli(ansatz, p)
                pauli_inserted = pauli_inserted.bind_parameters(angles)
                pauli_inserted = pauli_inserted.compose(measurement_circuit) # 
                pauli_inserted.measure_all()
                noisy_hardware_circuits.append(((p, coi), pauli_inserted))
    
    circuits_only = []
    for descriptor, qc in noisy_hardware_circuits:
        circuits_only.append(qc)
    res = em_instance.execute(circuits_only)
    counts_list = res.get_counts()

    for (descriptor, qc), counts in zip(noisy_hardware_circuits, counts_list):
        description, coi = descriptor
        commuting_operators = hamiltonian[coi]
        if description == 'noisy':
            for coef, pauliword in commuting_operators:
                n_expval += coef * expval_from_counts(pauliword, counts)
        else:
            for coef, pauliword in commuting_operators:
                em_expval += coef * q[0][q[1].index(description)] * expval_from_counts(pauliword, counts)

    em_expval += q[0][-1]
    return ef_expval, em_expval, n_expval

def em_expval_calc(ansatz, angles, hamiltonian, q, em_instance):
    em_expval = 0

    noisy_hardware_circuits = []

    for coi, commuting_operators in enumerate(hamiltonian):
        measurement_circuit = get_measuring_circuit(commuting_operators)
        for p in q[1]:
            if p != 'q0':
                pauli_inserted = insert_pauli(ansatz, p)
                pauli_inserted = pauli_inserted.bind_parameters(angles)
                pauli_inserted = pauli_inserted.compose(measurement_circuit)
                pauli_inserted.measure_all()
                noisy_hardware_circuits.append(((p, coi), pauli_inserted))
    
    circuits_only = []
    for descriptor, qc in noisy_hardware_circuits:
        circuits_only.append(qc)
    res = em_instance.execute(circuits_only)
    counts_list = res.get_counts()

    for (descriptor, qc), counts in zip(noisy_hardware_circuits, counts_list):
        description, coi = descriptor
        commuting_operators = hamiltonian[coi]
        for coef, pauliword in commuting_operators:
            em_expval += coef * q[0][q[1].index(description)] * expval_from_counts(pauliword, counts)

    em_expval += q[0][-1]
    return em_expval

def n_expval_calc(ansatz, angles, hamiltonian, em_instance):
    boundansatz = ansatz.bind_parameters(angles)

    n_expval = 0

    noisy_hardware_circuits = []

    for coi, commuting_operators in enumerate(hamiltonian):
        measurement_circuit = get_measuring_circuit(commuting_operators)
        noisy = boundansatz.compose(measurement_circuit)
        noisy.measure_all()
        noisy_hardware_circuits.append((('noisy', coi), noisy))
    
    circuits_only = []
    for descriptor, qc in noisy_hardware_circuits:
        circuits_only.append(qc)
    res = em_instance.execute(circuits_only)
    counts_list = res.get_counts()

    for (descriptor, qc), counts in zip(noisy_hardware_circuits, counts_list):
        description, coi = descriptor
        commuting_operators = hamiltonian[coi]
        for coef, pauliword in commuting_operators:
            n_expval += coef * expval_from_counts(pauliword, counts)

    return n_expval

def ef_expval_calc(ansatz, angles, hamiltonian, ef_instance):
    boundansatz = ansatz.bind_parameters(angles)

    ef_expval = 0

    for coi, commuting_operators in enumerate(hamiltonian):
        for coef, pauliword in commuting_operators:
            ef_expval += coef * pauli_expval([pauliword], boundansatz, ef_instance, included_measuring_circuit = False)[0]

    return ef_expval

def truncate_training_set(num_param_gates, num_trunc_P, num_trunc_T, s = 10, exhaustive = False):
    seed(s)
    paulis = ['I', 'X', 'Y', 'Z']
    cliffords = ['I','X','Y','Z','S','XS','YS','ZS','H','XH','YH','ZH','SH','XSH','YSH','ZSH','HS','XHS','YHS','ZHS','SHS','XSHS','YSHS','ZSHS']

    if not exhaustive:
        trunc_T = []
        while len(trunc_T) < num_trunc_T:
            temp = []
            for j in range(num_param_gates):
                temp.append(sample(cliffords, 1)[0])
            if not temp in trunc_T:
                trunc_T.append(temp)

        trunc_P = []
        while len(trunc_P) < num_trunc_P:
            temp = []
            for j in range(num_param_gates):
                temp.append(sample(paulis, 1)[0])
            temp = ''.join(temp)
            if not temp in trunc_P:
                trunc_P.append(temp)
    
    else:
        trunc_T = list(product(cliffords, repeat = num_param_gates))
        trunc_T = [list(i) for i in trunc_T]

        trunc_P = list(product(paulis, repeat = num_param_gates))
        trunc_P = [''.join(list(i)) for i in trunc_P]

    return trunc_T, trunc_P
