
from pickle import LIST, TUPLE
from typing import List
import warnings
warnings.filterwarnings("ignore")
import math
import numpy as np
import matplotlib.pyplot as plt

from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.aqua.operators import Z2Symmetries
from qiskit.algorithms.optimizers import SLSQP,POWELL,SPSA,COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.aqua.algorithms import VQE, NumPyEigensolver
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit import IBMQ, BasicAer, Aer
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumCircuit, Parameter

from qiskit.circuit import ParameterVector, QuantumCircuit
from expval_calc_q_optim import *
from qiskit.utils import QuantumInstance 
from qiskit.algorithms import optimizers



def calculate_pauli_hamiltonian(molecule_name, distance ,map_type = 'parity'):
  
    if molecule_name =='H2':
        freeze_list = []
        remove_list = []
        molecular_coordinates = "H 0 0 0; H 0 0 " + str(distance)
    elif molecule_name == 'LiH':
        freeze_list = [0,6]
        remove_list = [4,8]
        molecular_coordinates = "Li 0 0 0; H 0 0 " + str(distance)
    else:
        raise NotImplementedError

    driver = PySCFDriver(molecular_coordinates, unit=UnitsType.ANGSTROM,charge=0,spin=0,basis='sto3g')
    molecule = driver.run()
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)

    ##  number of oribials and particles
    num_spin_orbitals = molecule.num_orbitals * 2
    num_particles = molecule.num_alpha + molecule.num_beta
    nuclear_repulsion_energy = molecule.nuclear_repulsion_energy

    ## freeze the core orbitals
    if(freeze_list == []):
        ferOp_f = ferOp
        energy_shift = 0
    else:
        ferOp_f, energy_shift = ferOp.fermion_mode_freezing(freeze_list)

    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)

    ## remove the determined orbitals
    if(remove_list == []):
        ferOp_fr = ferOp_f
    else:
        ferOp_fr = ferOp_f.fermion_mode_elimination(remove_list)
    
    num_spin_orbitals -= len(remove_list)

    ### get  the qubit hamiltonian
    qubitOp = ferOp_fr.mapping(map_type=map_type)

    ### reduce number of qubits through z2 symmetries
    if map_type == 'parity':
        qubitOp_t = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
    else:
        qubitOp_t = qubitOp
    
    ## process the qubit operator ############
    qubit_ham = []
    for coeff,pauli in qubitOp_t.__dict__['_paulis']:
        if all([p == "I" for p in pauli.to_label()]):
            identity_coeff = coeff
            continue
        qubit_ham.append((coeff,pauli.to_label()))

    result = {'qubit_operator': qubit_ham,'coeff_identity_pauli': identity_coeff , 'shift':nuclear_repulsion_energy + energy_shift,'num_particles': num_particles,'num_spin_orbitals':num_spin_orbitals,'hf_energy': molecule.hf_energy,'qub_op_with_I':qubitOp_t }

    return result


def check_simplification(op1, op2):

    for i in range(len(op1)):
        if ( op1[i]!= op2[i] ) and  "I"  not in [ op1[i], op2[i]] :
            return False
    return True



def join_operators(op1, op2):

    assert(check_simplification(op1,op2))
    joined_op = []
    for i in range(len(op1)):
        if op1[i] == op2[i]:
            joined_op.append(op1[i])
        else:
            joined_op.append(op1[i] if op1[i] != "I" else op2[i])

    return joined_op




def optimize_measurements(obs_hamiltonian):

    final_solution = []
    for op1 in obs_hamiltonian:
        added = False
        for i, op2 in enumerate(final_solution):

            if check_simplification(op1, op2):
                final_solution[i] = join_operators(op1, op2)
                added = True
                break
        if not added:
            final_solution.append(op1)
        
    return final_solution

def grouping(final_solution, obs_hamiltonian ):
    
    grouped_list = []
    for i,meas_op in enumerate(final_solution):
        temp = []
        for coeff,operator in obs_hamiltonian:
            if(check_simplification(list(operator),meas_op)):
                temp.append((coeff,operator))
        grouped_list.append(temp)
    return grouped_list

  
def get_ansatz(n,m,ansatz = 'simple'):
    ## trivial circuit for H2 check might not work properly
    phi = Parameter('phi')
    qc = QuantumCircuit(n)
    if ansatz == 'simple':
        qc.h(0)
        qc.cx(0,1)
        qc.rx(phi,0)
        qc.cx(1,0)
        num_par_gate = 1
    elif ansatz == 'num_particle_preserving':
        qc, num_par_gate = n_qubit_A_circuit(n,m)
    else:
        raise NotImplementedError
    return qc, num_par_gate


def A_gate(qc, qubit1 , qubit2, theta):
    qc.cx(qubit2,qubit1)
    qc.ry(theta + math.pi/2,qubit2)
    qc.rz(math.pi,qubit2)
    qc.cx(qubit1,qubit2)
    qc.ry(theta + math.pi/2,qubit2)
    qc.rz(math.pi,qubit2)
    qc.cx(qubit2,qubit1)

def n_qubit_A_circuit(n,m, repeat = 1):
    qc = QuantumCircuit(n)
    index = 0
    theta  = ParameterVector('theta', repeat*(n-m)*m)
    ## primitive pattern
    for _ in range(repeat): 
        for i in range(m):
            qc.x(i)
            for j in range(m,n):
                A_gate(qc,i,j,theta[index])
                index += 1
    
    return qc, index*2
        
def main(molecule_name: str ,distance: float ,n: int , m: int , ansatz: str) :
    
    """
    n(int): number of orbitals
    m(int): number of particles
    distance(int): interatomic distance
    ansatz(str): value should be 'simple' or 'num_particle_preserving'
    """
    
    pauli_ham_dict = calculate_pauli_hamiltonian(molecule_name, distance)
    qubit_ham = pauli_ham_dict['qubit_operator']
    optimized = optimize_measurements( [list(term[1]) for term in qubit_ham] )
    group_pauli_op = grouping(optimized,qubit_ham)
    ansatz, num_par_gates = get_ansatz(n,m,ansatz)
    pauli_ham_dict['grouped_paulis'] =group_pauli_op


    return pauli_ham_dict, [ansatz,num_par_gates]

#def em_expval()






def run_VQE(molecule_name,distance,n,m, q, em_instance, ef_instance, ansatz, optimizer):


    qubit_ham, [ansatz,num_par_gates] = main(molecule_name,distance,n,m,ansatz)

    def cost_function_em(angle):
        expval =  em_expval_calc(ansatz,angle,qubit_ham['grouped_paulis'],q, em_instance)
        return expval
    def cost_function_ns(angle):
        expval = n_expval_calc(ansatz,angle,qubit_ham['grouped_paulis'], em_instance)
        return expval
    def cost_function_ef(angle):
        expval = ef_expval_calc(ansatz,angle,qubit_ham['grouped_paulis'], ef_instance)
        return expval
    
    optimized_em  = optimizer.optimize(num_vars = ansatz.num_parameters, objective_function = cost_function_em, initial_point = np.zeros(ansatz.num_parameters))
    optimized_ns = optimizer.optimize(num_vars = ansatz.num_parameters, objective_function = cost_function_ns, initial_point = np.zeros(ansatz.num_parameters))
    optimized_ef = optimizer.optimize(num_vars = ansatz.num_parameters, objective_function = cost_function_ef, initial_point = np.zeros(ansatz.num_parameters))

    noisy_vqe_energy = optimized_ns[1] + qubit_ham['coeff_identity_pauli'] + qubit_ham ['shift']
    vqe_energy = optimized_em[1] + qubit_ham['coeff_identity_pauli'] + qubit_ham ['shift']
    ef_vqe_energy = optimized_ef[1] + qubit_ham['coeff_identity_pauli'] + qubit_ham ['shift']

    exact_energy = NumPyEigensolver(qubit_ham['qub_op_with_I']).run().eigenvalues + qubit_ham ['shift'] 
    hf_energy = qubit_ham ['hf_energy']

    return {'vqe_em_energy': vqe_energy, 'noisy_vqe_energy': noisy_vqe_energy, 'ef_vqe_energy': ef_vqe_energy ,'exact_energy': exact_energy, 'hf_energy': hf_energy}

    
def plot_PES(molecule_name: str,
            distance_list: List[int],
            n:int , m: int, q , 
            em_instance : QuantumInstance,
            ef_instance: QuantumInstance, 
            ansatz : QuantumCircuit,
            optimizer: optimizers , 
            save_fig: bool = False):

    result = {'vqe_em_energy': [], 'noisy_vqe_energy': [], 'ef_vqe_energy': [], 'exact_energy': [], 'hf_energy': []}
    for distance in distance_list:
        res = run_VQE(molecule_name,distance,n,m, q, em_instance, ef_instance, ansatz, optimizer)
        result['exact_energy'].append(res['exact_energy'])
        result['vqe_em_energy'].append(res['vqe_em_energy'])
        result['noisy_vqe_energy'].append(res['noisy_vqe_energy'])
        result['ef_vqe_energy'].append(res['ef_vqe_energy'])
        result['hf_energy'].append(res['hf_energy'])

    print(result)

    plt.plot(distance_list, result['hf_energy'],'-o' , label="HF Energy")
    plt.plot(distance_list, result['exact_energy'], '-o', label="Exact Energy")
    plt.plot(distance_list, result['vqe_em_energy'], '-k', label="VQE_EM Energy")
    plt.plot(distance_list, result['noisy_vqe_energy'], '-r', label="VQE_noisy Energy")
    plt.plot(distance_list, result['ef_vqe_energy'], '-g', label="VQE_EF Energy")

    plt.title(molecule_name + "'s PES curve ")
    plt.xlabel('Atomic distance (Angstrom)')
    plt.ylabel('Energy(Hartree)')
    plt.legend()
    plt.show()

    if save_fig:
        plt.savefig("./{}PES curve with LBEM.png".format(molecule_name) )
    
    return result
