import numpy as np
from numpy import linalg as LA
from ncon import ncon
import a_mps
import pickle
import b_model
from c_tebd import *
from tqdm import tqdm

def doApplyMPO(psi, L, M1, M2, R):
    """ function for applying MPO to state """
#     print(psi.shape)

    return ncon([psi.reshape(L.shape[2], M1.shape[3], M2.shape[3], R.shape[2]), L, M1, M2, R],
                [[1, 3, 5, 7], [2, -1, 1], [2, 4, -2, 3], [4, 6, -3, 5], [6, -4, 7]]).reshape(
        L.shape[2] * M1.shape[3] * M2.shape[3] * R.shape[2])

def eigLanczos(psivec, linFunct, functArgs, maxit=2, krydim=4):
    """ Lanczos method for finding smallest algebraic eigenvector of linear \
    operator defined as a function"""

    if LA.norm(psivec) == 0:
        psivec = np.random.rand(len(psivec))

    psi = np.zeros([len(psivec), krydim + 1])
    A = np.zeros([krydim, krydim])
    dval = 0

    for ik in range(maxit):

        psi[:, 0] = psivec / max(LA.norm(psivec), 1e-16)
        for ip in range(1, krydim + 1):

            psi[:, ip] = linFunct(psi[:, ip - 1], *functArgs)

            for ig in range(ip):
                A[ip - 1, ig] = np.dot(psi[:, ip], psi[:, ig])
                A[ig, ip - 1] = np.conj(A[ip - 1, ig])

            for ig in range(ip):
                psi[:, ip] = psi[:, ip] - np.dot(psi[:, ig], psi[:, ip]) * psi[:, ig]
                psi[:, ip] = psi[:, ip] / max(LA.norm(psi[:, ip]), 1e-16)

        [dtemp, utemp] = LA.eigh(A)
        psivec = psi[:, range(0, krydim)] @ utemp[:, 0]

    psivec = psivec / LA.norm(psivec)
    dval = dtemp[0]

    return psivec, dval

def doDMRG_MPO(A, ML, M, MR, chi, numsweeps=10, dispon=2, updateon=True, maxit=2, krydim=4):
    """
    ------------------------
    by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 19/1/2019
    ------------------------
    Implementation of DMRG for a 1D chain with open boundaries, using the \
    two-site update strategy. Each update is accomplished using a custom \
    implementation of the Lanczos iteration to find (an approximation to) the \
    ground state of the superblock Hamiltonian. Input 'A' is containing the MPS \
    tensors whose length is equal to that of the 1D lattice. The Hamiltonian is \
    specified by an MPO with 'ML' and 'MR' the tensors at the left and right \
    boundaries, and 'M' the bulk MPO tensor. Automatically grow the MPS bond \
    dimension to maximum dimension 'chi'. Outputs 'A' and 'B' are arrays of the \
    MPS tensors in left and right orthogonal form respectively, while 'sWeight' \
    is an array of the Schmidt coefficients across different lattice positions. \
    'Ekeep' is a vector describing the energy at each update step.

    Optional arguments:
    `numsweeps::Integer=10`: number of DMRG sweeps
    `dispon::Integer=2`: print data never [0], after each sweep [1], each step [2]
    `updateon::Bool=true`: enable or disable tensor updates
    `maxit::Integer=2`: number of iterations of Lanczos method for each diagonalization
    `krydim::Integer=4`: maximum dimension of Krylov space in superblock diagonalization
    """

    ##### left-to-right 'warmup', put MPS in right orthogonal form
    chid = M.shape[2]  # local dimension
    Nsites = len(A)
    L = [0 for x in range(Nsites)]
    L[0] = ML
    R = [0 for x in range(Nsites)]
    R[Nsites - 1] = MR
    for p in range(Nsites - 1):
        chil = A[p].shape[0]
        chir = A[p].shape[2]
        utemp, stemp, vhtemp = LA.svd(A[p].reshape(chil * chid, chir), full_matrices=False)
        A[p] = utemp.reshape(chil, chid, chir)
        A[p + 1] = ncon([np.diag(stemp) @ vhtemp, A[p + 1]], [[-1, 1], [1, -2, -3]]) / LA.norm(stemp)
        L[p + 1] = ncon([L[p], M, A[p], np.conj(A[p])], [[2, 1, 4], [2, -1, 3, 5], [4, 5, -3], [1, 3, -2]])

    chil = A[Nsites - 1].shape[0]
    chir = A[Nsites - 1].shape[2]
    utemp, stemp, vhtemp = LA.svd(A[Nsites - 1].reshape(chil * chid, chir), full_matrices=False)
    A[Nsites - 1] = utemp.reshape(chil, chid, chir)
    sWeight = [0 for x in range(Nsites + 1)]
    sWeight[Nsites] = (np.diag(stemp) @ vhtemp) / LA.norm(stemp)

    Ekeep = np.array([])
    B = [0 for x in range(Nsites)]
    for k in range(1, numsweeps + 2):

        ##### final sweep is only for orthogonalization (disable updates)
        if k == numsweeps + 1:
            updateon = False
            dispon = 0

        ###### Optimization sweep: right-to-left
        for p in range(Nsites - 2, -1, -1):

            ##### two-site update
            chil = A[p].shape[0]
            chir = A[p + 1].shape[2]
            psiGround = ncon([A[p], A[p + 1], sWeight[p + 2]], [[-1, -2, 1], [1, -3, 2], [2, -4]]).reshape(
                chil * chid * chid * chir)
            if updateon:
                psiGround, Entemp = eigLanczos(psiGround, doApplyMPO, (L[p], M, M, R[p + 1]), maxit=maxit,
                                               krydim=krydim)
                Ekeep = np.append(Ekeep, Entemp)

            utemp, stemp, vhtemp = LA.svd(psiGround.reshape(chil * chid, chid * chir), full_matrices=False)
            chitemp = min(len(stemp), chi)
            A[p] = utemp[:, range(chitemp)].reshape(chil, chid, chitemp)
            sWeight[p + 1] = np.diag(stemp[range(chitemp)] / LA.norm(stemp[range(chitemp)]))
            B[p + 1] = vhtemp[range(chitemp), :].reshape(chitemp, chid, chir)

            ##### new block Hamiltonian
            R[p] = ncon([M, R[p + 1], B[p + 1], np.conj(B[p + 1])], [[-1, 2, 3, 5], [2, 1, 4], [-3, 5, 4], [-2, 3, 1]])

#             if dispon == 2:
#                 print('Sweep: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))

        ###### left boundary tensor
        chil = A[0].shape[0]
        chir = A[0].shape[2]
        Atemp = ncon([A[0], sWeight[1]], [[-1, -2, 1], [1, -3]]).reshape(chil, chid * chir)
        utemp, stemp, vhtemp = LA.svd(Atemp, full_matrices=False)
        B[0] = vhtemp.reshape(chil, chid, chir)
        sWeight[0] = utemp @ (np.diag(stemp) / LA.norm(stemp))

        ###### Optimization sweep: left-to-right
        for p in range(Nsites - 1):

            ##### two-site update
            chil = B[p].shape[0]
            chir = B[p + 1].shape[2]
            psiGround = ncon([sWeight[p], B[p], B[p + 1]], [[-1, 1], [1, -2, 2], [2, -3, -4]]).reshape(
                chil * chid * chid * chir)
            if updateon:
                psiGround, Entemp = eigLanczos(psiGround, doApplyMPO, (L[p], M, M, R[p + 1]), maxit=maxit,
                                               krydim=krydim)
                Ekeep = np.append(Ekeep, Entemp)

            utemp, stemp, vhtemp = LA.svd(psiGround.reshape(chil * chid, chid * chir), full_matrices=False)
            chitemp = min(len(stemp), chi)
            A[p] = utemp[:, range(chitemp)].reshape(chil, chid, chitemp)
            sWeight[p + 1] = np.diag(stemp[range(chitemp)] / LA.norm(stemp[range(chitemp)]))
            B[p + 1] = vhtemp[range(chitemp), :].reshape(chitemp, chid, chir)

            ##### new block Hamiltonian
            L[p + 1] = ncon([L[p], M, A[p], np.conj(A[p])], [[2, 1, 4], [2, -1, 3, 5], [4, 5, -3], [1, 3, -2]])

            ##### display energy
#             if dispon == 2:
#                 print('Sweep: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))

        ###### right boundary tensor
        chil = B[Nsites - 1].shape[0]
        chir = B[Nsites - 1].shape[2]
        Atemp = ncon([B[Nsites - 1], sWeight[Nsites - 1]], [[1, -2, -3], [-1, 1]]).reshape(chil * chid, chir)
        utemp, stemp, vhtemp = LA.svd(Atemp, full_matrices=False)
        A[Nsites - 1] = utemp.reshape(chil, chid, chir)
        sWeight[Nsites] = (stemp / LA.norm(stemp)) * vhtemp

#         if dispon == 1:
#             print('Sweep: %d of %d, Energy: %12.12d, Bond dim: %d' % (k, numsweeps, Ekeep[-1], chi))

    return Ekeep, A, sWeight, B


##### Example 1: Ising model #############
#######################################

##### Set bond dimensions and simulation options
Nsites = 50
chi = 25
num_state = 20
std = 1


OPTS_numsweeps = 5 # number of DMRG sweeps
OPTS_dispon = 2 # level of output display
OPTS_updateon = True # level of output display
OPTS_maxit = 2 # iterations of Lanczos method
OPTS_krydim = 4 # dimension of Krylov subspace

#### Define Hamiltonian MPO
chid = 2
sP = np.sqrt(2)*np.array([[0, 0],[1, 0]])
sM = np.sqrt(2)*np.array([[0, 1],[0, 0]])
sX = np.array([[0, 1], [1, 0]])
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0,-1]])
sI = np.array([[1, 0], [0, 1]])

g = 1

time = 5

# Target states
Js = np.linspace(-2,2,20)
for iJ in tqdm(range(0,len(Js))):
    J = Js[iJ]
    if J == 0: continue

    M = np.zeros([3,3,chid,chid])
    M[0,0,:,:] = sI; M[2,2,:,:] = sI
    M[0,1,:,:] = -sZ; M[1,2,:,:] = sZ
    M[0,2,:,:] = -g/J*sX
    ML = np.array([1,0,0]).reshape(3,1,1) #left MPO boundary
    MR = np.array([0,0,1]).reshape(3,1,1) #right MPO boundary

    #### Initialize MPS tensors
    A = [0 for x in range(Nsites)]
    A[0] = np.random.rand(1,chid,min(chi,chid))
    for k in range(1,Nsites):
        A[k] = np.random.rand(A[k-1].shape[2],chid,min(min(chi,A[k-1].shape[2]*chid),chid**(Nsites-k-1)))

    En2, A, sWeight, B = doDMRG_MPO(A,ML,M,MR,chi, numsweeps = OPTS_numsweeps, dispon = OPTS_dispon,
                                    updateon = OPTS_updateon, maxit = OPTS_maxit, krydim = OPTS_krydim)

    psi = a_mps.init_FM_MPS(Nsites, chid, 'finite')
    for i in range(0, Nsites):
        psi.Bs[i] = B[i]
        psi.Ss[i] = np.diag(sWeight[i]) 

    model = b_model.TFIModel(L=Nsites, J=1., g=2, bc='finite')
    dt = 0.02
    U_bonds = calc_U_bonds(model.H_bonds, 1j * dt)
    run_TEBD(psi, U_bonds, N_steps=int(time / dt), chi_max=chi, eps=1.e-10)
    file_name = '../data_mitigate/mps/Ising/Ising_' + str(Nsites) + 'qubits_' + 't' + str(time) + '_output_mps_J_' + str(
        round(J, 1))
    with open(file_name, "wb") as fp:
        pickle.dump(psi, fp)


# Train states
NUM_TRAIN = 100
for iJ in tqdm(range(0, len(Js))):
    J = Js[iJ]
    if J == 0: continue
    for idx_train in range(NUM_TRAIN):
        M = np.zeros([3,3,chid,chid])
        M[0,0,:,:] = sI; M[2,2,:,:] = sI
        M[0,1,:,:] = -sZ; M[1,2,:,:] = sZ
        M[0,2,:,:] = -g/J*sX
        ML = np.array([1,0,0]).reshape(3,1,1) #left MPO boundary
        MR = np.array([0,0,1]).reshape(3,1,1) #right MPO boundary

        #### Initialize MPS tensors
        A = [0 for x in range(Nsites)]
        A[0] = np.random.rand(1,chid,min(chi,chid))
        for k in range(1,Nsites):
            A[k] = np.random.rand(A[k-1].shape[2],chid,min(min(chi,A[k-1].shape[2]*chid),chid**(Nsites-k-1)))

        En2, A, sWeight, B = doDMRG_MPO(A,ML,M,MR,chi, numsweeps = OPTS_numsweeps, dispon = OPTS_dispon,
                                        updateon = OPTS_updateon, maxit = OPTS_maxit, krydim = OPTS_krydim)

        psi = a_mps.init_FM_MPS(Nsites, chid, 'finite')
        for i in range(0, Nsites):
            psi.Bs[i] = B[i]
            psi.Ss[i] = np.diag(sWeight[i])

        file_name = '../data_mitigate/mps/Random/Ising_' + str(Nsites) + 'qubits_' + 't' + str(time) + f'_{idx_train}' + '_output_mps_J_' + str(
            round(J, 1))
        with open(file_name, "wb") as fp:
            pickle.dump(psi, fp)
