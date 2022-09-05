# Linear Algebra
import numpy as np
from tqdm import tqdm

# we'll need them a lot
sxx = np.array([[0., 1.], [1., 0.]])
syy = np.array([[0., -1j], [1j, 0.]])
szz = np.array([[1., 0.], [0., -1.]])

"""Provides exact ground state energies for the transverse field ising model for comparison.

The Hamiltonian reads
.. math ::
    H = - J \\sum_{i} \\sigma^x_i \\sigma^x_{i+1} - g \\sum_{i} \\sigma^z_i
"""
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg.eigen.arpack as arp
import warnings
import scipy.integrate

def exact_E(L, J, h):
    """For comparison: obtain ground state energy from exact diagonalization.
    Exponentially expensive in L, only works for small enough `L` <~ 20.
    """
    if L >= 20:
        warnings.warn("Large L: Exact diagonalization might take a long time!")
    # get single site operaors
    sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
    id = sparse.csr_matrix(np.eye(2))
    sx_list = []  # sx_list[i] = kron([id, id, ..., id, sx, id, .... id])
    sz_list = []
    for i_site in range(L):
        x_ops = [id] * L
        z_ops = [id] * L
        x_ops[i_site] = sx
        z_ops[i_site] = sz
        X = x_ops[0]
        Z = z_ops[0]
        for j in range(1, L):
            X = sparse.kron(X, x_ops[j], 'csr')
            Z = sparse.kron(Z, z_ops[j], 'csr')
        sx_list.append(X)
        sz_list.append(Z)
    H_x = sparse.csr_matrix((2**L, 2**L))
    H_zz = sparse.csr_matrix((2**L, 2**L))
    for i in range(L - 1):
        H_zz = H_zz + sz_list[i] * sz_list[(i + 1) % L]
    for i in range(L):
        H_x = H_x + sx_list[i]
    H = -J * H_zz - h * H_x
    E, V = arp.eigsh(H, k=2, which='SA', return_eigenvectors=True, ncv=20)
    return E, V[:,0]

def exact_E_rand(L, J, h):
    """For comparison: obtain ground state energy from exact diagonalization.
    Exponentially expensive in L, only works for small enough `L` <~ 20.
    """
    if L >= 20:
        warnings.warn("Large L: Exact diagonalization might take a long time!")
    # get single site operaors
    sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
    id = sparse.csr_matrix(np.eye(2))
    sx_list = []  # sx_list[i] = kron([id, id, ..., id, sx, id, .... id])
    sz_list = []
    for i_site in range(L):
        x_ops = [id] * L
        z_ops = [id] * L
        x_ops[i_site] = sx
        z_ops[i_site] = sz
        X = x_ops[0]
        Z = z_ops[0]
        for j in range(1, L):
            X = sparse.kron(X, x_ops[j], 'csr')
            Z = sparse.kron(Z, z_ops[j], 'csr')
        sx_list.append(X)
        sz_list.append(Z)
    H_x = sparse.csr_matrix((2**L, 2**L))
    H_zz = sparse.csr_matrix((2**L, 2**L))
    for i in range(L - 1):
        rand_J = np.random.normal(J,0.1)
        H_zz = H_zz + rand_J*sz_list[i] * sz_list[(i + 1) % L]
    for i in range(L):
        H_x = H_x + sx_list[i]
    H = -H_zz - h * H_x
    E, V = arp.eigsh(H, k=2, which='SA', return_eigenvectors=True, ncv=20)
    return E, V[:,0]


# state_num = 41
# L = 6
# h = 1
# ratio_space = np.linspace(-2,2,state_num)
# for i in tqdm(range(0,state_num)):
#     J = ratio_space[i]*h
#     for k in tqdm(range(0,10)):
#         state = exact_E_rand(L, J, h)
#         np.save('6qubit/Ising_ground_state_6qubit_random_test' + str(round(ratio_space[i],1))+'_'+str(k), state)

# for i in tqdm(range(0,state_num)):
#     J = ratio_space[i]*h
#     for k in tqdm(range(0,50)):
#         state = exact_E_rand(L, J, h)
#         np.save('6qubit/Ising_ground_state_6qubit_random' + str(round(ratio_space[i],1))+'_'+str(k), state)


# state_num = int(input("number of ratios:"))
# L = 10
# h = 0.5
# ratio_space = np.linspace(0.5,2,state_num)
# for i in tqdm(range(0,state_num)):
#     J = ratio_space[i]*h
#     for k in tqdm(range(0,1)):
#         state = exact_E(L, J, h)
#         np.save('10qubit/Ising_ground_state_10qubit_random_fix' + str(J)+'_'+str(k), state)