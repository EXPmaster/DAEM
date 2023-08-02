import os
import numpy as np
import pickle
from a_mps import *
from tqdm import  tqdm


def generate_two_site_pauli_measurement_prob_distribution(J, all_test_bonds, all_eigvals, psi, Nsites, Paulis, chid = 2):
    dataset = []
    # dim all_test_bonds: [num_observables, num_noisescale, 4, 2, 2, 2, 2]
    for m_loc in range(0, Nsites - 1):
        for i in range(0, len(all_test_bonds)):
            obs = [Paulis[i // 3], Paulis[i % 3]]
            test_H_bonds = all_test_bonds[i]
            values = all_eigvals[i]
            results = []
            for noise_scale in range(len(test_H_bonds)):
                op = test_H_bonds[noise_scale]
                vs = values[noise_scale]
                pdf = []
                for i in range(len(op)):
                    theta = psi.get_theta2(m_loc)  # vL i j vR
                    op_theta = np.tensordot(op[i], theta, axes=([2, 3], [1, 2]))
                    # i j [i*] [j*], vL [i] [j] vR
                    pdf.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2, 3], [2, 0, 1, 3]]))
                    # [vL*] [i*] [j*] [vR*], [i] [j] [vL] [vR]
                # results.append(np.real_if_close(pdf))
                results.append(np.real_if_close(np.dot(vs, pdf)))
            results = np.array(results)
            selected_qubits = [m_loc, m_loc + 1]
            dataset.append([J, obs, selected_qubits, noise_scale, results[:-1], results[-1]])
    return dataset


def gen_large_data(data_root, save_path):
    Nsites = 50
    time = 5

    sxx = np.array([[0., 1.], [1., 0.]])
    syy = np.array([[0., -1j], [1j, 0.]])
    szz = np.array([[1., 0.], [0., -1.]])
    Paulis = [sxx, syy, szz]

    dataset = []
    for file_name in tqdm(os.listdir(data_root)):
        J = float(file_name.split('_')[-1])
        with open(os.path.join(data_root, file_name), "rb") as fp:
            psi = pickle.load(fp)
        
        obs_projectors = []
        obs_eigvals = []
        noise_scales = np.concatenate([np.round(np.arange(0.05, 0.29, 0.02), 3), [0.]])
        for p in noise_scales:
            
            # Dephasing noise
            p = (1 - np.exp(-2 * p)) / 2
            obs_x = (1 - p) * sxx + p * szz @ sxx @ szz
            obs_y = (1 - p) * syy + p * szz @ syy @ szz
            obs_z = szz

            # # Depolarizing noise
            # obs_x = (1 - p) * sxx + p / 2 * np.eye(2)
            # obs_y = (1 - p) * syy + p / 2 * np.eye(2)
            # obs_z = (1 - p) * szz + p / 2 * np.eye(2)

            Ex, Vx = np.linalg.eig(obs_x)
            Ey, Vy = np.linalg.eig(obs_y)
            Ez, Vz = np.linalg.eig(obs_z)
            V = [Vx, Vy, Vz]
            E = [Ex, Ey, Ez]
            mats = []
            eigvs = []
            for i in range(0,3):
                mat = []
                eigv = []
                for j in range(0,2):
                    v = V[i][:,j]
                    eigv.append(E[i][j])
                    tmp = np.matmul(v.reshape(2,1),v.reshape(1,2).conj())
                    mat.append(tmp)
                eigvs.append(eigv)
                mats.append(mat)

            chid = 2
            all_test_bonds = []
            all_eigen_values = []
            for mb_1 in range(0, 3):
                for mb_2 in range(0, 3):
                    test_H_bonds = []
                    eigen_vals = []
                    for i in range(0, 2):
                        for j in range(0, 2):
                            test_H_bond = np.kron(mats[mb_1][i], mats[mb_2][j])
                            test_H_bond = np.reshape(test_H_bond, [chid, chid, chid, chid])
                            test_H_bonds.append(test_H_bond)
                            eigen_vals.append(eigvs[mb_1][i] * eigvs[mb_2][j])
                    all_test_bonds.append(test_H_bonds)
                    all_eigen_values.append(eigen_vals)
            obs_projectors.append(all_test_bonds)
            obs_eigvals.append(all_eigen_values)
        # dim: [num_noisescale, num_observables, 4, 2, 2, 2, 2]
        obs_eigvals = np.transpose(np.array(obs_eigvals), (1, 0, 2))
        obs_projectors = np.transpose(np.array(obs_projectors), (1, 0, 2, 3, 4, 5, 6))
        data = generate_two_site_pauli_measurement_prob_distribution(J, obs_projectors, obs_eigvals, psi, Nsites, Paulis)
        dataset.extend(data)

    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    gen_large_data('../data_mitigate/mps/Random', '../data_mitigate/mps/trainset_phasedamp.pkl')
    gen_large_data('../data_mitigate/mps/Ising', '../data_mitigate/mps/testset_phasedamp.pkl')
