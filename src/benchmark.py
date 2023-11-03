import argparse
import random
import pickle
import functools
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from vqe import VQETrainer
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerLine2D
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from qiskit import Aer, QuantumCircuit, transpile, execute
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import DensityMatrix, state_fidelity
from qiskit.opflow import PauliOp
from tqdm import tqdm
from torchquantum import switch_little_big_endian_state

from model import *
from utils import AverageMeter, abs_deviation, gen_rand_obs
from my_envs import IBMQEnv
from datasets import MitigateDataset
from cdr_trainer import CDRTrainer
from zne_trainer import ZNETrainer


@torch.no_grad()
def evaluate_new():
    eval_results = []
    # load GAN model
    ckpt = torch.load(args.weight_path, map_location=args.device)
    model_g = SuperviseModel(args.num_qubits)
    model_g.load_state_dict(ckpt['model_g'], strict=False)
    # model_g.load_envs(args, force=True)
    model_g.to(args.device)
    model_g.eval()

    cdr_model = CDRTrainer(args.env_path)

    with open(args.testset, 'rb') as f:
        testset = pickle.load(f)

    # random.shuffle(testset)
    # For all

    all_results = pd.DataFrame(columns=['g', 'QEM strategy', 'MAE'])

    for params, obs, pos, scale, exp_noisy, exp_ideal in tqdm(testset):

        # CDR
        cdr_model.fit(params, functools.reduce(np.kron, obs[::-1]))

        # ZNE
        zne_model = ZNETrainer()
        zne_predicts = zne_model.fit_and_predict(exp_noisy)[-1]
        # zne_model.plot_fig(exp_noisy)
        # assert False
        meas_ideal = exp_ideal
        meas_noisy = exp_noisy[0]
        all_results = all_results.append(
            {'g': params, 'QEM strategy': 'w/o', 'MAE': abs(meas_ideal - meas_noisy)},
            ignore_index=True
        )
        
        # NN prediction
        # obs_kron = np.kron(obs[0], obs[1])
        obs = torch.tensor(obs, dtype=torch.cfloat)[None].to(args.device)
        param = torch.FloatTensor([params])[None].to(args.device)
        pos = torch.tensor(pos)[None].to(args.device)
        scale = torch.FloatTensor([0.])[None].to(args.device)
        exp_noisy = torch.FloatTensor(exp_noisy)[None].to(args.device)
        predicts = model_g(param, obs, pos, scale, exp_noisy).squeeze().item()
        all_results = all_results.append(
            {'g': params, 'QEM strategy': 'EM-NET', 'MAE': abs(meas_ideal - predicts)},
            ignore_index=True
        )

        # ZNE prediction
        all_results = all_results.append(
            {'g': params, 'QEM strategy': 'ZNE', 'MAE': abs(zne_predicts - meas_ideal)},
            ignore_index=True
        )

        # CDR prediction
        cdr_predicts = cdr_model.predict(np.array(meas_noisy).reshape(-1, 1))
        all_results = all_results.append(
            {'g': params, 'QEM strategy': 'CDR', 'MAE': abs(cdr_predicts - meas_ideal)},
            ignore_index=True
        )
    all_results.to_pickle('../result_data/vqe4l_markov_phasedamp_results.pkl')


def draw_vqe():
    all_results = pd.read_pickle('../result_data/vqe4l_markov_phasedamp_results.pkl')
    # sns.set(font_scale=1.5)
    # plt.rcParams.update({'font.size': 35, 'figure.figsize': (14.5, 11)})
    sns.set_style('ticks')
    sns.set_context('paper', font_scale=2.0)
    ax = sns.lineplot(data=all_results, x='g', y='MAE', hue='QEM strategy',
                style='QEM strategy',
                errorbar=('se', 0.2),
                linestyle='',
                markeredgecolor=None,
                marker='o', dashes=False, sort=True)
    
    for child in ax.findobj(PolyCollection):
        child.set_edgecolor(None)

    def update_prop(handle, orig):
        handle.update_from(orig)
        handle.set_linestyle('')
        handle.set_marker("o")

    ax.set_ylim(0.0, 0.26)
    ax.set_xlim(0.3, 2.0)
    # ax.set_xlabel('g', fontsize=15)
    # ax.set_ylabel('MAE', fontsize=15)
    ax.get_legend().remove()
    # plt.legend(loc=(0.01, 0.75), title="QEM strategy", handler_map={plt.Line2D: HandlerLine2D(update_func=update_prop)})
    plt.savefig('../figures/vqe_4qubits_markov_phasedamp.pdf', bbox_inches='tight')


@torch.no_grad()
def evaluate_swaptest():
    eval_results = []
    # load GAN model
    ckpt = torch.load(args.weight_path, map_location=args.device)
    model_g = SuperviseModel(args.num_qubits)
    model_g.load_state_dict(ckpt['model_g'], strict=False)
    # model_g.load_envs(args, force=True)
    model_g.to(args.device)
    model_g.eval()

    cdr_model = CDRTrainer(None, '../runs/cdr_st11q.pkl')

    with open(args.testset, 'rb') as f:
        testset = pickle.load(f)

    # random.shuffle(testset)
    # For all
    with open('../data_mitigate/data_st11q_pd/testset_fixed.pkl', 'rb') as f:
        testset_fixed = pickle.load(f)

    all_results = pd.DataFrame(columns=['Input state', 'QEM strategy', 'MAE', 'Fidelity'])
    testset = testset + testset_fixed

    for idx, (params, obs, pos, scale, exp_noisy, exp_ideal) in tqdm(enumerate(testset)):
        # ZNE
        zne_model = ZNETrainer()
        zne_predicts = zne_model.fit_and_predict(exp_noisy)[-1]
        # zne_model.plot_fig(exp_noisy)
        # assert False
        meas_ideal = exp_ideal
        meas_noisy = exp_noisy[0]
        all_results = all_results.append(
            {'Input state': idx, 'QEM strategy': 'w/o', 'MAE': abs(meas_ideal - meas_noisy),
            'Fidelity': exp_ideal},
            ignore_index=True
        )
        
        # GAN prediction
        # obs_kron = np.kron(obs[0], obs[1])
        obs = torch.tensor(obs, dtype=torch.cfloat)[None].to(args.device)
        param = torch.FloatTensor([params])[None].to(args.device)
        pos = torch.tensor(pos)[None].to(args.device)
        scale = torch.FloatTensor([0.])[None].to(args.device)
        exp_noisy = torch.FloatTensor(exp_noisy)[None].to(args.device)
        predicts = model_g(param, obs, pos, scale, exp_noisy).squeeze().item()
        all_results = all_results.append(
            {'Input state': idx, 'QEM strategy': 'DAEM', 'MAE': abs(meas_ideal - predicts),
            'Fidelity': predicts},
            ignore_index=True
        )

        # ZNE prediction
        all_results = all_results.append(
            {'Input state': idx, 'QEM strategy': 'ZNE', 'MAE': abs(zne_predicts - meas_ideal),
            'Fidelity': zne_predicts},
            ignore_index=True
        )

         # CDR prediction
        cdr_predicts = cdr_model.predict(np.array(meas_noisy).reshape(-1, 1))
        all_results = all_results.append(
            {'Input state': idx, 'QEM strategy': 'CDR', 'MAE': abs(cdr_predicts - meas_ideal),
            'Fidelity': cdr_predicts},
            ignore_index=True
        )
    all_results.to_pickle('../result_data/swaptest_11qubits_phasedamp_results.pkl')
       

def draw_swaptest():
    from matplotlib import ticker
    all_results = pd.read_pickle('../result_data/swaptest_11qubits_phasedamp_results.pkl')
    # sns.lineplot(data=all_results, x='Input state', y='MAE', hue='QEM strategy',
    #             style='QEM strategy', markers=True, dashes=False, sort=True)
    # sns.stripplot(data=all_results, x="MAE", y="QEM strategy", hue="QEM strategy", size=2)
    sns.set_style('ticks')
    sns.set_context('paper', font_scale=2.0)
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 3]})

    sns.boxplot(data=all_results, x="QEM strategy", y="MAE", hue="QEM strategy", dodge=False, width=0.3, ax=ax1, palette='muted')
    sns.boxplot(data=all_results, x="QEM strategy", y="MAE", hue="QEM strategy", dodge=False, width=0.3, ax=ax2, palette='muted')
    ax1.set_ylim(0.1, 0.14)
    ax2.set_ylim(0, 0.04)
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    # hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    ax1.set_ylabel("")
    ax2.set_ylabel("")
    # by default, seaborn also gives each subplot its own legend, which makes no sense at all
    # soe remove both default legends first
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    # then create a new legend and put it to the side of the figure (also requires trial and error)
    # ax1.legend(loc=(0.6, -0.2))  # , title="QEM strategy")
    # set a new label on the plot (basically just a piece of text) and move it to where it makes sense
    f.text(-0.03, 0.5, "MAE", va="center", rotation="vertical")
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    plt.savefig('../figures/swaptest_11qubits_phasedamp.pdf', bbox_inches='tight')
    plt.close()
    length = len(all_results)
    ax = sns.barplot(all_results.iloc[32: 36], x='QEM strategy', y='Fidelity', palette='muted')
    ax.set_xticklabels(['Ideal', 'DAEM', 'ZNE', 'CDR'])
    ax.set_ylim(0, 1)
    plt.savefig('../figures/swaptest_state1_fidelity.pdf', bbox_inches='tight')
    plt.close()
    ax = sns.barplot(all_results.iloc[length - 8: length - 4], x='QEM strategy', y='Fidelity', palette='muted')
    ax.set_xticklabels(['Ideal', 'DAEM', 'ZNE', 'CDR'])
    ax.set_ylim(0, 1)
    plt.savefig('../figures/swaptest_state2_fidelity.pdf', bbox_inches='tight')
    plt.close()
    ax = sns.barplot(all_results.iloc[24: 28], x='QEM strategy', y='Fidelity', palette='muted')
    ax.set_xticklabels(['Ideal', 'DAEM', 'ZNE', 'CDR'])
    ax.set_ylim(0, 1)
    plt.savefig('../figures/swaptest_state3_fidelity.pdf', bbox_inches='tight')
    plt.close()


@torch.no_grad()
def evaluate_mps():
    eval_results = []
    # load GAN model
    ckpt = torch.load(args.weight_path, map_location=args.device)
    model_g = SuperviseModel(args.num_qubits)
    model_g.load_state_dict(ckpt['model_g'], strict=False)
    # model_g.load_envs(args, force=True)
    model_g.to(args.device)
    model_g.eval()

    with open(args.testset, 'rb') as f:
        testset = pickle.load(f)

    # random.shuffle(testset)
    # For all

    all_results = pd.DataFrame(columns=['Ising coefficient', 'QEM strategy', 'MAE'])

    for params, obs, pos, scale, exp_noisy, exp_ideal in tqdm(testset):
        # ZNE
        zne_model = ZNETrainer()
        zne_predicts = zne_model.fit_and_predict(exp_noisy)[-1]
        # zne_model.plot_fig(exp_noisy)
        # assert False
        meas_ideal = exp_ideal
        meas_noisy = exp_noisy[0]
        # all_results[params][0].append(abs(meas_ideal - meas_noisy))
        all_results = all_results.append(
            {'Ising coefficient': params, 'QEM strategy': 'w/o', 'MAE': abs(meas_ideal - meas_noisy)},
            ignore_index=True
        )
        # GAN prediction
        # obs_kron = np.kron(obs[0], obs[1])
        obs = torch.tensor(obs, dtype=torch.cfloat)[None].to(args.device)
        param = torch.FloatTensor([params])[None].to(args.device)
        pos = torch.tensor(pos)[None].to(args.device)
        scale = torch.FloatTensor([0.])[None].to(args.device)
        exp_noisy = torch.FloatTensor(exp_noisy)[None].to(args.device)
        predicts = model_g(param, obs, pos, scale, exp_noisy).squeeze().item()
        all_results = all_results.append(
            {'Ising coefficient': params, 'QEM strategy': 'EM-NET', 'MAE': abs(meas_ideal - predicts)},
            ignore_index=True
        )

        # ZNE prediction
        all_results = all_results.append(
            {'Ising coefficient': params, 'QEM strategy': 'ZNE', 'MAE': abs(zne_predicts - meas_ideal)},
            ignore_index=True
        )
    all_results.to_pickle('../result_data/mps_ampdamp_results.pkl')


def draw_mps():
    all_results = pd.read_pickle('../result_data/mps_ampdamp_results.pkl')
    sns.set_style('ticks')
    sns.set_context('paper', font_scale=2.0)
    ax = sns.lineplot(data=all_results, x='Ising coefficient', y='MAE', hue='QEM strategy',
                style='QEM strategy',
                errorbar=('se', 1.0),
                linestyle='',
                markeredgecolor=None,
                marker='o', dashes=False, sort=True)
    
    for child in ax.findobj(PolyCollection):
        child.set_edgecolor(None)

    def update_prop(handle, orig):
        handle.update_from(orig)
        handle.set_linestyle('')
        handle.set_marker("o")
    # ax.set_ylim(0, 0.03)
    ax.set_xlabel('g')
    ax.get_legend().remove()
    # plt.legend(handler_map={plt.Line2D: HandlerLine2D(update_func=update_prop)})
    plt.savefig('../figures/ising_50qubits_ampdamp.pdf', bbox_inches='tight')


@torch.no_grad()
def evaluate_qaoa():
    from qiskit.circuit.library.standard_gates import HGate, SdgGate, IGate
    eval_results = []
    # load GAN model
    ckpt = torch.load(args.weight_path, map_location=args.device)
    model_g = SuperviseModel(args.num_qubits, mitigate_prob=True)
    model_g.load_state_dict(ckpt['model_g'], strict=False)
    # model_g.load_envs(args, force=True)
    model_g.to(args.device)
    model_g.eval()

    with open(args.testset, 'rb') as f:
        testset = pickle.load(f)

    # random.shuffle(testset)
    # For all

    all_results = pd.DataFrame(columns=['param', 'QEM strategy', 'Bitstring', 'Frequency'])

    for idx, (params, obs, pos, scale, exp_noisy, exp_ideal) in tqdm(enumerate(testset)):
        meas_ideal = exp_ideal
        meas_noisy = exp_noisy[0]
        # GAN prediction
        # obs_kron = np.kron(obs[0], obs[1])
        obs = torch.tensor(obs, dtype=torch.cfloat)[None].to(args.device)
        param = torch.FloatTensor([params])[None].to(args.device)
        position = torch.tensor(pos)[None].to(args.device)
        scale = torch.FloatTensor([0.])[None].to(args.device)
        exp_noisy = torch.FloatTensor(exp_noisy)[None].to(args.device)
        # print(meas_noisy)
        # print(exp_ideal)
        # assert False
        predicts = model_g(param, obs, position, scale, exp_noisy).softmax(1).squeeze().cpu().numpy()
        for k in range(len(meas_noisy)):
            bitstring = bin(k)[2:].zfill(6)
            all_results = all_results.append(
                {
                    'param': params,
                    'QEM strategy': 'Noisy',
                    'Bitstring': bitstring,
                    'Frequency': meas_noisy[k],
                },
                ignore_index=True
            )
            all_results = all_results.append(
                {
                    'param': params,
                    'QEM strategy': 'EM-NET',
                    'Bitstring': bitstring,
                    'Frequency': predicts[k],
                },
                ignore_index=True
            )
            all_results = all_results.append(
                {
                    'param': params,
                    'QEM strategy': 'Ideal',
                    'Bitstring': bitstring,
                    'Frequency': meas_ideal[k],
                },
                ignore_index=True
            )
    all_results.to_pickle('../result_data/qaoa_results.pkl')


def draw_qaoa():
    from rustworkx.visualization import mpl_draw
    all_results = pd.read_pickle('../result_data/qaoa_results.pkl')
    graphs = pd.read_pickle('../data_mitigate/qaoa_6q_dep_distr/graphs.pkl')
    
    for select_param, g in enumerate(graphs):
        mpl_draw(g, node_size=600, linewidths=2)
        plt.savefig(f'../figures/viz_qaoa/qaoa_6qubits_depolarize_{select_param}_graph.pdf', bbox_inches='tight')
        plt.close()
    
    sns.set(rc={'figure.figsize':(32, 8)})
    sns.set_style('ticks')
    sns.set_context('paper', font_scale=3.0)
    for select_param in range(20):
        ax = sns.barplot(
            # data=all_results.loc[all_results['param'] == 2],
            data=all_results.loc[(all_results['param'] == select_param) & (all_results['QEM strategy'] == 'Noisy')],
            x='Bitstring', y='Frequency', palette=[sns.color_palette('muted')[0]],
        )
        ax.set_ylim(0, 0.3)
        ax.set_ylabel("")
        ax.get_xaxis().set_visible(False)
        plt.savefig(f'../figures/viz_qaoa/qaoa_6qubits_depolarize_{select_param}_noisy.pdf', bbox_inches='tight')
        plt.close()

        ax = sns.barplot(
            # data=all_results.loc[all_results['param'] == 2],
            data=all_results.loc[(all_results['param'] == select_param) & (all_results['QEM strategy'] == 'Ideal')],
            x='Bitstring', y='Frequency', palette=[sns.color_palette('Set2')[0]]
        )
        ax.set_ylim(0, 0.3)
        ax.set_ylabel("")
        ax.get_xaxis().set_visible(False)
        plt.savefig(f'../figures/viz_qaoa/qaoa_6qubits_depolarize_{select_param}_ideal.pdf', bbox_inches='tight')
        plt.close()

        ax = sns.barplot(
            # data=all_results.loc[all_results['param'] == 2],
            data=all_results.loc[(all_results['param'] == select_param) & (all_results['QEM strategy'] == 'EM-NET')],
            x='Bitstring', y='Frequency', palette=[sns.color_palette('muted')[1]],
        )
        ax.set_ylim(0, 0.3)
        ax.set_ylabel("")
        ax.get_xaxis().set_visible(False)
        plt.savefig(f'../figures/viz_qaoa/qaoa_6qubits_depolarize_{select_param}_mitigate.pdf', bbox_inches='tight')
        plt.close()


@torch.no_grad()
def evaluate_cv():
    distance = nn.CosineSimilarity()
    # distance = nn.KLDivLoss(reduction="batchmean", log_target=False)
    eval_results = []
    # load Supervised model
    ckpt = torch.load(args.weight_path, map_location=args.device)
    model_g = CvModel()
    model_g.load_state_dict(ckpt['model_g'], strict=False)
    model_g.to(args.device)
    model_g.eval()

    with open(args.testset, 'rb') as f:
        testset = pickle.load(f)

    # For all
    all_results = pd.DataFrame(columns=['Evolution time', 'QEM strategy', 'Fidelity'])

    for params, exp_noisy, exp_ideal in tqdm(testset):
        if abs(params - 1.0) < 1e-4: continue
        meas_ideal = torch.FloatTensor(exp_ideal)[None].flatten(1)
        meas_noisy = torch.FloatTensor(exp_noisy[0])[None].flatten(1)
        all_results = all_results.append(
            {'Evolution time': params, 'QEM strategy': 'w/o', 'Fidelity': distance(meas_noisy, meas_ideal).item()},
            ignore_index=True
        )
        # Supervised prediction
        param = torch.FloatTensor([params])[None].to(args.device)
        exp_noisy_tensor = torch.FloatTensor(exp_noisy)[None].to(args.device)
        predicts = model_g(param, exp_noisy_tensor).flatten(1).cpu()
        # all_results[params][1].append(distance(predicts, meas_ideal).item())
        all_results = all_results.append(
            {'Evolution time': params, 'QEM strategy': 'EM-NET', 'Fidelity': distance(predicts, meas_ideal).item()},
            ignore_index=True
        )
        predicts = predicts.reshape(exp_ideal.shape).numpy()
        eval_results.append([params, exp_noisy[0], predicts, exp_ideal])
    
    with open('../result_data/cv_results.pkl', 'wb') as f:
        pickle.dump([all_results, eval_results], f)


def draw_cv():
    import matplotlib as mpl
    with open('../result_data/cv_results.pkl', 'rb') as f:
        all_results, eval_results = pickle.load(f)

    xvec = np.linspace(-4, 4, 48)
    cmap = "RdBu_r"
    recorded_params = []

    def draw_img(value, save_path):
        fig, ax = plt.subplots()
        img = ax.pcolormesh(xvec, xvec, value, cmap=cmap, vmin=-0.02, vmax=0.02)
        ax.set_aspect(1)
        # fig.colorbar(img, ax=ax)
        plt.savefig(save_path)
        plt.close()
    
    plt.rcParams.update({'font.size': 17})
    fig, ax = plt.subplots(figsize=(0.5, 10))
    norm = mpl.colors.Normalize(vmin=-0.02, vmax=0.02)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    plt.savefig('../figures/viz_cv/colorbar.pdf', bbox_inches='tight')
    plt.close()

    plt.rcParams.update({'font.size': 25})

    for i, (param, noisy_meas, predict_meas, ideal_meas) in enumerate(eval_results):
        draw_img(predict_meas, os.path.join('../figures/viz_cv', f'state_{round(param, 3)}_predict.pdf'))
        draw_img(ideal_meas, os.path.join('../figures/viz_cv', f'state_{round(param, 3)}_ideal.pdf'))
        draw_img(noisy_meas, os.path.join('../figures/viz_cv', f'state_{round(param, 3)}_noisy.pdf'))
    
    sns.set_style('ticks')
    sns.set_context('paper', font_scale=2.0)
    ax = sns.lineplot(data=all_results, x='Evolution time', y='Fidelity', hue='QEM strategy',
                style='QEM strategy',
                markeredgecolor=None,
                marker='o',
                dashes=False, sort=True)
    ax.legend().set_title('')
    plt.savefig('../figures/continuous_variable_fidelity.pdf', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-path', default='../environments/circuits/vqe_4l', type=str)
    parser.add_argument('--weight-path', default='../runs/env_st11q_pd_2023-09-06-16-12/gan_model.pt', type=str)
    parser.add_argument('--testset', default='../data_mitigate/data_st11q_pd/testset.pkl', type=str)
    parser.add_argument('--num-qubits', default=50, type=int, help='number of qubits')
    parser.add_argument('--gpus', default='0', type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # evaluate_new()
    # evaluate_qaoa()
    # evaluate_cv()
    # evaluate_mps()
    # evaluate_swaptest()

    # draw_qaoa()
    # draw_vqe()
    # draw_cv()
    # draw_mps()
    draw_swaptest()
