# python tools/evaluate.py --policy_path log_origin --policy_name Unimal-v0
import argparse
import os
import torch
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

from metamorph.config import cfg
from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import *
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu
from metamorph.envs.vec_env.pytorch_vec_env import VecPyTorch
from metamorph.algos.ppo.model import Agent

from collections import defaultdict


def set_cfg_options():
    calculate_max_iters()
    maybe_infer_walkers()
    calculate_max_limbs_joints()


def calculate_max_limbs_joints():
    if cfg.ENV_NAME != "Unimal-v0":
        return

    num_joints, num_limbs = [], []

    metadata_paths = []
    for agent in cfg.ENV.WALKERS:
        metadata_paths.append(os.path.join(
            cfg.ENV.WALKER_DIR, "metadata", "{}.json".format(agent)
        ))

    for metadata_path in metadata_paths:
        metadata = fu.load_json(metadata_path)
        num_joints.append(metadata["dof"])
        num_limbs.append(metadata["num_limbs"] + 1)

    # Add extra 1 for max_joints; needed for adding edge padding
    cfg.MODEL.MAX_JOINTS = max(num_joints) + 1
    cfg.MODEL.MAX_LIMBS = max(num_limbs) + 1
    cfg.MODEL.MAX_JOINTS = 16
    cfg.MODEL.MAX_LIMBS = 12
    print (cfg.MODEL.MAX_JOINTS, cfg.MODEL.MAX_LIMBS)


def calculate_max_iters():
    # Iter here refers to 1 cycle of experience collection and policy update.
    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )
    cfg.PPO.EARLY_EXIT_MAX_ITERS = (
        int(cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )


def maybe_infer_walkers():
    if cfg.ENV_NAME != "Unimal-v0":
        return

    # Only infer the walkers if this option was not specified
    if len(cfg.ENV.WALKERS):
        return

    cfg.ENV.WALKERS = [
        xml_file.split(".")[0]
        for xml_file in os.listdir(os.path.join(cfg.ENV.WALKER_DIR, "xml"))
    ]


# plot histgram of a context feature
def plot_context_hist(feature):

    cfg.ENV.WALKER_DIR = 'unimals_100/train'
    cfg.ENV.WALKERS = []
    cfg.OUT_DIR = './eval'
    cfg.MODEL.CONTEXT_OBS_TYPES = [feature]
    cfg.PPO.NUM_ENVS = 2
    set_cfg_options()

    agents = list(os.listdir('unimals_single_task'))
    context = []
    print (feature)
    for agent in agents:
        env = make_env(cfg.ENV_NAME, 0, 0, xml_file=agent)()
        obs = env.reset()
        context.append(obs['context'].reshape(cfg.MODEL.MAX_LIMBS, -1)[:env.metadata["num_limbs"]])
        env.close()
    context = np.concatenate(context, axis=0)

    print (context.min(axis=0))
    print (context.max(axis=0))

    # for i in range(context.shape[1]):
    #     plt.figure()
    #     plt.hist(context[:, i], bins=100)
    #     plt.title(f'{feature}_{i}: min: {context[:, i].min()}, max: {context[:, i].max()}')
    #     plt.savefig(f'figures/context_hist/{feature}_{i}.png')
    #     plt.close()


def plot_training_stats(folders, names=None, key='grad_norm', batch_size=8*16, kl_threshold=None, prefix=None):
    if names is None:
        names = folders
    if kl_threshold is None:
        kl_threshold = [0.2 for _ in range(len(folders))]
    plt.figure()
    # ES_record = {}
    all_folders_ES_idx = {}
    for n, folder in enumerate(folders):
        all_seeds_stat = []
        all_seeds_ES_idx = []
        for seed in os.listdir('output/' + folder):
            # if seed == '1409' and folder == 'ft_baseline_KL_5_wo_PE+dropout':
            #     continue
            if 'checkpoint_100.pt' not in os.listdir(os.path.join('output', folder, seed)):
                continue
            with open(os.path.join('output', folder, seed, 'Unimal-v0_results.json'), 'r') as f:
                log = json.load(f)
            try:
                data = np.array(log['__env__'][key])
            except:
                continue
            approx_kl = np.array(log['__env__']['approx_kl'])
            
            iter_avg = []
            early_stop_idx = []
            i = 0
            while i < len(data):
                # check for early stop
                early_stop_pos = np.where(approx_kl[i:(i + batch_size)] > kl_threshold[n])[0]
                if len(early_stop_pos) > 0:
                    end_index = i + early_stop_pos[0] + 1
                    early_stop_idx.append(early_stop_pos[0])
                else:
                    end_index = i + batch_size
                    early_stop_idx.append(batch_size)

                avg = data[i:end_index].mean()
                iter_avg.append(avg)
                i = end_index

            all_seeds_stat.append(iter_avg)
            # plt.plot(iter_avg, label=f'{folder}-{seed}')
            all_seeds_ES_idx.append(np.array(early_stop_idx))
        if len(all_seeds_stat) == 0:
            continue
        min_len = min([len(x) for x in all_seeds_stat])
        avg_stat = np.stack([np.array(x[:min_len]) for x in all_seeds_stat]).mean(0)
        all_folders_ES_idx[folder] = np.stack([x[:min_len] for x in all_seeds_ES_idx]).mean(0)
        plt.plot(avg_stat, label=names[n])
    plt.legend()
    plt.xlabel('update num')
    plt.ylabel(key)
    if prefix is None:
        plt.savefig(f'figures/train_stats/{key}.png')
    else:
        plt.savefig(f'figures/train_stats/{prefix}_{key}.png')
    plt.close()

    plt.figure()
    print (all_folders_ES_idx.keys())
    w = 10
    for i, folder in enumerate(folders):
        plt.plot(np.convolve(all_folders_ES_idx[folder], np.ones(w), 'valid') / w, \
            label=f'{names[i]}: {all_folders_ES_idx[folder].mean():.2f}')
    plt.legend()
    plt.savefig(f'figures/train_stats/{prefix}_ES_index.png')
    plt.close()


def plot_test_performance(results):
    scores = []
    for r in results:
        with open(f'eval/{r}.pkl', 'rb') as f:
            score = pickle.load(f)
            scores.append(score)
    
    idx = np.argsort(-scores[0])
    plt.figure()
    for i, score in enumerate(scores):
        plt.plot(score[idx], label=results[i])
        print (results[i], score.mean())
    plt.legend()
    plt.savefig('figures/test_results.png')
    plt.close()


def scatter_train_performance_compared_to_subset_train(folder, option):

    MT_agent_score = defaultdict(list)
    for seed in os.listdir('output/' + folder):
        with open(os.path.join('output', folder, seed, 'Unimal-v0_results.json'), 'r') as f:
            log = json.load(f)
        del log['__env__']
        del log['fps']
        for agent in log:
            s = log[agent]['reward']['reward'][-1]
            MT_agent_score[agent].append(s)

    subset_agent_score = defaultdict(list)
    for index in [0, 1, 2]:
        folder = f'output/log_train_subset_{index}'
        with open(f'{folder}/Unimal-v0_results.json', 'r') as f:
            log = json.load(f)
        del log['__env__']
        del log['fps']
        for agent in log:
            s = log[agent]['reward']['reward'][-1]
            subset_agent_score[agent].append(s)

    ST_agent_score = defaultdict(list)
    ST_folder = 'output/log_single_task'
    for agent in os.listdir(ST_folder):
        for seed in os.listdir(os.path.join(ST_folder, agent)):
            path = os.path.join(ST_folder, agent, seed)
            # use this if check to avoid the seed which is not fully trained
            if 'checkpoint_100.pt' in os.listdir(path):
                with open(f'{path}/Unimal-v0_results.json', 'r') as f:
                    log = json.load(f)
                s = log[agent]['reward']['reward'][-1]
                ST_agent_score[agent].append(s)

    if option == 'ST-subset':
        x = ST_agent_score
        y = subset_agent_score
        x_label = 'single task train'
        y_label = 'train on 10 morphologies'
        fig_name = 'compare_final_train_score_ST_subset'

    plt.figure()
    for agent in subset_agent_score:
        plt.scatter(np.array(x[agent]).mean(), np.array(y[agent]).mean(), c='blue')
    plt.plot([500, 5500], [500, 5500], 'k-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f'figures/{fig_name}.png')
    plt.close()


def get_context():

    cfg.ENV.WALKER_DIR = 'unimals_100/test'
    cfg.ENV.WALKERS = []
    cfg.OUT_DIR = './eval'
    cfg.PPO.NUM_ENVS = 2
    set_cfg_options()

    agents = list(os.listdir('unimals_single_task_test'))
    context = {}
    for agent in agents:
        env = make_env(cfg.ENV_NAME, 0, 0, xml_file=agent)()
        obs = env.reset()
        # context[agent] = obs['context'].reshape(cfg.MODEL.MAX_LIMBS, -1)[:env.metadata["num_limbs"]]
        env.close()
    
    # with open('train_context.pkl', 'wb') as f:
        # pickle.dump(context, f)


def analyze_ratio_hist(folders):
    seed = 1410
    colors = ['blue', 'orange']

    for i in range(0, 1200, 50):
        plt.figure(figsize=(16, 9))
        for j, folder in enumerate(folders):
            if f'ratio_hist_{i}.pkl' not in os.listdir(f'output/{folder}/{seed}/ratio_hist'):
                return
            with open(f'output/{folder}/{seed}/ratio_hist/ratio_hist_{i}.pkl', 'rb') as f:
                hist = pickle.load(f)
            for epoch in range(8):
                if len(hist[epoch]) == 0:
                    break
                batch_avg = np.stack([np.array(x) for x in hist[epoch]]).mean(axis=0)
                plt.subplot(2, 4, epoch + 1)
                plt.plot(np.linspace(0., 2., 100), batch_avg, label=folder, c=colors[j])
                plt.plot([1., 1.], [0, batch_avg.max()], '-k')
        plt.legend()
        plt.savefig(f'figures/ratio_hist/{i}.png')
        plt.close()


def analyze_ratio_hist_trend(folders, prefix=None):
    seed = 1410
    colors = ['blue', 'orange']

    for i in range(0, 1200, 50):
        plt.figure(figsize=(16, 9))
        for j, folder in enumerate(folders):
            plt.subplot(1, 2, j + 1)
            if f'ratio_hist_{i}.pkl' not in os.listdir(f'output/{folder}/{seed}/ratio_hist'):
                return
            with open(f'output/{folder}/{seed}/ratio_hist/ratio_hist_{i}.pkl', 'rb') as f:
                hist = pickle.load(f)
            for epoch in range(8):
                if len(hist[epoch]) == 0:
                    break
                batch_avg = np.stack([np.array(x) for x in hist[epoch]]).mean(axis=0)
                plt.plot(np.linspace(0., 2., 100), batch_avg, c=colors[j], alpha=0.3 + epoch * 0.1)
                plt.plot([1., 1.], [0, batch_avg.max()], '--k')
            plt.legend()
        if prefix is None:
            plt.savefig(f'figures/ratio_hist/trend_{i}.png')
        else:
            plt.savefig(f'figures/ratio_hist/trend_{prefix}_{i}.png')
        plt.close()


def test_init_ratio_hist(folders, prefix=None):
    seed = 1410
    colors = ['blue', 'orange']

    for i in range(0, 1200, 50):
        plt.figure(figsize=(16, 9))
        for j, folder in enumerate(folders):
            plt.subplot(1, 2, j + 1)
            if f'ratio_hist_{i}.pkl' not in os.listdir(f'output/{folder}/{seed}/ratio_hist'):
                return
            with open(f'output/{folder}/{seed}/ratio_hist/ratio_hist_{i}.pkl', 'rb') as f:
                hist = pickle.load(f)
            batch_init = np.array(hist[0][0])
            plt.plot(np.linspace(0., 2., 100), batch_init, c=colors[j])
            plt.plot([1., 1.], [0, batch_init.max()], '-k')
            plt.legend()
        if prefix is None:
            plt.savefig(f'figures/ratio_hist/init_ratio_{i}.png')
        else:
            plt.savefig(f'figures/ratio_hist/init_ratio_{prefix}_{i}.png')
        plt.close()


def scatter_test_score(x, y, name_x, name_y):
    plt.figure()
    for agent in x:
        plt.scatter(np.array(x[agent]).mean(), np.array(y[agent]).mean())
    plt.plot([500, 5500], [500, 5500], 'k-')
    plt.xlabel(name_x)
    plt.ylabel(name_y)
    plt.savefig(f'figures/compare_test_score/{name_x}-{name_y}.png')
    plt.close()


def plot_test_score(files, fig_name):
    scores = {}
    for i, x in enumerate(files):
        with open(f'eval/{x}.pkl', 'rb') as f:
            score = pickle.load(f)
        scores[x] = score
        if i == 0:
            agent_list = list(score.keys())
            avg_score = np.array([score[agent].mean() for agent in agent_list])
            order = np.argsort(avg_score)
    
    plt.figure()
    for f in scores:
        all_agent_score = np.stack([scores[f][agent] for agent in agent_list])
        all_agent_score = all_agent_score[order]
        avg_score = all_agent_score.mean(axis=1)
        std_score = all_agent_score.std(axis=1)
        plt.plot(avg_score, label=f'{f}: {avg_score.mean():.0f} +- {std_score.mean():.0f}')
        plt.fill_between(np.arange(len(avg_score)), avg_score - std_score, avg_score + std_score, alpha=0.25)
    plt.legend()
    plt.savefig(f'figures/compare_test_score/{fig_name}.png')
    plt.close()


def scatter_test_score(x, y):
    suffix = [
        '_terminate_on_fall_deterministic', 
        '_terminate_on_fall', 
        '_deterministic', 
        '', 
    ]

    plt.figure(figsize=(16, 12))
    for i, suf in enumerate(suffix):
        with open(f'eval/{x}{suf}.pkl', 'rb') as f:
            result_x = pickle.load(f)
        with open(f'eval/{y}{suf}.pkl', 'rb') as f:
            result_y = pickle.load(f)
        agents = list(result_x.keys())
        avg_x = np.array([np.array(result_x[r]).mean() for r in agents])
        avg_y = np.array([np.array(result_y[r]).mean() for r in agents])
        std_x = np.array([np.array(result_x[r]).std() for r in agents])
        std_y = np.array([np.array(result_y[r]).std() for r in agents])
        plt.subplot(2, 2, i + 1)
        plt.plot([0, 6000], [0, 6000], '--k')
        plt.scatter(avg_x, avg_y)
        if i == 0:
            plt.xlabel(f'{x}: {avg_x.mean():.2f} +- {std_x.mean():.2f}')
            plt.ylabel(f'{y}: {avg_y.mean():.2f} +- {std_y.mean():.2f}')
        else:
            plt.xlabel(f'{avg_x.mean():.2f} +- {std_x.mean():.2f}')
            plt.ylabel(f'{avg_y.mean():.2f} +- {std_y.mean():.2f}')
        plt.title(suf)
    plt.savefig(f'figures/compare_test_score/compare_score_{x}-vs-{y}.png')
    plt.close()


def scatter_train_score(x, y):

    score_x = {}
    with open(f'output/{x}/Unimal-v0_results.json', 'r') as f:
        log = json.load(f)
    del log['__env__']
    del log['fps']
    for agent in log:
        score_x[agent] = log[agent]['reward']['reward'][-1]

    score_y = {}
    with open(f'output/{y}/Unimal-v0_results.json', 'r') as f:
        log = json.load(f)
    del log['__env__']
    del log['fps']
    for agent in log:
        score_y[agent] = log[agent]['reward']['reward'][-1]
    
    agents = list(score_x.keys())
    plt.figure()
    plt.plot([500, 5500], [500, 5500], '--k')
    plt.scatter([score_x[agent] for agent in agents], [score_y[agent] for agent in agents])
    plt.xlabel(x)
    plt.ylabel(y)
    x = x.replace('/', '_')
    y = y.replace('/', '_')
    plt.savefig(f'figures/compare_train_score/{x}-vs-{y}.png')
    plt.close()


def analyze_ratio_hist_per_epoch(folder, prefix=None, seed=1409):

    for i in range(0, 1200, 50):
        plt.figure(figsize=(16, 9))
        if f'ratio_hist_{i}.pkl' not in os.listdir(f'output/{folder}/{seed}/ratio_hist'):
            return
        with open(f'output/{folder}/{seed}/ratio_hist/ratio_hist_{i}.pkl', 'rb') as f:
            hist = pickle.load(f)
        for epoch in range(8):
            plt.subplot(2, 4, epoch + 1)
            if len(hist[epoch]) == 0:
                break
            batch_ratio = hist[epoch]
            for j, batch in enumerate(batch_ratio):
                plt.plot(np.linspace(0., 2., 100), batch, c='b', alpha=0.25 + epoch * 0.05)
                break
            plt.plot([1., 1.], [0, 500], '--k')
        plt.legend()
        if prefix is None:
            plt.savefig(f'figures/ratio_hist/epoch_viz_{i}.png')
        else:
            plt.savefig(f'figures/ratio_hist/epoch_viz_{prefix}_{i}.png')
        plt.close()



if __name__ == '__main__':

    get_context()
    
    # context_features = [
    #     "body_pos", "body_ipos", "body_iquat", "geom_quat", # limb model
    #     "body_mass", "body_shape", # limb hardware
    #     "jnt_pos", # joint model
    #     "joint_range", "joint_axis", "gear" # joint hardware
    # ]

    # for feature in context_features:
    #     plot_context_hist(feature)
    
    # folders = [
    #     'log_HN_fix_normalization_wo_PE/1409', 
    #     'log_baseline_wo_PE/1410', 
    #     'log_origin', 
    # ]
    # plot_grad_norm(folders, key='approx_kl')

    # results = [
    #     'log_HN_fix_normalization_wo_PE_1409', 
    #     'log_HN_fix_normalization_wo_PE_1410', 
    #     'log_baseline_wo_PE_norm_twice_1409', 
    #     'log_baseline_wo_PE_norm_twice_1410', 
    #     'log_baseline_1409', 
    #     'log_baseline_1410', 
    #     'log_HN_fix_normalization_PE_in_base_1409', 
    #     'log_HN_fix_normalization_PE_in_base_1410', 
    # ]
    # plot_test_performance(results)

    # folders = [
    #     'hist_ratio_baseline', 
    #     'hist_ratio_wo_dropout', 
    # ]
    # folders = [
    #     'ST_100M_floor-5506-11-8-01-12-33-50', 
    #     'ST_100M_wo_dropout_floor-5506-11-8-01-12-33-50', 
    # ]
    # agents = os.listdir('output/log_single_task_wo_dropout')
    # plot_training_stats(folders, key='ratio', suffix=None)

    # folders = [
    #     'csr_baseline', 
    #     'csr_baseline_wo_dropout', 
    #     # 'csr_fixed_attention_wo_PE+dropout', 
    #     # 'csr_fix_attention_MLP_wo_PE+dropout', 
    #     'csr_baseline_KL_5_wo_PE+dropout', 
    #     'csr_fix_attention_MLP_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'baseline + dropout', 
    #     'baseline', 
    #     'baseline + KL=0.05', 
    #     'baseline + KL=0.05 + fix attention', 
    # ]
    # kl = [0.2, 0.2, 0.05, 0.05]
    # prefix = 'csr'
    folders = [
        'ft_fix_attention_MLP_KL_5_wo_PE+dropout', 
        'ft_HN_fix_attention_MLP_KL_5_wo_PE+dropout', 
        'ft_HN_fix_attention_MLP_wo_PE+dropout', 
        'ft_baseline_KL_5_wo_PE+dropout', 
        'ft_baseline_dropout_wo_PE', 
    ]
    names = [
        'baseline', 
        'baseline wo dropout', 
        'fixed attention, dropout, wo PE', 
        'fixed attention, dropout both, wo PE', 
        'KL threshold = 0.05, wo dropout', 
        'fixed attention, KL threshold = 0.05, wo dropout', 
    ]
    names = folders
    kl = [0.05, 0.05, 0.2, 0.05, 0.2]
    prefix = 'ft'

    # folders = [
    #     'incline_baseline', 
    #     'incline_baseline_KL_5_wo_dropout', 
    #     'incline_MLP_fix_attention_KL_5_wo_dropout', 
    #     'incline_baseline_wo_dropout', 
    #     'incline_MLP_fix_attention_wo_dropout', 
    # ]
    # names = folders
    # kl = [0.2, 0.05, 0.05, 0.2, 0.2]
    # prefix = 'incline'

    # folders = [
    #     'exploration_baseline', 
    #     'exploration_baseline_KL_5_wo_dropout', 
    #     # 'exploration_baseline_wo_dropout', 
    #     'exploration_MLP_fix_attention_KL_5_wo_dropout', 
    #     # 'exploration_MLP_fix_attention_wo_dropout', 
    # ]
    # names = folders
    # kl = [0.2, 0.05, 0.05]
    # prefix = 'exploration'

    for stat in ['approx_kl', 'grad_norm', 'pi_loss', 'ratio', 'val_loss', 'clip_frac']:
    # for stat in ['clip_frac']:
        plot_training_stats(folders, names, key=stat, prefix=prefix, kl_threshold=kl)
        # agents = [
        #     'floor-1409-11-14-01-15-44-14', 
        #     'floor-5506-11-8-01-12-33-50', 
        #     'mvt-5506-12-4-17-12-01-27', 
        # ]
        # for agent in agents:
        #     # folders = [
        #     #     f'log_single_task_wo_dropout/{agent}', 
        #     #     f'log_single_task/{agent}', 
        #     #     f'log_single_task_wo_pe+dropout/{agent}', 
        #     # ]
        #     folders = [
        #         f'ST_100M_{agent}', 
        #         f'ST_100M_wo_dropout_{agent}', 
        #     ]
        #     names = [
        #         'ST baseline', 
        #         'ST wo dropout', 
        #     ]
        #     plot_training_stats(folders, names=names, key=stat, prefix=f'ST_100M_{agent}')
            # analyze_ratio_hist_trend(folders, prefix=agent)
    # scatter_train_performance(folders)
    # scatter_train_performance_compared_to_subset_train('log_baseline', 'ST-subset')
    # get_context()

    folders = [
        'hist_ratio_baseline', 
        'hist_ratio_wo_dropout', 
    ]
    # folders = [folders[0], folders[1]]
    # analyze_ratio_hist_trend(folders)
    # test_init_ratio_hist(folders)

    score_files = [
        'log_baseline_1409_terminate_on_fall', 
        'log_baseline_1409_deterministic', 
        'log_baseline_1409_terminate_on_fall_deterministic', 
        'log_baseline_1409', 
    ]
    fig_name = 'compare_mode_baseline_1409'

    score_files = [
        'log_baseline_1410_terminate_on_fall', 
        'log_baseline_1410_deterministic', 
        'log_baseline_1410_terminate_on_fall_deterministic', 
        'log_baseline_1410', 
    ]
    fig_name = 'compare_mode_baseline_1410'

    score_files = [
        'log_fix_attention_wo_PE_1409_terminate_on_fall', 
        'log_fix_attention_wo_PE_1409_deterministic', 
        'log_fix_attention_wo_PE_1409_terminate_on_fall_deterministic', 
        'log_fix_attention_wo_PE_1409', 
    ]
    fig_name = 'compare_mode_fix_attention_wo_PE_1409'

    # analyze_ratio_hist_per_epoch('ST_100M_wo_dropout_floor-5506-11-8-01-12-33-50')

    # score_files = [
    #     'log_HN_fix_attention_1409_terminate_on_fall', 
    #     'log_HN_fix_attention_1409_deterministic', 
    #     'log_HN_fix_attention_1409_terminate_on_fall_deterministic', 
    #     'log_HN_fix_attention_1409', 
    # ]
    # fig_name = 'compare_mode_HN_fix_attention_1409'

    # plot_test_score(score_files, fig_name)
    # scores = []
    # for x in score_files:
    #     with open(f'output/eval/{x}.pkl', 'rb') as f:
    #         scores.append(pickle.load(f))
    # scatter_test_score(scores[0], scores[1], score_files[0], score_files[1])
    # scatter_test_score('log_baseline_wo_PE+dropout_1409', 'log_baseline_wo_PE+dropout_1410')
    # scatter_train_score('log_fix_attention_wo_PE/1409', 'log_fix_attention_wo_PE/1410')