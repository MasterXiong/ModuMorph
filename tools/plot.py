import json
import pickle
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict


def run():
    # useful_agents = os.listdir('output/log_single_task_wo_pe')
    all_agents = os.listdir('output/log_single_task')
    for agent in all_agents:
        # if agent not in useful_agents:
            # os.system(f'rm -r output/log_single_task/{agent}')
        os.system(f'rm output/log_single_task/{agent}/*.pt')
        os.system(f'rm output/log_single_task/{agent}/*.yaml')
        os.system(f'rm output/log_single_task/{agent}/*.json')
        os.system(f'rm -r output/log_single_task/{agent}/tensorboard')


def compare_ST_training(folders, agents):

    if folders is None:
        folders = [
            'log_single_task', 
            'log_single_task_wo_pe+dropout', 
            'log_single_task_wo_dropout', 
        ]

    all_curves = {}
    for folder in folders:
        train_curve = defaultdict(list)
        if agents is None:
            agents = os.listdir(f'output/{folder}')
        for agent in agents:
            if agent in folder:
                for seed in os.listdir(f'output/{folder}'):
                    path = f'output/{folder}/{seed}'
                    if 'checkpoint_100.pt' in os.listdir(path):
                        with open(f'{path}/Unimal-v0_results.json', 'r') as f:
                            log = json.load(f)
                        train_curve[agent].append(log[agent]['reward']['reward'])                
            else:
                for seed in os.listdir(f'output/{folder}/{agent}'):
                    path = f'output/{folder}/{agent}/{seed}'
                    if 'checkpoint_100.pt' in os.listdir(path):
                        with open(f'{path}/Unimal-v0_results.json', 'r') as f:
                            log = json.load(f)
                        train_curve[agent].append(log[agent]['reward']['reward'])
        all_curves[folder] = train_curve

    for agent in all_curves[folders[0]]:
        all_include = True
        for folder in folders[1:]:
            if agent not in all_curves[folder]:
                all_include = False
                break
        if not all_include:
            continue
            
        plt.figure()
        for folder in folders:
            c = all_curves[folder][agent]
            length = min([len(x) for x in c])
            all_seeds = np.stack([np.array(x)[:length] for x in c])
            avg, std = all_seeds.mean(0), all_seeds.std(0)
            plt.plot([i*2560*32 for i in range(len(avg))], avg, label=f'{folder} ({all_seeds.shape[0]} seeds)')
            plt.fill_between([i*2560*32 for i in range(len(avg))], avg - std, avg + std, alpha=0.25)
        plt.legend()
        plt.xlabel('Time step')
        plt.ylabel('Returns')
        plt.title(agent)
        plt.savefig(f'figures/ST_results/ST_train_{agent}.png')
        plt.close()
        

def compare_train_curve():
    folders = [
        'log_baseline/1409', 
        'log_baseline/1410', 
        #    'log_train_wo_pe/1411', 
        # 'log_baseline_wo_PE/1409', 
        # 'log_baseline_wo_PE/1410', 
        # 'log_HN_context_double_norm_wo_PE/1409', 
        # 'log_HN_context_double_norm_wo_PE/1410', 
        # 'log_baseline_wo_PE_context_norm_twice/1409', 
        # 'log_baseline_wo_PE_context_norm_twice/1410', 
        # # 'test_HN_with_context_normlization_wo_embed_scale_wo_es', 
        # 'test_HN_with_context_normalization_wo_embed_scale', 
        # # 'test_HN_with_context_normalization_wo_embed_scale_kl_1.', 
        # # 'test_HN_context_constant_norm_kl_1.', 
        # # 'test_HN_context_constant_norm_wo_es',
        # # 'log_HN_fix_normalization_wo_PE/1409', 
        # # 'log_HN_fix_normalization_wo_PE/1410', 
        # # 'log_HN_fix_normalization_wo_PE/1411', 
        # # # 'test_HN_PE_as_context_es_1', 
        # # # 'log_baseline_agrim_NM_feature/1410',  
        # 'log_baseline_wo_PE/1410',  
        # 'log_baseline_wo_PE/1409',
        # 'log_baseline_wo_PE_wo_context/1410',  
        # 'log_baseline_wo_PE_wo_context/1409',  
        # 'log_baseline_wo_PE_context_fixed_norm/1410', 
        # 'log_HN_fix_normalization_PE_in_base/1409', 
        # 'log_HN_fix_normalization_PE_in_base/1410', 
        # 'log_HN_fix_normalization_PE_in_HN/1409', 
        # 'log_HN_fix_normalization_PE_in_HN/1410', 
        # 'log_baseline_200M/1409', 
        # 'log_baseline_200M/1410', 
        # 'log_HN_PE_in_base_200M/1409', 
        # 'log_HN_PE_in_base_200M/1410', 
        'log_baseline_wo_dropout/1409', 
        'log_baseline_wo_dropout/1410', 
        # 'test', 
    ]

    # effect of using PE dropout in baseline
    # removing dropout leads to worse performance. But why?
    # folders = [
    #     'log_baseline/1409', 
    #     'log_baseline/1410', 
    #     'log_baseline_wo_dropout/1409', 
    #     'log_baseline_wo_dropout/1410', 
    # ]

    # training for 200M steps
    folders = [
        'log_baseline_200M/1409', 
        'log_baseline_200M/1410', 
        'log_HN_PE_in_base_200M/1409', 
        'log_HN_PE_in_base_200M/1410', 
    ]

    # use HN for both embedding and decoding layers
    # folders = [
    #     # 'log_baseline/1409', 
    #     # 'log_baseline/1410', 
    #     # 'log_HN_fix_normalization_PE_in_base/1409', 
    #     # 'log_HN_fix_normalization_PE_in_base/1410', 
    #     'log_baseline_wo_PE/1409', 
    #     'log_baseline_wo_PE/1410', 
    #     'log_HN_2_layer_wo_PE/1409', 
    #     'log_HN_2_layer_wo_PE/1410', 
    #     'log_HN_embed_wo_PE/1409', 
    #     'log_HN_embed_wo_PE/1410', 
    # ]

    # test whether PE or dropout leads to better performance
    folders = [
        'log_baseline/1409', 
        'log_baseline/1410', 
        'log_baseline_dropout_wo_PE/1409', 
        'log_baseline_dropout_wo_PE/1410', 
        'log_baseline_wo_dropout/1409', 
        'log_baseline_wo_dropout/1410', 
        'log_fix_attention_wo_PE/1409', 
        'log_fix_attention_wo_PE/1410', 
    ]

    # plt.figure()
    # for folder in folders:
    #     with open(f'output/{folder}/Unimal-v0_results.json', 'r') as f:
    #         train_results = json.load(f)
    #     plt.plot(train_results['__env__']['reward']['reward'], label=folder)
    #     print (train_results['__env__']['reward']['reward'][-1])
    # plt.legend()
    # plt.savefig('figures/train_curve.png')
    # plt.close()

    folders = [
        'log_baseline', 
        'log_baseline_wo_PE', 
        'log_HN_fix_normalization_wo_PE', 
        'log_HN_fix_normalization_PE_in_base', 
        'log_HN_fix_normalization_PE_in_HN', 
        'log_HN_2_layer_wo_PE', 
        'log_HN_embed_wo_PE', 
    ]
    names = [
        'baseline', 
        'baseline wo PE', 
        'HN for decoder, wo PE', 
        'HN with PE in base', 
        'HN with PE in HN', 
        'HN for embed and decoder, wo PE', 
        'HN for embed, wo PE', 
    ]
    index = [1,2,5,6]

    folders = [
        'log_baseline_200M', 
        'log_HN_PE_in_base_200M', 
        'log_baseline', 
        'log_HN_PE_in_base', 
    ]
    names = [
        'baseline (200M)', 
        'HN (200M)', 
        'baseline (100M)', 
        'HN (100M)', 
    ]
    index = [2,3]

    # folders = [
    #     'log_baseline', 
    #     'log_baseline_wo_PE', 
    #     'log_baseline_dropout_wo_PE', 
    #     'log_baseline_wo_dropout', 
    #     'log_fix_attention_wo_PE', 
    #     'log_fix_attention_with_PE', 
    #     'log_fix_attention_with_dropout', 
    #     'log_fix_attention_no_context_in_base_wo_PE', 
    #     'log_baseline_wo_PE_aligned_dropout', 
    #     'log_HN_fix_attention', 
    #     'ft_fix_attn_morphology_attn_wo_PE+dropout', 
    #     'log_baseline_wo_context', 
    #     'log_baseline_wo_PE_wo_context', 
    # ]
    # names = [
    #     'baseline', 
    #     'baseline wo PE', 
    #     'baseline wo PE + dropout', 
    #     'baseline - dropout', 
    #     'baseline wo PE + fix attention', 
    #     'baseline + fix attention', 
    #     'baseline wo PE + fix attention + dropout', 
    #     'baseline wo PE + fix attention + no context in base', 
    #     'baseline wo PE + aligned dropout', 
    #     'baseline wo PE + fix attention + HN', 
    #     'baseline wo PE + fix attention + morphology info', 
    #     'baseline wo context', 
    #     'baseline wo context+PE+dropout', 
    # ]
    # index = [4,10]
    # index = [0, -2, -1]

    folders = [
        'csr_baseline', 
        # 'csr_baseline_wo_dropout', 
        # 'csr_fixed_attention_wo_PE+dropout', 
        # 'csr_fix_attention_MLP_wo_PE+dropout', 
        'csr_fix_attention_MLP_KL_5_wo_PE+dropout', 
        'csr_fix_attention_MLP_KL_3_wo_PE+dropout', 
        # 'csr_baseline_wo_PE', 
        # 'csr_baseline_KL_7_wo_dropout', 
        'csr_baseline_KL_5_wo_PE+dropout', 
        'csr_baseline_KL_3_wo_dropout', 
        # 'csr_fix_attention_MLP_KL_3_wo_PE+dropout', 
    ]
    names = folders
    names = ['MetaMorph', 'fix attention, KL=0.05', 'fix attention, KL=0.03', 'KL=0.05', 'KL=0.03']
    suffix = 'csr_fix_attention'

    # folders = [
    #     # 'log_fix_attention_wo_PE', 
    #     # 'ft_fix_attention_MLP_context_encoder_wo_PE+dropout', 
    #     # 'ft_fix_attention_MLP_context_encoder_node_depth_wo_PE+dropout', 
    #     # 'ft_fix_attention_context_layer_1_wo_PE+dropout', 
    #     # 'ft_fix_attention_MLP_1_layer_wo_PE+dropout', 
    #     'ft_fix_attention_MLP_wo_PE+dropout', 
    #     # 'ft_fix_attention_MLP_dropout_wo_PE', 
    #     # 'ft_fix_attention_MLP_dropout_both_wo_PE', 
    #     'ft_fix_attention_MLP_KL_3_wo_PE+dropout', 
    #     'ft_fix_attention_MLP_KL_5_wo_PE+dropout', 
    #     'ft_fix_attention_MLP_KL_10_wo_PE+dropout', 
    #     # 'ft_HN_fix_attention_MLP_wo_PE+dropout', 
    #     # 'ft_HN_fix_attention_MLP_KL_5_wo_PE+dropout', 
    #     # 'ft_baseline_KL_5_wo_PE+dropout', 
    #     # 'ft_baseline_dropout_wo_PE', 
    #     # 'ft_SWAT_PE_KL_5_wo_PE+dropout', 
    #     # 'ft_SWAT_PE_wo_PE', 
    #     # 'ft_separate_PE_KL_5_wo_PE+dropout', 
    #     # 'ft_separate_PE_wo_PE', 
    #     # 'ft_baseline_wo_context', 
    #     # 'ft_context_PE_KL_5_wo_PE+dropout', 
    #     # 'ft_HN_depth_input_fix_attention_MLP_KL_5_wo_PE+dropout', 
    #     # 'ft_context_PE_tree_PE_KL_5_wo_PE+dropout', 
    #     # 'ft_context_PE_graph_PE_KL_5_wo_PE+dropout', 
    #     # 'ft_context_PE_tree+graph_PE_KL_5_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = 'ft'

    folders = [
        'ft_baseline', 
        ''
    ]

    # folders = [
    #     'incline_baseline', 
    #     'incline_baseline_KL_5_wo_dropout', 
    #     # 'incline_baseline_wo_dropout', 
    #     'incline_MLP_fix_attention_KL_5_wo_dropout', 
    #     # 'incline_MLP_fix_attention_wo_dropout', 
    # ]
    # names = folders
    # suffix = 'incline'

    # folders = [
    #     'exploration_baseline', 
    #     'exploration_baseline_KL_3_wo_dropout', 
    #     # 'exploration_baseline_KL_5_wo_dropout', 
    #     # 'exploration_baseline_KL_10_wo_dropout', 
    #     # 'exploration_baseline_wo_dropout', 
    #     'exploration_MLP_fix_attention_KL_3_wo_dropout', 
    #     # 'exploration_MLP_fix_attention_wo_dropout', 
    # ]
    # # names = folders
    # # names = ['MetaMorph', 'KL=0.03', 'KL=0.05', 'KL=0.1', 'KL=0.2']
    # names = ['MetaMorph', 'KL=0.03', 'fix attention, KL=0.03']
    # suffix = 'exploration_fix_attention'

    # folders = [
    #     'obstacle_baseline', 
    #     'obstacle_baseline_KL_3_wo_dropout', 
    #     'obstacle_baseline_KL_5_wo_dropout', 
    #     # 'obstacle_baseline_KL_10_wo_dropout', 
    #     # 'obstacle_baseline_wo_dropout', 
    #     'obstacle_MLP_fix_attention_KL_3_wo_dropout', 
    #     'obstacle_MLP_fix_attention_KL_5_wo_dropout', 
    # ]
    # # names = folders
    # # names = ['MetaMorph', 'KL=0.03', 'KL=0.05', 'KL=0.1', 'KL=0.2']
    # names = ['MetaMorph', 'KL=0.03', 'KL=0.05', 'fix attention, KL=0.03', 'fix attention, KL=0.05']
    # suffix = 'obstacle_fix_attention'

    # folders = [
    #     'escape_bowl_baseline', 
    #     'escape_bowl_baseline_KL_5_wo_dropout', 
    #     'escape_bowl_MLP_fix_attention_KL_5_wo_dropout', 
    # ]
    # names = folders
    # suffix = 'escape'

    # folders = [
    #     'hist_ratio_baseline', 
    #     'hist_ratio_wo_dropout', 
    # ]
    # names = folders
    # index = [0,1]

    # folders = [
    #     'ST_100M_floor-1409-11-14-01-15-44-14', 
    #     'ST_100M_floor-5506-11-8-01-12-33-50', 
    #     'ST_100M_mvt-5506-12-4-17-12-01-27', 
    #     'ST_100M_wo_dropout_floor-1409-11-14-01-15-44-14', 
    #     'ST_100M_wo_dropout_floor-5506-11-8-01-12-33-50', 
    #     'ST_100M_wo_dropout_mvt-5506-12-4-17-12-01-27', 
    # ]
    # names = folders
    # index = [0,3]

    # dropout and KL ES
    # folders = [
    #     'ft_baseline_dropout_wo_PE', 
    #     'ft_baseline_wo_PE+dropout', 
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'baseline + dropout', 
    #     'baseline', 
    #     'baseline + KL=0.05', 
    # ]

    # folders = [
    #     'csr_baseline', 
    #     'csr_baseline_wo_dropout', 
    #     'csr_baseline_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'csr baseline + dropout', 
    #     'csr baseline', 
    #     'csr baseline + KL=0.05', 
    # ]

    # folders = [
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     'ft_fix_attention_MLP_KL_5_wo_PE+dropout', 
    #     'ft_baseline_dropout_wo_PE', 
    # ]
    # names = [
    #     'baseline + KL=0.05', 
    #     'fixed attention + KL=0.05', 
    #     'baseline + dropout', 
    # ]

    # folders = [
    #     'csr_baseline_KL_5_wo_PE+dropout', 
    #     'csr_fix_attention_MLP_KL_5_wo_PE+dropout', 
    #     'csr_baseline', 
    # ]
    # names = [
    #     'csr baseline + KL=0.05', 
    #     'csr fixed attention + KL=0.05', 
    #     'csr baseline + dropout', 
    # ]

    plt.figure()
    for i in range(len(folders)):
        all_curves = []
        seed_count = 0
        for seed in [1409, 1410, 1411]:
            try:
                with open(f'output/{folders[i]}/{seed}/Unimal-v0_results.json', 'r') as f:
                    train_results = json.load(f)
                return_curve = train_results['__env__']['reward']['reward']
                all_curves.append(return_curve)
                seed_count += 1
            except:
                pass
        print (seed_count)
        l = min([len(x) for x in all_curves])
        avg_curve = np.stack([x[:l] for x in all_curves])
        plt.plot(avg_curve.mean(axis=0), label=f'{names[i]} ({seed_count} seeds)')
        plt.fill_between(np.arange(avg_curve.shape[1]), avg_curve.mean(axis=0) - avg_curve.std(axis=0), avg_curve.mean(axis=0) + avg_curve.std(axis=0), alpha=0.25)
        # for i in range(avg_curve.shape[0]):
        #     plt.plot(all_curves[i], label=names[i], alpha=0.4)
        print (names[i], avg_curve.mean(axis=0)[-1])
    plt.legend(prop = {'size':8})
    plt.xlabel('PPO update number')
    plt.ylabel('return')
    plt.savefig(f'figures/train_curve_{suffix}.png')
    plt.close()

    # path = 'output/log_hypernet_1410/checkpoint_500.pt'
    # m, _ = torch.load(path)
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # w = m.v_net.hnet.weight.data
    # plt.hist(w.cpu().numpy().ravel())
    # plt.subplot(1, 2, 2)
    # w = m.mu_net.hnet.weight.data
    # plt.hist(w.cpu().numpy().ravel())
    # plt.savefig('figures/hypernet_weight.png')
    # plt.close()



if __name__ == '__main__':
    folders = [
        'log_single_task', 
        'log_single_task_wo_pe+dropout', 
        'log_single_task_wo_dropout', 
        'ST_100M_floor-5506-11-8-01-12-33-50', 
        'ST_100M_wo_dropout_floor-5506-11-8-01-12-33-50', 
    ]
    # compare_ST_training(folders, ['floor-5506-11-8-01-12-33-50'])
    compare_train_curve()
    # run()