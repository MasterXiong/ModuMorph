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


def plot_grad_norm(folders, key='grad_norm', batch_size=8*16):
    plt.figure()
    for folder in folders:
        with open('./output/' + folder + '/Unimal-v0_results.json', 'r') as f:
            log = json.load(f)
        data = np.array(log['__env__'][key])
        
        iter_avg = []
        for i in range(0, len(data), batch_size):
            avg = data[i:(i + batch_size)].mean()
            iter_avg.append(avg)

        plt.plot(iter_avg, label=folder)
    plt.legend()
    plt.savefig(f'figures/{key}.png')
    plt.close()



if __name__ == '__main__':
    
    # context_features = [
    #     "body_pos", "body_ipos", "body_iquat", "geom_quat", # limb model
    #     "body_mass", "body_shape", # limb hardware
    #     "jnt_pos", # joint model
    #     "joint_range", "joint_axis", "gear" # joint hardware
    # ]

    # for feature in context_features:
    #     plot_context_hist(feature)
    
    folders = [
        'test_HN_context_constant_norm_wo_es', 
        'log_origin', 
    ]
    plot_grad_norm(folders)
    # plot_grad_norm(['log_origin'], 'approx_kl', 1)