import os
import random

import gym
import numpy as np
from gym import spaces
from gym import utils

from gym.spaces import Box
from gym.spaces import Dict

from metamorph.config import cfg
from metamorph.envs.modules.agent import create_agent_xml
from metamorph.envs.tasks.unimal import UnimalEnv
from metamorph.utils import file as fu
from metamorph.utils import spaces as spu


class ModularObservationPadding(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.max_limbs = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_JOINTS

        num_limbs = self.metadata["num_limbs"]
        self.num_limb_pads = self.max_limbs - num_limbs

        self.limb_obs_size = self.observation_space.shape[0] // num_limbs

        inf = np.float32(np.inf)
        self.observation_space = dict()
        shape = (self.limb_obs_size * cfg.MODEL.MAX_LIMBS,)
        self.observation_space['proprioceptive'] = Box(-inf, inf, shape, np.float32)
        self.observation_space['context'] = Box(-inf, inf, shape, np.float32)
        self.observation_space['obs_padding_mask'] = Box(-inf, inf, (self.max_limbs, ), np.float32)
        self.observation_space['act_padding_mask'] = Box(-inf, inf, (self.max_limbs, ), np.float32)
        self.observation_space['edges'] = Box(-inf, inf, (self.max_joints * 2, ), np.float32)
        self.observation_space = Dict(self.observation_space)

        obs_padding_mask = [False] * num_limbs + [True] * self.num_limb_pads
        self.obs_padding_mask = np.asarray(obs_padding_mask)

        act_padding_mask = [True] + [False] * (num_limbs - 1) + [True] * self.num_limb_pads
        self.act_padding_mask = np.asarray(act_padding_mask)

    def observation(self, obs):

        padding = [0.0] * (self.limb_obs_size * self.num_limb_pads)
        obs_dict = dict()

        obs_dict["proprioceptive"] = np.concatenate([obs, padding]).ravel()
        obs_dict["context"] = np.concatenate([obs, padding]).ravel()
        obs_dict["obs_padding_mask"] = self.obs_padding_mask
        obs_dict["act_padding_mask"] = self.act_padding_mask
        obs_dict["edges"] = np.zeros(self.max_joints * 2)

        return obs_dict

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)


class ConsistentModularObservationPadding(ModularObservationPadding):

    def __init__(self, env):

        super().__init__(env)

        agent_limb_names = env.agent_limb_names
        if 'all_train' in cfg.ENV.WALKER_DIR:
            full_limb_names = [
                'torso', 'thigh', 'leg', 'lower_leg', 'foot', 
                'left1', 'left2', 'left3', 'right1', 'right2', 'right3', 
                'right_thigh', 'right_shin', 'left_thigh', 'left_shin', 'right_upper_arm', 'right_lower_arm', 'left_upper_arm', 'left_lower_arm', 
            ]
        else:
            full_limb_names = env.full_limb_names
        self.limb_index = [full_limb_names.index(name) for name in agent_limb_names]

        self.obs_padding_mask = np.asarray([True] * self.max_limbs)
        self.obs_padding_mask[self.limb_index] = False

        self.act_padding_mask = np.asarray([True] * self.max_limbs)
        self.act_padding_mask[self.limb_index] = False
        # torso has no action
        self.act_padding_mask[0] = True

    def observation(self, obs):

        padded_obs = np.zeros([self.max_limbs, self.limb_obs_size])
        padded_obs[self.limb_index] = obs.reshape(-1, self.limb_obs_size)
        padded_obs = padded_obs.ravel()

        obs_dict = dict()
        obs_dict["proprioceptive"] = padded_obs
        obs_dict["context"] = padded_obs
        obs_dict["obs_padding_mask"] = self.obs_padding_mask
        obs_dict["act_padding_mask"] = self.act_padding_mask
        obs_dict["edges"] = np.zeros(self.max_joints * 2)

        return obs_dict


class ModularActionPadding(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.max_limbs = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_LIMBS
        self._update_action_space()
        act_padding_mask = [True] + [False] * self.metadata["num_joints"] + [True] * self.num_limb_pads
        self.act_padding_mask = np.asarray(act_padding_mask)

    def _update_action_space(self):
        num_pads = self.max_limbs - self.metadata["num_limbs"]
        low, high = self.action_space.low, self.action_space.high
        low = np.concatenate([[-1.], low, [-1.] * num_pads]).astype(np.float32)
        high = np.concatenate([[-1.], high, [-1.] * num_pads]).astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        new_action = action[~self.act_padding_mask]
        return new_action


class ConsistentModularActionPadding(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.max_limbs = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_LIMBS

        agent_limb_names = env.agent_limb_names
        if 'all_train' in cfg.ENV.WALKER_DIR:
            full_limb_names = [
                'torso', 'thigh', 'leg', 'lower_leg', 'foot', 
                'left1', 'left2', 'left3', 'right1', 'right2', 'right3', 
                'right_thigh', 'right_shin', 'left_thigh', 'left_shin', 'right_upper_arm', 'right_lower_arm', 'left_upper_arm', 'left_lower_arm', 
            ]
        else:
            full_limb_names = env.full_limb_names
        self.joint_index = [full_limb_names.index(name) for name in agent_limb_names]
        # torso has no joint action
        self.joint_index.remove(0)

        self._update_action_space()

        self.act_padding_mask = np.asarray([True] * self.max_limbs)
        self.act_padding_mask[self.joint_index] = False

    def _update_action_space(self):
        low = -1. * np.ones(self.max_limbs, dtype=np.float32)
        high = -1. * np.ones(self.max_limbs, dtype=np.float32)
        high[self.joint_index] = 1.
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        new_action = action[~self.act_padding_mask]
        return new_action