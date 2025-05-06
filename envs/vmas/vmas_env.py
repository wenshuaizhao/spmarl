import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import torch
from .vmas.simulator.scenario import BaseScenario
from typing import Union
from .vmas import make_env
from .vmas.simulator.core import Agent
import matplotlib.pyplot as plt
import time
import torch

class ContextualVMASEnv(gym.Env):
    def __init__(self, args):
        self.args = args
        self.max_steps = args.max_steps
        self.num_agents = args.num_agents
        self.sparse_reward = args.sparse_reward
        self.env = make_env(args.scenario_name, num_envs=1,
                            n_agents=args.num_agents, continuous_actions=args.continuous_actions, max_steps=self.max_steps, sparse_reward=self.sparse_reward)
        # configure spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.share_observation_space = self.env.observation_space

        self.max_agents = args.max_num_agents
        self.step_count = 0
        self.obs_dim = self.env.observation_space[0].shape[0]
        self.share_obs_dim = self.obs_dim
        self.full_obs = np.zeros([self.args.max_num_agents, self.obs_dim])
        self.full_share_obs = np.zeros(
            [self.args.max_num_agents, self.share_obs_dim])
        self.full_rewards = np.zeros([self.args.max_num_agents, 1])
        self.full_dones = np.zeros([self.args.max_num_agents, 1])
        self.full_infos = np.zeros([self.args.max_num_agents, 1])
        self.full_masks = np.zeros([self.args.max_num_agents, 1])
        # self.max_obs=

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
        self.env.seed(seed=seed)

    def step(self, action_n):
        self.full_obs = np.zeros([self.args.max_num_agents, self.obs_dim])
        self.full_share_obs = np.zeros(
            [self.args.max_num_agents, self.share_obs_dim])
        self.full_rewards = np.zeros([self.args.max_num_agents, 1])
        self.full_dones = np.zeros([self.args.max_num_agents, 1])
        self.full_infos = np.zeros([self.args.max_num_agents, 1])
        self.full_masks = np.zeros([self.args.max_num_agents, 1])
        
        actions_to_env = [torch.tensor([[np.argmax(row)]]) for row in action_n]
        obs, rews, dones, infos = self.env.step(actions_to_env)
        self.full_obs[:len(obs), :] = np.array([o.numpy().squeeze() for o in obs])
        self.full_rewards[:len(rews), :] = np.array([o.numpy().squeeze() for o in rews]).reshape(-1, 1)
        self.full_dones[:len(rews), :] = np.array(
            [dones.numpy().squeeze() for _ in range(len(rews))]).reshape(-1, 1)
        self.full_infos[:self.num_agents, :] = 1.
        self.full_masks[:self.num_agents, :] = 1.
        s_reward = rews[0].item()
        s_done = dones.item()
        self.step_count += 1
        return self.full_obs, self.full_share_obs, self.full_rewards, self.full_dones, self.full_infos, s_reward, s_done

    def reset(self):
        obs = self.env.reset()
        self.step_count = 0
        assert self.num_agents == self.env.n_agents
        info_n = np.zeros([self.args.max_num_agents, 1])
        obs_n = np.zeros([self.args.max_num_agents, self.obs_dim])
        share_obs_n = self.full_share_obs.copy()
        info_n[:self.num_agents, :] = 1.
        obs_n[:self.num_agents, :] = np.array([o.numpy().squeeze() for o in obs])
        share_obs_n[:self.num_agents, :] = np.array([o.numpy().squeeze() for o in obs])
        # print('In reset, the number of agents is {}'.format(self.num_agents))
        return obs_n, share_obs_n, info_n

    def render(self, mode='human', close=False):
        return self.env.render(mode=mode, close=close)

    def update_context(self, context=None):
        if context is not None:
            self.num_agents = int(context)
            self.args.num_agents = int(context)
            self.env = make_env(self.args.scenario_name, num_envs=1,
                                n_agents=self.num_agents, continuous_actions=self.args.continuous_actions, max_steps=self.max_steps, sparse_reward=self.sparse_reward)
            assert self.num_agents == self.env.n_agents
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
            self.share_observation_space = self.env.observation_space
            # print('In update context, the number of agents is {}'.format(self.num_agents))
        else:
            raise RuntimeError("Context should not be None.")
