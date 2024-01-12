from .StarCraft2v2.wrapper import StarCraftCapabilityEnvWrapper # original smac v2 environment
import random
from gym.spaces import Discrete
import numpy as np

class SMACv2(StarCraftCapabilityEnvWrapper):
    def __init__(self, **kwargs):
        super(SMACv2, self).__init__(obs_last_action=False, **kwargs)
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        
        self.max_agents = self.env.n_agents

        for i in range(self.env.n_agents):
            self.action_space.append(Discrete(self.env.n_actions))
            self.observation_space.append([self.env.get_obs_size()])
            self.share_observation_space.append([self.env.get_state_size()])

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        obs, state = super().reset()

        obs_pad = np.zeros((obs.shape[0], self.observation_space[0][0]-obs.shape[1]))
        obs = np.concatenate([obs, obs_pad], -1)
        state_pad = np.zeros((self.share_observation_space[0][0]-state.shape[0]))
        state = np.concatenate([state, state_pad], -1)

        # obs = np.concatenate([obs, np.stack([np.zeros_like(obs[0]) for i in range(self.max_agents-self.env.n_agents)])], 0)
        state = [state for i in range(self.env.n_agents)]
        # state = np.concatenate([state, np.stack([np.zeros_like(state[0]) for i in range(self.max_agents-self.env.n_agents)])], 0)

        avail_actions = [self.get_avail_agent_actions(i) for i in range(self.env.n_agents)]
        # avail_actions.extend([[0]*self.action_space[0].n for i in range(self.max_agents-self.env.n_agents)])
        active_masks = np.ones((self.max_agents, 1), dtype=np.float32)
        # active_masks[self.env.n_agents:] = np.zeros(((self.max_agents-self.env.n_agents), 1), dtype=np.float32)

        if self.max_agents-self.env.n_agents > 0:
            obs = np.concatenate([obs, np.stack([np.zeros_like(obs[0]) for i in range(self.max_agents-self.env.n_agents)])], 0)
            state = np.concatenate([state, np.stack([np.zeros_like(state[0]) for i in range(self.max_agents-self.env.n_agents)])], 0)
            avail_actions.extend([[0]*self.action_space[0].n for i in range(self.max_agents-self.env.n_agents)])
            active_masks[self.env.n_agents:] = np.zeros(((self.max_agents-self.env.n_agents), 1), dtype=np.float32)

        return obs, state, avail_actions, active_masks

    def step(self, actions):
        reward, terminated, info = super().step(actions)
        local_obs = self.get_obs()
        obs_pad = np.zeros((local_obs.shape[0], self.observation_space[0][0]-local_obs.shape[1]))
        local_obs = np.concatenate([local_obs, obs_pad], -1)
        # local_obs = np.concatenate([local_obs, np.stack([np.zeros_like(local_obs[0]) for i in range(self.max_agents-self.env.n_agents)])], 0)

        state = self.get_state()
        state_pad = np.zeros((self.share_observation_space[0][0]-state.shape[0]))
        state = np.concatenate([state, state_pad], -1)
        global_state = [state] * self.env.n_agents
        # global_state = np.concatenate([global_state, np.stack([np.zeros_like(global_state[0]) for i in range(self.max_agents-self.env.n_agents)])], 0)

        rewards = [[reward]] * self.env.n_agents
        # rewards.extend([[0] for i in range(self.max_agents-self.env.n_agents)])
        # dones = [terminated] * self.env.n_agents
        dones = []
        for i in range(self.env.n_agents):
            if terminated:
                dones.append(True)
            else:
                dones.append(self.env.death_tracker_ally[i])
        # dones.extend([True for i in range(self.max_agents-self.env.n_agents)])
        infos = [info] * self.env.n_agents
        # infos.extend([info for i in range(self.max_agents-self.env.n_agents)])
        avail_actions = [self.get_avail_agent_actions(i) for i in range(self.env.n_agents)]
        # avail_actions.extend([[0]*self.action_space[0].n for i in range(self.max_agents-self.env.n_agents)])
        active_masks = np.ones((self.max_agents, 1), dtype=np.float32)
        # active_masks[self.env.n_agents:] = np.zeros(((self.max_agents-self.env.n_agents), 1), dtype=np.float32)

        if self.max_agents-self.env.n_agents > 0:
            local_obs = np.concatenate([local_obs, np.stack([np.zeros_like(local_obs[0]) for i in range(self.max_agents-self.env.n_agents)])], 0)
            global_state = np.concatenate([global_state, np.stack([np.zeros_like(global_state[0]) for i in range(self.max_agents-self.env.n_agents)])], 0)
            rewards.extend([[0] for i in range(self.max_agents-self.env.n_agents)])
            dones.extend([True for i in range(self.max_agents-self.env.n_agents)])
            infos.extend([info for i in range(self.max_agents-self.env.n_agents)])
            avail_actions.extend([[0]*self.action_space[0].n for i in range(self.max_agents-self.env.n_agents)])
            active_masks[self.env.n_agents:] = np.zeros(((self.max_agents-self.env.n_agents), 1), dtype=np.float32)
        
        bad_transition = True if self.env._episode_steps >= self.env.episode_limit else False
        for info in infos:
            info['bad_transition'] = bad_transition
            info["battles_won"] = self.env.battles_won
            info["battles_game"] = self.env.battles_game
            info["battles_draw"] = self.env.timeouts
            info["restarts"] = self.env.force_restarts
            info["won"] = self.env.win_counted
        return local_obs, global_state, rewards, dones, infos, avail_actions, active_masks
