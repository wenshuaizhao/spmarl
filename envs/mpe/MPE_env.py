from .environment import MultiAgentEnv
from .scenarios import load
import numpy as np

def MPEEnv(args):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario from script
    scenario = load(args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world,
                        scenario.reward, scenario.observation, scenario.info)

    return env

class ContextualMPEEnv(MultiAgentEnv):
    def __init__(self, args):
        self.args=args
        self.scenario=load(self.args.scenario_name + ".py").Scenario()
        self.world=self.scenario.make_world(self.args)
        self.env=MultiAgentEnv(args, self.world, self.scenario.reset_world,
                        self.scenario.reward, self.scenario.observation, self.scenario.share_observation, self.scenario.info)
        # configure spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.share_observation_space = self.env.share_observation_space
        
        self.max_agents=args.max_num_agents
        # self.max_obs=
    
    def seed(self, seed=None):
        self.env.seed(seed=seed)
    
    def step(self, action_n):
        obs_n, share_obs_n, full_reward_n, done_n, info_n =self.env.step(action_n)
        s_reward=full_reward_n[info_n.squeeze()==True].mean()
        s_done=done_n[info_n.squeeze()==True].mean()
        return obs_n, share_obs_n, full_reward_n, done_n, info_n, s_reward, s_done
    
    def reset(self):
        return self.env.reset()
        
    def render(self, mode='human', close=False):
        return self.env.render(mode=mode, close=close)
    
    def pad(self, data):
        # the data should be in form of list.
        # each element is the observation of each agent
        return np.pad(data)
        
    
    def update_context(self, context=None):
        if context is not None:
            self.args.num_agents=int(context)
            self.scenario=load(self.args.scenario_name + ".py").Scenario()
            self.world=self.scenario.make_world(self.args)
            self.env.close()
            self.env=MultiAgentEnv(self.args, self.world, self.scenario.reset_world,
                            self.scenario.reward, self.scenario.observation, self.scenario.share_observation, self.scenario.info)
            # self.action_space = self.env.action_space
            # self.observation_space = self.env.observation_space
            # self.share_observation_space = self.env.share_observation_space
        else:
            raise RuntimeError("Context should not be None.")
            