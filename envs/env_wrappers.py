"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
from utils.util import tile_images
from utils.cmarl_util import Buffer, ContextBuffer

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions, activate_masks):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions, activate_masks)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob = env.reset()
            else:
                if np.all(done):
                    ob = env.reset()

            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send((ob))
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError

def cmarl_worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, share_obs, reward, done, info, s_reward, s_done = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob, share_obs, info = env.reset()
            else:
                if np.all(done):
                    ob, share_obs, info = env.reset()

            remote.send((ob, share_obs, reward, done, info, s_reward, s_done))
        elif cmd == 'reset':
            ob, share_obs, info = env.reset()
            remote.send((ob, share_obs, info))
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == 'update_context':
            env.update_context(context=data)
        else:
            raise NotImplementedError

class CMARLSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, teacher=None, spaces=None, eval=False):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.context_buffer = ContextBuffer(dim=1, max_buffer_size=10000)
        
        self.teacher=teacher
        self.target=self.teacher.target
        self.discount_factor=0.98
        self.max_agents=20
        self.eval=eval
        self.teacher_name=self.teacher.teacher_name
        self.total_steps=0
        
        self.undiscounted_rewards = np.zeros(len(env_fns))
        self.discounted_rewards = np.zeros(len(env_fns))
        self.cur_discs = np.ones(len(env_fns))
        self.step_lengths = np.zeros(len(env_fns))
        
        self.cur_contexts = [None] * len(env_fns)
        self.processed_contexts = [None] * len(env_fns)
        self.cur_initial_states = [None] * len(env_fns)
        
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=cmarl_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions, active_masks):
        
        for remote, action, active_mask in zip(self.remotes, actions, active_masks):
            action=action[active_mask.squeeze()==True]
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, s_reward, s_done = zip(*results)
        obs, share_obs, rews, dones, infos = np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones).squeeze(), np.stack(infos)
        contexts=np.repeat(np.stack(self.cur_contexts)[:, None], 30, axis=1)[:,:,None]
        step=(obs, share_obs, rews, dones, infos, s_reward, s_done)
        self.update(step)
        return obs, share_obs, rews, dones, infos, contexts

    def reset(self):
        # contexts = self.teacher.sample(number=len(self.remotes))
        # contexts=np.random.randint(low=8, high=20, size=len(self.ps))
        contexts=self.teacher.sample(size=len(self.ps)).squeeze()
        if self.eval:
            contexts=np.ones_like(contexts)*self.target
        for i in range(len(self.cur_contexts)):
            self.cur_contexts[i]=contexts[i] 
        for remote, context in zip(self.remotes, self.cur_contexts): remote.send(('update_context', context))
        for remote in self.remotes: remote.send(('reset', None))
            
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, info = zip(*results)
        
        return np.stack(obs), np.stack(share_obs), np.stack(info)
    
    def reset_data(self, env_id):
        self.undiscounted_rewards[env_id] = 0.
        self.discounted_rewards[env_id] = 0.
        self.cur_discs[env_id] = 1.
        self.step_lengths[env_id] = 0.

        self.cur_contexts[env_id] = None
        self.cur_initial_states[env_id] = None
    
    def update(self, step):
        obs, share_obs, rews, dones, infos, s_reward, s_done = step
        if not self.eval:
            self.total_steps+=len(self.ps)
            self.undiscounted_rewards += s_reward # only use the reward of one agent
            self.discounted_rewards += self.cur_discs * s_reward
            self.cur_discs *= self.discount_factor
            self.step_lengths += 1.
            if any(s_done):
                index=[i for i, x in enumerate(s_done) if x]
                for i in index:
                    # context=np.random.randint(low=8, high=20, size=1)
                    data = self.cur_contexts[i], self.undiscounted_rewards[i], self.discounted_rewards[i], self.step_lengths[i]
                    self.context_buffer.update_buffer(data)
                    self.reset_data(i)
                    
                    context=self.teacher.sample(size=1).squeeze()
                    
                    self.remotes[i].send(('update_context', context))
                    self.cur_contexts[i]=context
                    self.remotes[i].send(('reset', None))
                    ob, share_ob, info = self.remotes[i].recv()
                    obs[i], share_obs[i], infos[i] = ob, share_ob, info
             
        else:
            if any(s_done):
                index=[i for i, x in enumerate(s_done) if x]
                for i in index:
                    context=self.target
                    self.remotes[i].send(('update_context', context))
                    self.remotes[i].send(('reset', None))
                    ob, share_ob, info = self.remotes[i].recv()
                    obs[i], share_obs[i], infos[i] = ob, share_ob, info
                    
    def update_teacher(self, *args):
        context_info={}
        steps=self.context_buffer.size
        
        if self.teacher_name in ['spmarl', 'sprl']:
            context_info['g_mean']=self.teacher.context_dist.get_weights()[0]
            context_info['g_std']=np.sqrt(self.teacher.context_dist.covariance_matrix().sum())
        if self.teacher_name=='spmarl':
            self.teacher.update_distribution(self.total_steps, self.context_buffer.contexts[:steps], self.context_buffer.returns[:steps], np.stack(args[0])[:, None], np.stack(args[1])[:, None])
        else:
            self.teacher.update_distribution(self.total_steps, self.context_buffer.contexts[:steps], self.context_buffer.returns[:steps])
        
        context_info['sampled_mean']=self.context_buffer.contexts[:steps].mean()
        context_info['sampled_std']=self.context_buffer.contexts[:steps].std()
        self.context_buffer.clear()
        return context_info

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":   
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame) 
        
class CMARLDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns, teacher=None, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions, active_masks):
        
        self.actions=actions[active_masks.squeeze(-1)==True]

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip([self.actions], self.envs)]
        obs, share_obs, rews, dones, infos, s_reward, s_done = zip(*results)
        obs, share_obs, rews, dones, infos = np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), np.stack(infos)
        for (i, done) in enumerate(s_done):
            if done:
                obs[i], share_obs[i], infos[i] = self.envs[i].reset()
            
        self.actions=None
        
        return obs, share_obs, rews, dones, infos

    def reset(self):
            
        results = [env.reset() for env in self.envs]
        obs, share_obs, info = zip(*results)
        
        return np.stack(obs), np.stack(share_obs), np.stack(info)
    
    def reset_data(self, env_id):
        self.undiscounted_rewards[env_id] = 0.
        self.discounted_rewards[env_id] = 0.
        self.cur_discs[env_id] = 1.
        self.step_lengths[env_id] = 0.

        self.cur_contexts[env_id] = None
        self.cur_initial_states[env_id] = None
    
    def update(self, step):
        obs, share_obs, rews, dones, infos, s_reward, s_done = step
        
        self.undiscounted_rewards += s_reward # only use the reward of one agent
        self.discounted_rewards += self.cur_discs * s_reward
        self.cur_discs *= self.discount_factor
        self.step_lengths += 1.
        if any(s_done):
            index=[i for i, x in enumerate(s_done) if x]
            for i in index:
                context=np.random.randint(low=8, high=20, size=1)
                self.remotes[i].send(('update_context', context))
                self.reset_data(i)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError


class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)


    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":   
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame) 


def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(done):
                    ob, s_ob, available_actions = env.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == 'reset':
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == 'render_vulnerability':
            fr = env.render_vulnerability(data)
            remote.send((fr))
        else:
            raise NotImplementedError
        

def shareselfpacedworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, s_ob, reward, done, info, available_actions, active_masks = env.step(data)
            # if 'bool' in done.__class__.__name__:
            #     if done:
            #         ob, s_ob, available_actions, active_masks = env.reset()
            # else:
            #     if np.all(done):
            #         ob, s_ob, available_actions, active_masks = env.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions, active_masks))
        elif cmd == 'reset':
            ob, s_ob, available_actions, active_masks = env.reset()
            remote.send((ob, s_ob, available_actions, active_masks))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == 'render_vulnerability':
            fr = env.render_vulnerability(data)
            remote.send((fr))
        elif cmd == 'update_context':
            env.update_context(context=data)
        else:
            raise NotImplementedError
        
def matrixshareselfpacedworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info, active_masks, avail_actions = env.step(data)
            # if 'bool' in done.__class__.__name__:
            #     if done:
            #         ob, s_ob, available_actions, active_masks = env.reset()
            # else:
            #     if np.all(done):
            #         ob, s_ob, available_actions, active_masks = env.reset()

            remote.send((ob, reward, done, info, active_masks, avail_actions))
        elif cmd == 'reset':
            ob, active_masks, avail_actions = env.reset()
            remote.send((ob, active_masks, avail_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == 'render_vulnerability':
            fr = env.render_vulnerability(data)
            remote.send((fr))
        elif cmd == 'update_context':
            env.update_context(context=data)
        else:
            raise NotImplementedError

def pursuitworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, s_ob, reward, done, info = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob, s_ob = env.reset()
            else:
                if np.all(done):
                    ob, s_ob = env.reset()

            remote.send((ob, s_ob, reward, done, info))
        elif cmd == 'reset':
            ob, s_ob = env.reset()
            remote.send((ob, s_ob))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == 'render_vulnerability':
            fr = env.render_vulnerability(data)
            remote.send((fr))
        else:
            raise NotImplementedError
        
class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class ShareSelfPacedSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, teacher=None, args=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.train = True
        nenvs = len(env_fns)

        self.context_buffer = ContextBuffer(dim=1, max_buffer_size=10000)
        
        self.teacher=teacher
        self.teacher_name=self.teacher.teacher_name
        self.target=self.teacher.target
        
        self.max_agents=args.max_num_agents

        self.total_steps=0
        
        self.undiscounted_rewards = np.zeros(len(env_fns))
        self.discounted_rewards = np.zeros(len(env_fns))
        self.cur_discs = np.ones(len(env_fns))
        self.step_lengths = np.zeros(len(env_fns))
        
        self.cur_contexts = [None] * len(env_fns)
        self.processed_contexts = [None] * len(env_fns)
        self.cur_initial_states = [None] * len(env_fns)

        self.discount_factor = args.gamma
        
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=shareselfpacedworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions, active_masks):
        for remote, action, active_mask in zip(self.remotes, actions, active_masks):
            action=action[active_mask.squeeze()==True]
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions, active_masks = zip(*results)
        obs, share_obs, rews, dones, infos, available_actions, active_masks = self.update((np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions), np.stack(active_masks)))
        # try:
        contexts=np.repeat(np.stack(self.cur_contexts)[:, None], self.max_agents, axis=1)[:,:,None]
        # except:
        #     raise RuntimeError
        return obs, share_obs, rews, dones, infos, available_actions, active_masks, contexts

    def reset(self, train=True):
        self.train = train
        # contexts = self.teacher.sample(number=len(self.remotes))
        # contexts=np.random.randint(low=5, high=20, size=len(self.ps)) if train else np.ones(len(self.ps), dtype=np.int) * 5
        # contexts=np.ones(len(self.ps), dtype=np.int) * 10 if train else np.ones(len(self.ps), dtype=np.int) * self.target_context
        contexts=self.teacher.sample(size=len(self.ps)).squeeze() if train else np.ones(len(self.ps), dtype=np.int) * self.target
        for i in range(len(self.cur_contexts)):
            self.cur_contexts[i]=contexts[i] 
        for remote, context in zip(self.remotes, self.cur_contexts): 
            remote.send(('update_context', context))

        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions, active_masks = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions), np.stack(active_masks)
    
    def reset_data(self, env_id):
        self.undiscounted_rewards[env_id] = 0.
        self.discounted_rewards[env_id] = 0.
        self.cur_discs[env_id] = 1.
        self.step_lengths[env_id] = 0.

        self.cur_contexts[env_id] = None
        self.cur_initial_states[env_id] = None
    
    def update(self, step):
        obs, share_obs, rews, dones, infos, available_actions, active_masks = step
        self.total_steps+=len(self.ps)
        dones_env = np.zeros(len(self.cur_contexts), dtype=np.bool_)
        for i in range(len(self.cur_contexts)):
            self.undiscounted_rewards[i] += rews[i][active_masks[i].squeeze()==True].mean()
            self.discounted_rewards[i] += self.cur_discs[i] * rews[i][active_masks[i].squeeze()==True].mean()
            self.cur_discs[i] *= self.discount_factor
            self.step_lengths[i] += 1.

            dones_env[i] = np.all(dones[i][active_masks[i].squeeze()==True], axis=-1)

        if any(dones_env):
            index=[i for i, x in enumerate(dones_env) if x]
            for i in index:
                # context=np.random.randint(low=5, high=20) if self.train else 5
                # context=10 if self.train else 5
                data = self.cur_contexts[i], self.undiscounted_rewards[i], self.discounted_rewards[i], self.step_lengths[i]
                self.context_buffer.update_buffer(data)
                self.reset_data(i)
                context=self.teacher.sample(size=1).squeeze().item() if self.train else self.target.item()
                self.remotes[i].send(('update_context', context))
                self.cur_contexts[i]=context
                self.remotes[i].send(('reset', None))
                obs_, share_obs_, available_actions_, active_masks_ = self.remotes[i].recv()
                obs[i], share_obs[i], available_actions[i], active_masks[i] = obs_, share_obs_, available_actions_, active_masks_
        return obs, share_obs, rews, dones, infos, available_actions, active_masks
    
    def update_teacher(self, *args):
        context_info={}
        steps=self.context_buffer.size
        
        if self.teacher_name in ['spmarl', 'sprl']:
            context_info['g_mean']=self.teacher.context_dist.get_weights()[0]
            context_info['g_std']=np.sqrt(self.teacher.context_dist.covariance_matrix().sum())
        if self.teacher_name=='spmarl':
            self.teacher.update_distribution(self.total_steps, self.context_buffer.contexts[:steps], self.context_buffer.returns[:steps], np.stack(args[0])[:, None], np.stack(args[1])[:, None])
        else:
            self.teacher.update_distribution(self.total_steps, self.context_buffer.contexts[:steps], self.context_buffer.returns[:steps])
        
        context_info['sampled_mean']=self.context_buffer.contexts[:steps].mean()
        context_info['sampled_std']=self.context_buffer.contexts[:steps].std()
        self.context_buffer.clear()
        return context_info

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

class MatrixShareSelfPacedSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, teacher=None, args=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.train = True
        nenvs = len(env_fns)

        self.context_buffer = ContextBuffer(dim=1, max_buffer_size=10000)
        
        self.teacher=teacher
        self.teacher_name=self.teacher.teacher_name
        self.target=self.teacher.target
        
        self.max_agents=args.max_num_agents

        self.total_steps = 0
        
        self.undiscounted_rewards = np.zeros(len(env_fns))
        self.discounted_rewards = np.zeros(len(env_fns))
        self.cur_discs = np.ones(len(env_fns))
        self.step_lengths = np.zeros(len(env_fns))
        
        self.cur_contexts = [None] * len(env_fns)
        self.processed_contexts = [None] * len(env_fns)
        self.cur_initial_states = [None] * len(env_fns)

        self.discount_factor = args.gamma
        
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=matrixshareselfpacedworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions, active_masks):
        for remote, action, active_mask in zip(self.remotes, actions, active_masks):
            action=action[active_mask.squeeze()==True]
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, active_masks, avail_actions = zip(*results)
        obs, rews, dones, infos, active_masks, avail_actions = self.update((np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(active_masks), np.stack(avail_actions)))
        contexts=np.repeat(np.stack(self.cur_contexts)[:, None], self.max_agents, axis=1)[:,:,None]
        return obs, rews, dones, infos, active_masks, avail_actions, contexts

    def reset(self, train=True):
        self.train = train
        # contexts = self.teacher.sample(number=len(self.remotes))
        # contexts=np.random.randint(low=2, high=10, size=len(self.ps)) if train else np.ones(len(self.ps), dtype=np.int) * 2
        # contexts=np.ones(len(self.ps), dtype=np.int) * 10 if train else np.ones(len(self.ps), dtype=np.int) * self.target_context
        contexts=self.teacher.sample(size=len(self.ps)).squeeze() if train else np.ones(len(self.ps), dtype=np.int) * self.target
        for i in range(len(self.cur_contexts)):
            self.cur_contexts[i]=contexts[i] 
        for remote, context in zip(self.remotes, self.cur_contexts): 
            remote.send(('update_context', context))

        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, active_masks, avail_actions = zip(*results)
        return np.stack(obs), np.stack(active_masks), np.stack(avail_actions)
    
    def reset_data(self, env_id):
        self.undiscounted_rewards[env_id] = 0.
        self.discounted_rewards[env_id] = 0.
        self.cur_discs[env_id] = 1.
        self.step_lengths[env_id] = 0.

        self.cur_contexts[env_id] = None
        self.cur_initial_states[env_id] = None
    
    def update(self, step):
        obs, rews, dones, infos, active_masks, avail_actions = step
        self.total_steps+=len(self.ps)
        dones_env = np.zeros(len(self.cur_contexts), dtype=np.bool_)
        for i in range(len(self.cur_contexts)):
            self.undiscounted_rewards[i] += rews[i][active_masks[i].squeeze()==True].mean()
            self.discounted_rewards[i] += self.cur_discs[i] * rews[i][active_masks[i].squeeze()==True].mean()
            self.cur_discs[i] *= self.discount_factor
            self.step_lengths[i] += 1.

            dones_env[i] = np.all(dones[i][active_masks[i].squeeze()==True], axis=-1)

        if any(dones_env):
            index=[i for i, x in enumerate(dones_env) if x]
            for i in index:
                # context=np.random.randint(low=2, high=10) if self.train else 2
                # context=10 if self.train else self.target_context
                data = self.cur_contexts[i], self.undiscounted_rewards[i], self.discounted_rewards[i], self.step_lengths[i]
                if self.train:
                    self.context_buffer.update_buffer(data) 
                self.reset_data(i)
                context=self.teacher.sample(size=1).squeeze() if self.train else self.target
                self.remotes[i].send(('update_context', context))
                self.cur_contexts[i]=context
                self.remotes[i].send(('reset', None))
                obs_, active_masks_, avail_actions_ = self.remotes[i].recv()
                obs[i], active_masks[i], avail_actions[i] = obs_, active_masks_, avail_actions_
        return obs, rews, dones, infos, active_masks, avail_actions

    def update_teacher(self, *args):
        context_info={}
        steps=self.context_buffer.size
        
        if self.teacher_name in ['spmarl', 'sprl']:
            context_info['g_mean']=self.teacher.context_dist.get_weights()[0]
            context_info['g_std']=np.sqrt(self.teacher.context_dist.covariance_matrix().sum())
        if self.teacher_name=='spmarl':
            self.teacher.update_distribution(self.total_steps, self.context_buffer.contexts[:steps], self.context_buffer.returns[:steps], np.stack(args[0])[:, None], np.stack(args[1])[:, None])
        else:
            self.teacher.update_distribution(self.total_steps, self.context_buffer.contexts[:steps], self.context_buffer.returns[:steps])
        
        context_info['sampled_mean']=self.context_buffer.contexts[:steps].mean()
        context_info['sampled_std']=self.context_buffer.contexts[:steps].std()
        self.context_buffer.clear()
        return context_info

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

class PursuitSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=pursuitworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs = zip(*results)
        return np.stack(obs), np.stack(share_obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

def choosesimpleworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset(data)
            remote.send((ob))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError


class ChooseSimpleSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=choosesimpleworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(('reset', choose))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":   
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def chooseworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == 'reset':
            ob, s_ob, available_actions = env.reset(data)
            remote.send((ob, s_ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError


class ChooseSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=chooseworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(('reset', choose))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def chooseguardworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset(data)
            remote.send((ob))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError




# single env
class DummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError



class ShareDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
        self.actions = None

        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()
    
    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

class PursuitDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos = map(
            np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i], share_obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i], share_obs[i] = self.envs[i].reset()
        self.actions = None

        return obs, share_obs, rews, dones, infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs = map(np.array, zip(*results))
        return obs, share_obs

    def close(self):
        for env in self.envs:
            env.close()
    
    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

class ChooseDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results))
        self.actions = None
        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self, reset_choose):
        results = [env.reset(choose)
                   for (env, choose) in zip(self.envs, reset_choose)]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

class ChooseSimpleDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.actions = None
        return obs, rews, dones, infos

    def reset(self, reset_choose):
        obs = [env.reset(choose)
                   for (env, choose) in zip(self.envs, reset_choose)]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

class GuardSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = False  # could cause zombie process
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
        
class ChooseGuardSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=chooseguardworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = False  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(('reset', choose))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
