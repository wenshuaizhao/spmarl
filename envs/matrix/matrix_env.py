import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import pygame

blue = (0, 0, 128)
red = (255, 0, 0)


class ClimbingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, args=None, size=3):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        # self.reward_array = np.array([[0, 6, 5], [-30, 7, 0], [11, -30, 0]])
        self.scenario_name=args.scenario_name
        self.num_agents=2
        if self.scenario_name=='climbing':
            self.reward_array = np.array([[0, 6, 5], [-30, 7, 0], [11, -30, 0]])
        elif self.scenario_name.startswith('penalty'):
            self.k=-int(self.scenario_name.split('_')[1])
            self.reward_array = np.array([[self.k, 0, 10], [0, 2, 0], [10, 0, self.k]])
        elif self.scenario_name=='permutation':
            self.num_agents = args.num_agents
            self.max_agents = args.num_agents
        elif self.scenario_name=='nonmonotonic':
            self.reward_array = np.array([[-12, 0, 0], [-12, 0, 0], [8, -12, -12]])

        self._step = 0
        self.highlight = (0, 0)
        self.length=1
        self.share_reward= True
        self.use_fixed_obs=True
        self.fixed_obs=[i+1 for i in range(args.num_agents)]

        # Observations 
        self.agent_ob_space=spaces.Box(0, self.length, shape=(args.num_agents,), dtype=int)
        self.observation_space = [self.agent_ob_space]*self.num_agents
        self.share_observation_space = [self.agent_ob_space for _ in range(self.num_agents)]

        # Actions
        if self.scenario_name=='permutation':
            self.agent_action_space=spaces.Discrete(self.num_agents)
            self.action_space = [self.agent_action_space]*self.num_agents
        else:
            self.agent_action_space=spaces.Discrete(3)
            self.action_space = [self.agent_action_space]*self.num_agents


        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        # self.render_mode = render_mode

        self.window = None
        self.clock = None
    
    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def _get_obs(self):
        return np.array([[0], [1]])

    def _get_info(self):
        return {'step': self._step}
    
    def update_context(self, context=None):
        if context is not None:
            self.num_agents = int(context)
        else:
            raise RuntimeError("Context should not be None.")

    def reset(self):
        # We need the following line to seed self.np_random
        self._step = 0   
        
        if self.use_fixed_obs:
            # obs=np.array([np.array(self.fixed_obs)] * self.num_agents)
            
            obs = np.eye(self.max_agents)
        else:
            obs=np.array([np.array([self._step, self._step])] * self.num_agents)
        
        active_masks = np.ones((self.max_agents, 1), dtype=np.float32)
        avail_actions = np.array([[0] * self.action_space[0].n] * self.max_agents)
        avail_actions[:self.num_agents,:self.num_agents] = 1
        
        if self.max_agents-self.num_agents > 0:
            # obs = np.concatenate([obs, np.stack([np.zeros_like(obs[0]) for i in range(self.max_agents-self.num_agents)])], 0)
            active_masks[self.num_agents:] = np.zeros(((self.max_agents-self.num_agents), 1), dtype=np.float32)

        return obs, active_masks, avail_actions

    def step(self, actions: tuple):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        if not isinstance(actions, tuple):
            actions = tuple(actions)
        self.highlight = actions
        if self.scenario_name == 'permutation':
            reward = np.array([[self.num_agents == len(set(actions))]]*self.num_agents, dtype=np.float32)
        else:
            reward = np.array([[self.reward_array[actions]]]*self.num_agents)
        info = self._get_info()
        self._step += 1
        if self._step==self.length:
            done=True
            self._step=0
        else:
            done=False

        active_masks = np.ones((self.max_agents, 1), dtype=np.float32)

        if self.use_fixed_obs:
            # obs=np.array([np.array(self.fixed_obs)] * self.num_agents)
            # obs = np.concatenate([np.arange(self.num_agents)[:, np.newaxis]+1] * 2, -1)
            obs = np.eye(self.max_agents)
        else:
            obs=np.array([np.array([self._step, self._step])] * self.num_agents)
        
        done=np.array([done] * self.num_agents)
        if self.max_agents-self.num_agents > 0:
            # obs = np.concatenate([obs, np.stack([np.zeros_like(obs[0]) for i in range(self.max_agents-self.num_agents)])], 0)
            reward = np.concatenate([reward, np.stack([np.zeros_like(reward[0]) for i in range(self.max_agents-self.num_agents)])], 0)
            done = np.concatenate([done, np.stack([np.zeros_like(done[0]) for i in range(self.max_agents-self.num_agents)])], 0)
            active_masks[self.num_agents:] = np.zeros(((self.max_agents-self.num_agents), 1), dtype=np.float32)
        
        avail_actions = np.array([[0] * self.action_space[0].n] * self.max_agents)
        avail_actions[:self.num_agents,:self.num_agents] = 1

        return obs, reward, done, info, active_masks, avail_actions

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size+100))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size+100))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            font2 = pygame.font.SysFont('didot.ttc', 72)
            text = font2.render('Step: '+str(self._step), True, blue)
            textRect = text.get_rect()
            textRect.center = (self.window_size // 2, self.window_size+50)
            self.window.blit(text, textRect)
            for x in range(self.size):
                for y in range(self.size):
                    text = font2.render(
                        str(self.reward_array[x, y]), True, red if (x,y)== self.highlight else blue)
                    textRect = text.get_rect()
                    textRect.center = ((self.window_size // self.size)*y+(self.window_size // (
                        self.size*2)), (self.window_size // self.size)*x+(self.window_size // (self.size*2)))
                    self.window.blit(text, textRect)

            # self.window.blit([text, text2], [textRect, textRect2])
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
