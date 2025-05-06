import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from snake_robot import generate_snake_mjcf

class PlanarSnakeEnv(gym.Env):
    def __init__(self, n_segments=5, total_length=1.0):
        """
        Initialize Planar Snake Robot environment.
        
        Args:
            n_segments (int): Number of body segments (excluding head)
            total_length (float): Total length of the snake robot
        """
        self.n_segments = n_segments
        self.total_length = total_length
        self.viewer = None
        self.frame_skip = 5
        
        # Generate MJCF model
        xml_string = generate_snake_mjcf(n_segments, total_length)
        
        # Load model directly from string
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        
        # Save initial state
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        
        # Set action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_segments,),
            dtype=np.float32
        )
        
        # Observation: joint positions and velocities
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def step(self, action):
        """Take a step in the environment."""
        # Get middle point position before step
        middle_idx = (self.n_segments + 1) // 2
        xpos_before = self.data.xpos[middle_idx][0]
        
        # Apply action
        self.data.ctrl[:] = action
        
        # Step the simulation
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        
        # Get middle point position after step
        xpos_after = self.data.xpos[middle_idx][0]
        
        # Calculate reward as forward distance of middle point
        reward = (xpos_after - xpos_before) / (self.model.opt.timestep * self.frame_skip)
        
        # Get observation
        obs = self._get_obs()
        
        # Always False for planar snake
        terminated = False
        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset state with small random noise
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv)
        
        self.set_state(qpos, qvel)
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Get observation."""
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy()
        ])

    def set_state(self, qpos, qvel):
        """Set the state of the environment."""
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        """Render the environment."""
        if self.viewer is None:
            try:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
                
                # Set camera parameters
                self.viewer.cam.distance = self.model.stat.extent * 1.0
                self.viewer.cam.elevation = 0
                self.viewer.cam.azimuth = 90
            except Exception as e:
                print(f"Failed to initialize viewer: {e}")
                print("Continuing without visualization")
                return None

        try:
            self.viewer.sync()
            return self.viewer
        except Exception as e:
            print(f"Render failed: {e}")
            return None

    def close(self):
        """Clean up resources."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# Example usage
if __name__ == "__main__":
    env = PlanarSnakeEnv(n_segments=5, total_length=1.0)
    
    obs, _ = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()