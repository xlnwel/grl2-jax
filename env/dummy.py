import numpy as np
import gym


class Dummy(gym.Wrapper):
    def __init__(self, obs_shape=(4,), action_dim=3):
        self.count = 0
        self.observation_space = gym.spaces.Box(high=float('inf'), low=0, shape=obs_shape)
        self.action_space = gym.spaces.Discrete(action_dim)
        self._reset_done_interval()

    def reset(self):
        self.count = 0
        self._reset_done_interval()
        return np.ones(self.observation_space.shape) * self.count
    
    def step(self, action=None):
        r = self.count
        d = self.count % self.done_interval == 0
        self.count += 1
        o = np.ones(self.observation_space.shape) * self.count

        return o, r, d, {}
    
    def _reset_done_interval(self):
        self.done_interval = np.random.randint(10, 1000)