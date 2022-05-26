import numpy as np
import gym


class DummyEnv:
    """ Useful to break the inheritance of unexpected attributes.
    For example, control tasks in gym by default use frame_skip=4,
    but we usually don't count these frame skips in practice.
    """
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.spec = env.spec
        self.reward_range = env.reward_range
        self.metadata = env.metadata

        self.reset = env.reset
        self.step = env.step
        self.render = env.render
        self.close = env.close
        self.seed = env.seed

    @property
    def is_multiagent(self):
        return getattr(self.env, 'is_multiagent', False)

    def random_action(self):
        return self.action_space.sample()


class RandomEnv():
    """ Wrapper for built-in gym environments to 
    hide unexpected attributes. 
    Single agent is assumed.
    """
    def __init__(self, obs_shape=(4,), action_dim=3, is_action_discrete=True, **kwargs):
        self.count = 0
        self.observation_space = gym.spaces.Box(high=float('inf'), low=0, shape=obs_shape)
        self.action_space = gym.spaces.Discrete(action_dim)
        self.action_dim = action_dim
        self.is_action_discrete = is_action_discrete
        self.max_episode_steps = int(1e3)
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self._reset_done_interval()

    def reset(self):
        self.count = 1
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

    @property
    def is_multiagent(self):
        return False
