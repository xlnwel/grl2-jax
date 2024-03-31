import numpy as np
import gym

from core.names import DEFAULT_ACTION


class DummyEnv:
  """ Useful to break the inheritance of unexpected attributes.
  For example, control tasks in gym by default use frame_skip=4,
  but we usually don't count these frame skips in practice.
  """
  def __init__(self, env):
    self.env = env
    self.observation_space = env.observation_space
    self.obs_shape = {'obs': self.observation_space.shape}
    self.obs_dtype = {
      'obs': np.float32 if np.issubdtype(self.observation_space.dtype, np.floating) else self.observation_space.dtype
    }
    self.action_space = {DEFAULT_ACTION: env.action_space}
    self.action_shape = {k: v.shape for k, v in self.action_space.items()}
    self.action_dtype = {k: v.dtype for k, v in self.action_space.items()}
    self.is_action_discrete = {k: isinstance(v, gym.spaces.Discrete) 
                               for k, v in self.action_space.items()}
    self.action_dim = {
      k: self.action_space[k].n if v else self.action_shape[k][0]
      for k, v in self.is_action_discrete.items()}
    self.spec = env.spec
    self.reward_range = env.reward_range
    self.metadata = env.metadata

    self.render = env.render
    self.close = env.close
    self._seed = env.seed

  @property
  def is_multiagent(self):
    return getattr(self.env, 'is_multiagent', False)

  def seed(self, seed):
    seed = self.env.seed(seed)

  def random_action(self):
    if self.is_action_discrete:
      return {DEFAULT_ACTION: np.random.randint(self.action_dim)}
    else:
      return {DEFAULT_ACTION: np.random.uniform(-1, 1, size=self.action_shape)}
  
  def reset(self):
    obs = self.env.reset()
    obs = {'obs': obs}
    return obs

  def step(self, action):
    action = action[DEFAULT_ACTION]
    obs, reward, done, info = self.env.step(action)
    obs = {'obs': obs}
    info = {}
    return obs, reward, done, info
