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


class RandomEnv:
  def __init__(self, n_units, **kwargs):
    self._epslen = 0
    self.observation_space = gym.spaces.Box(high=float('inf'), low=0, shape=(56,))
    self.obs_shape = {
      'obs': self.observation_space.shape, 
      'global_state': self.observation_space.shape, 
    }
    self.obs_dtype = {
      'obs': np.float32,
      'global_state': np.float32
    }
    self.action_space = {
      'action_discrete': gym.spaces.Discrete(4), 
      'action_continuous': gym.spaces.Box(low=-1, high=1, shape=(2,))}
    self.action_shape = {k: v.shape for k, v in self.action_space.items()}
    self.action_dtype = {k: v.dtype for k, v in self.action_space.items()}
    self.is_action_discrete = {k: isinstance(v, gym.spaces.Discrete) 
                               for k, v in self.action_space.items()}
    self.action_dim = {
      k: self.action_space[k].n if v else self.action_shape[k][0]
      for k, v in self.is_action_discrete.items()}
    
    self.n_units = n_units
    
    self._epslen = 0
    self._score = np.zeros(self.n_units)
    self._dense_score = np.zeros(self.n_units)

    self.max_episode_steps = int(1e3)
    self.reward_range = (-float('inf'), float('inf'))
    self.metadata = {}
    self._reset_done_interval()

  def seed(self, seed=None):
    pass

  def random_action(self):
    action = {k: [None] * self.n_units for k in self.action_space}
    for k, v in self.action_shape.items():
      for i in range(self.n_units):
        action[k][i] = v.sample()
    action = {k: np.array(v) for k, v in action.items()}
    return action

  def reset(self):
    self._epslen = 0
    self._score = np.zeros(self.n_units)
    self._dense_score = np.zeros(self.n_units)

    self._reset_done_interval()

    return self._get_obs()
  
  def step(self, action=None):
    r = np.ones(self.n_units) * self._epslen / self.done_interval
    d = np.ones(self.n_units) * (self._epslen % self.done_interval == 0)
    self._epslen += 1
    o = self._get_obs()

    self._score += r
    self._dense_score += r

    info = {
      'score': self._score,
      'dense_score': self._dense_score, 
      'epslen': self._epslen, 
      'game_over': self._epslen % self.done_interval == 0
    }

    return o, r, d, info
  
  def _get_obs(self):
    return {k: np.random.random((self.n_units,) + v) for k, v in self.obs_shape.items()}

  def _reset_done_interval(self):
    self.done_interval = np.random.randint(10, 1000)

  @property
  def is_multiagent(self):
    return False
