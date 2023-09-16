import numpy as np
from gym.spaces import Discrete

from env.utils import *


class IteratedPrisonersDilemma:
  """
  A two-agent vectorized environment.
  Possible actions for each agent are (C)ooperate and (D)efect.
  """
  # Possible actions
  NUM_AGENTS = 2
  NUM_ACTIONS = 2
  NUM_OBS = 5
  NUM_STATES = 5

  def __init__(self, max_episode_steps, use_idx=False, use_hidden=False, **kwargs):
    self.max_episode_steps = max_episode_steps

    self.payout_mat1 = np.array([[-1., -3.], [0., -2.]])
    self.payout_mat2 = np.array([[-1., 0.], [-3., -2.]])

    self.use_idx = use_idx
    self.use_hidden = use_hidden

    self.action_space = Discrete(self.NUM_ACTIONS)
    self.action_shape = self.action_space.shape
    self.action_dim = self.action_space.n
    self.action_dtype = np.int32
    self.is_action_discrete = True

    self.obs_shape = dict(
      obs=(self.NUM_OBS,), 
    )
    self.obs_dtype = dict(
      obs=np.float32, 
    )
    if use_idx:
      self.obs_shape['idx'] = (self.NUM_AGENTS,)
      self.obs_dtype['idx'] = np.float32
    if use_hidden:
      self.obs_shape['hidden_state'] = (self.NUM_STATES, )
      self.obs_dtype['hidden_state'] = np.float32

    self.step_count = None
    self._dense_score = np.zeros(self.NUM_ACTIONS, np.float32)
    self._score = np.zeros(self.NUM_ACTIONS, np.float32)
    self._coop_score = np.zeros(self.NUM_ACTIONS, np.float32)
    self._coop_def_score = np.zeros(self.NUM_ACTIONS, np.float32)
    self._defect_score = np.zeros(self.NUM_ACTIONS, np.float32)
    self._epslen = 0

  def seed(self, seed=None):
    pass

  def random_action(self):
    return [np.random.randint(d) for d in self.action_dim]

  def reset(self):
    self.step_count = 0
    self._dense_score = np.zeros(self.NUM_ACTIONS, np.float32)
    self._score = np.zeros(self.NUM_ACTIONS, np.float32)
    self._coop_score = np.zeros(self.NUM_ACTIONS, np.float32)
    self._coop_def_score = np.zeros(self.NUM_ACTIONS, np.float32)
    self._defect_score = np.zeros(self.NUM_ACTIONS, np.float32)
    self._epslen = 0
    self._defect_step = []

    obs = self._get_obs()

    return obs

  def step(self, action):
    a0, a1 = action
    if isinstance(a0, (list, tuple, np.ndarray)):
      a0 = a0[0]
      a1 = a1[0]

    self.step_count += 1

    obs = self._get_obs([a0, a1])
    
    reward = np.array([self.payout_mat1[a0, a1], self.payout_mat2[a0, a1]], np.float32)
    done = self.step_count == self.max_episode_steps
    dones = np.ones(self.NUM_AGENTS, np.float32) * done

    self._dense_score += reward
    self._coop_score += a0 == a1 == 0
    self._coop_def_score += a0 != a1
    self._defect_score += a0 == a1 == 1
    self._score += reward
    self._epslen += 1

    info = {
      'dense_score': self._dense_score, 
      'score': self._score, 
      'epslen': self._epslen, 
      'coop_score': self._coop_score, 
      'coop_defect_score': self._coop_def_score, 
      'defect_score': self._defect_score, 
      'game_over': done, 
    }

    return obs, reward, dones, info

  def _get_obs(self, action=None):
    obs = {
      'obs': np.zeros((self.NUM_AGENTS, self.NUM_OBS), np.float32), 
    }
    if self.use_idx:
      obs['idx'] = np.eye(self.NUM_AGENTS, dtype=np.float32)
    if self.use_hidden:
      obs['hidden_state'] = np.zeros((self.NUM_AGENTS, self.NUM_STATES), np.float32)

    if action is None:
      obs['obs'][:, -1] = 1
      # if self.use_hidden:
      #   obs['hidden_state'][:, -1] = 1
    else:
      a0, a1 = action
      
      obs['obs'][0, a0 * 2 + a1] = 1
      obs['obs'][1, a1 * 2 + a0] = 1
      if self.use_hidden:
        obs['hidden_state'][:, a0 * 2 + a1] = 1

    obs['obs'][0, -3] = 1
    obs['obs'][1, -2] = 1
    if self.use_hidden:
      obs['hidden_state'][:, -1] = self._epslen / self.max_episode_steps

    return obs

  def close(self):
    pass