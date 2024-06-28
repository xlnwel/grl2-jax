import numpy as np
import gym

from envs.utils import *


class RandomEnv:
  def __init__(self, uid2aid=[0, 1], uid2gid=[0, 1], max_episode_steps=100, **kwargs):
    self.uid2aid = uid2aid
    self.uid2gid = uid2gid
    self.aid2uids = compute_aid2uids(self.uid2aid)
    self.gid2uids = compute_aid2uids(self.uid2gid)
    self.aid2gids = compute_aid2gids(uid2aid, uid2gid)
    self.n_units = len(self.uid2aid)
    assert self.n_units == 2, uid2aid
    self.n_agents = len(self.aid2uids)

    self.observation_space = [{
      'obs': gym.spaces.Box(high=float('inf'), low=0, shape=(6,)),
      'global_state': gym.spaces.Box(high=float('inf'), low=0, shape=(6,))
    } for _ in range(self.n_agents)]
    self.obs_shape = [{
      k: v.shape for k, v in obs.items()
    } for obs in self.observation_space]
    self.obs_dtype = [{
      k: v.dtype for k, v in obs.items()
    } for obs in self.observation_space]
    self.action_space = [{
      'action_discrete': gym.spaces.Discrete(4), 
      'action_continuous': gym.spaces.Box(low=-1, high=1, shape=(2,))
    } for _ in range(self.n_agents)]
    self.action_shape = [{
      k: v.shape for k, v in a.items()
    } for a in self.action_space]
    self.action_dtype = [{
      k: v.dtype for k, v in a.items()
    } for a in self.action_space]
    self.is_action_discrete = [{
      k: isinstance(v, gym.spaces.Discrete) for k, v in a.items()
    } for a in self.action_space]
    self.action_dim = [{
      k: aspace[k].n if v else ashape[k][0] for k, v in iad.items()
    } for aspace, ashape, iad in zip(self.action_space, self.action_shape, self.is_action_discrete)]

    self._epslen = 0
    self._score = np.zeros(self.n_units)
    self._dense_score = np.zeros(self.n_units)

    self.max_episode_steps = max_episode_steps
    self.reward_range = (-float('inf'), float('inf'))
    self.metadata = {}
    self._reset_done_interval()

  def seed(self, seed=None):
    pass

  def random_action(self):
    action = [{
      k: np.array([v.sample() for _ in self.aid2uids[i]]) for k, v in a.items()} 
      for i, a in enumerate(self.action_space)]
    return action

  def reset(self):
    """重置环境

    Returns:
      Dict: 环境重置后的观测
    """
    self._epslen = 0
    self._score = np.zeros(self.n_units)
    self._dense_score = np.zeros(self.n_units)
    obs = np.zeros((self.n_units, self.n_units * 3))
    obs = {'obs': obs, 'global_state': obs}

    self._reset_done_interval()
    self._pick_maximum()

    return obs
  
  def step(self, action=None):
    if len(action) == 1:
      a1 = {k: v[:1] for k, v in action[0].items()}
      a2 = {k: v[1:] for k, v in action[0].items()}
    else:
      a1, a2 = action
    if self.max_action['action_discrete']:
      a1_disc = a1['action_discrete'] % self.max_action['action_discrete']
      a2_disc = a2['action_discrete'] % self.max_action['action_discrete']
    else:
      a1_disc = a1['action_discrete']
      a2_disc = a2['action_discrete']
    if self.max_action['action_continuous']:
      a1_cont = (a1['action_continuous'] + 1) % self.max_action['action_continuous']
      a2_cont = (a2['action_continuous'] + 1) % self.max_action['action_continuous']
    else:
      a1_cont = (a1['action_continuous'] + 1)
      a2_cont = (a2['action_continuous'] + 1)
    if np.sum(a1_disc > a2_disc) + np.sum(a1_cont > a2_cont):
      r = np.array([1, -1])
    elif np.sum(a1_disc < a2_disc) + np.sum(a1_cont < a2_cont):
      r = np.array([-1, 1])
    else:
      r = np.zeros(self.n_units)
    assert len(r) == self.n_units
    d = np.ones(self.n_units) * (self._epslen % self.done_interval == 0)
    self._epslen += 1
    o1 = np.concatenate([
      a1['action_discrete'][..., None], a1['action_continuous'], 
      a2['action_discrete'][..., None], a2['action_continuous']
    ], axis=-1)
    o2 = np.concatenate([
      a2['action_discrete'][..., None], a2['action_continuous'], 
      a1['action_discrete'][..., None], a1['action_continuous']
    ], axis=-1)
    o = np.concatenate([o1, o2], axis=0)
    o = {'obs': o, 'global_state': o}

    self._dense_score += r
    self._score = np.sign(self._dense_score)

    info = {
      'score': self._score,
      'dense_score': self._dense_score, 
      'epslen': self._epslen, 
      'game_over': self._epslen % self.done_interval == 0
    }

    return o, r, d, info
  
  def _reset_done_interval(self):
    self.done_interval = np.random.randint(10, 1000)

  def _pick_maximum(self):
    self.max_action = {'action_discrete': np.random.randint(4), 'action_continuous': np.random.randint(3)}

  @property
  def is_multiagent(self):
    return False

  def close(self):
    pass