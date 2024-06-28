import numpy as np
# import Tacview_render. tcp_ip_server_TacView client as tacview 
# import navy, import time
from math import pi
import gym
f32 = np.float32
inf = float('inf')

from envs.utils import *


class AircraftF16:
  def __init__(
    self, 
    id, 
    sim_step, 
    auto_throttle, 
    n_missile, 
    color, 
    launch_interval, 
    ctrl_mod, 
  ):
    self.observation_space = gym.spaces.Box(
      np.array([-inf]*49), np.array([inf]*49), dtype='float')
    self.action_space = gym.spaces.Box(
      np.array([-3.0, -50.0, 0.01]) , np.array([19.0, 50.0 ,120.01]), dtype='float')

  def plane_step(self, action, planes, target_id):
    pass
  
  def reset(self, pos):
    return {}

  def step(self, planes, target_id):
    obs = {}
    reward = np.random.randint(0, 2)
    done = np.random.randint(0, 2)
    return obs, reward, done, {}


class AerialCombat:
  def __init__(
    self, 
    env_name, 
    plane_config, 
    max_sim_time, 
    born_point, 
    uid2aid=[0, 1], 
    uid2gid=[0, 1]
  ):
    self.sim_step = plane_config['sim_step']
    self.born_point = born_point
    self.max_episode_steps = max_sim_time / self.sim_step

    self.done = False
    self.timer = 0 # tacview用的时间
    self.render_init = False
    self.viewer = None
    self.n_units = len(uid2aid)
    self.units = [AircraftF16(**plane_config) for i in range(self.n_units)]

    self.observation_space = [a.observation_space for a in self.units]
    self.action_space = [a.action_space for a in self.units]
    self.is_action_discrete = [isinstance(a, gym.spaces.Discrete) for a in self.action_space]
    self.action_dim = [
      self.action_space[i].n if self.is_action_discrete[i] else self.action_space[i].shape[0] 
      for i in range(self.n_units)]
    self.action_dtype = [np.int32 for _ in self.action_space]

    self.uid2gid = uid2gid
    self.uid2aid = uid2aid
    self.aid2uids = compute_aid2uids(self.uid2aid)
    self.gid2uids = compute_aid2uids(self.uid2gid)
    self.aid2gids = compute_aid2gids(uid2aid, uid2gid)
    self.n_agents = len(self.aid2uids)

    self._epslen = 0
    self._score = np.zeros(self.n_units, dtype=f32)
    self._dense_score = np.zeros(self.n_units, dtype=f32)

  def random_action(self):
    action = [a.sample() for a in self.action_space]
    return action

  def seed(self, seed=None):
    return 

  def reset(self):
    self._epslen = 0
    self._score = np.zeros(self.n_units, dtype=f32)
    self._dense_score = np.zeros(self.n_units, dtype=f32)

    self.done = False
    agent_obs = []
    for i in range(self.n_units):
      obs = self.units[i].reset((
        self.born_point['pn'][i], 
        self.born_point['pe'][i],
        self.born_point['alt'][i], 
        self.born_point['heading'][i],
        self.born_point['vt'][i])
      )
      agent_obs.append(obs)

    return agent_obs 
  
  def step(self, actions):
    self._epslen += 1

    task_menu = {0: 1, 1: 0}
    for i in range(self.n_units):
      target_id = task_menu[i]
      self.units[i].plane_step(actions[i], planes=self.units, target_id=target_id)

    unit_obs = []
    unit_reward = []
    unit_done = []
    for i in range(self.n_units):
      target_id = task_menu[i]
      obs, reward, done, _ = self.units[i].step(self.units, target_id=target_id)
      unit_obs.append(obs)
      unit_reward.append(reward)
      unit_done.append(done)
    
    done = np.any(unit_done)
    unit_reward = np.array(unit_reward)
    unit_done = np.array([done for _ in range(self.n_units)])
    if self._epslen >= self.max_episode_steps:
      done = True
      if self.units[0].health > self.units[1].health:
        unit_reward[0] = unit_reward[0] + 50
        unit_reward[1] = unit_reward[1] - 50
      if self.units[0].health < self.units[1].health:
        unit_reward[0] = unit_reward[0] - 50
        unit_reward[1] = unit_reward[1] + 50

    self._dense_score += np.array(unit_reward)
    if done:
      # TODO: 
      self._score = np.array(unit_reward) > 0

    info = {
      'score': self._score,
      'dense_score': self._dense_score,
      'epslen': self._epslen,
      'game_over': done
    }

    return unit_obs, unit_reward, unit_done, info
