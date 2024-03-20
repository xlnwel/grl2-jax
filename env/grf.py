import numpy as np
from gym.spaces import Discrete
f32 = np.float32

from core.names import DEFAULT_ACTION
from env.grf_env.selected_agents import SelectedAgents
from env.utils import *


def do_flatten(obj):
  """Run flatten on either python list or numpy array."""
  if type(obj) == list:
      return np.array(obj).flatten()
  return obj.flatten()


class Representation:
  RAW='raw'
  CUSTOM='custom'
  MAT='mat'
  SIMPLE115='simple115v2'


class GRF:
  def __init__(
    self,
    # built-in configs for grf
    env_name,
    representation=Representation.SIMPLE115,
    write_goal_dumps=False,
    write_full_episode_dumps=False,
    render=False,
    write_video=False,
    dump_frequency=1000,
    logdir='results/grf',
    extra_players=None,
    control_left=True,
    control_right=False,
    # custom grf configs
    score_reward_scale=1, 
    reward_type='sum',
    # required configs for grl
    max_episode_steps=3000,
    use_action_mask=False,
    use_sample_mask=False, 
    seed=None, 
    use_idx=False, 
    **kwargs,
  ):
    self.name = env_name
    self.representation = representation
    self.to_render = render
    self.score_reward_scale = score_reward_scale
    self.reward_type = reward_type

    rewards = 'scoring'
    # print('other config options', other_config_options)
    self.env = SelectedAgents(
      self.name, 
      representation=representation,
      rewards=rewards,
      write_goal_dumps=write_goal_dumps,
      write_full_episode_dumps=write_full_episode_dumps,
      render=render,
      write_video=write_video,
      dump_frequency=dump_frequency,
      logdir=logdir,
      extra_players=extra_players,
      control_left=control_left,
      control_right=control_right,
    )
    self.seed(seed)
    self.n_left_units = self.env.n_left_controlled_units
    self.n_right_units = self.env.n_right_controlled_units
    self.n_units = self.env.n_controlled_units

    self.max_episode_steps = max_episode_steps

    self.use_action_mask = use_action_mask  # if action mask is used
    self.use_sample_mask = use_sample_mask              # if life mask is used
    self.use_idx = use_idx

    self.action_space = Discrete(19)

    self.observation_space = self.env.observation_space
    self.reward_range = self.env.reward_range
    self.metadata = self.env.metadata
    obs = self.reset()
    self.obs_shape = {k: v.shape[-1:] for k, v in obs.items()}
    self.obs_dtype = {k: v.dtype for k, v in obs.items()}

    # The following stats should be updated in self.step and be reset in self.reset
    # The episodic score we use to evaluate agent's performance. It excludes shaped rewards
    self._score = np.zeros(self.n_units, dtype=f32)
    # The accumulated episodic rewards we give to the agent. It includes shaped rewards
    self._dense_score = np.zeros(self.n_units, dtype=f32)
    # The length of the episode
    self._epslen = 0
    self._left_score = np.zeros(self.n_units, dtype=f32)
    self._right_score = np.zeros(self.n_units, dtype=f32)

    self._checkpoint_reward = .1
    self._num_checkpoints = 10
    self._collected_checkpoints = [0, 0]


    if representation == Representation.SIMPLE115:
      self.feature_mask = {
        'obs': np.array([1] * 88 + [1] * 6 + [0] * 21), 
        'global_state': np.array([1] * 88 + [1] * 6 + [0] * 21)
      }
    else:
      self.feature_mask = {
        'obs': np.array([1] * 8 + [1] * 6 + [0] * 3 + [0] * 11 * self.n_units + [0] * 7), 
        'global_state': np.array([1] * 8 + [1] * 6 + [0] * 3 + [0] * 11 + [0] * 7)
      }
    for k, v in self.feature_mask.items():
      assert v.shape == self.obs_shape[k], (k, v.shape, self.obs_shape[k])

  def random_action(self):
    action = self.env.random_action()

    return action

  def reset(self):
    obs = self.env.reset()

    self._score = np.zeros(self.n_units, dtype=f32)
    self._dense_score = np.zeros(self.n_units, dtype=f32)
    self._epslen = 0
    self._left_score = np.zeros(self.n_units, dtype=f32)
    self._right_score = np.zeros(self.n_units, dtype=f32)
    self._ckpt_score = np.zeros(self.n_units, dtype=f32)

    self._collected_checkpoints = [0, 0]

    # return [{'obs': o[None], 'global_state': o[None]} for o in obs]
    return self._get_obs(obs)

  def step(self, action):
    action = action[0][DEFAULT_ACTION]
    obs, reward, done, info = self.env.step(action)

    if self.reward_type == 'sum':
      reward = np.sum(reward) * np.ones(self.n_left_units)
    else:
      reward = self._get_reward(reward, info)

    self._epslen += 1
    self._dense_score += reward
    self._left_score += 1 if info['score_reward'] == 1 else 0
    self._right_score += 1 if info['score_reward'] == -1 else 0
    diff_score = self._left_score - self._right_score
    if self.n_right_units != 0:
      diff_score[-self.n_right_units:] *= -1
    if self.name.startswith('11_vs_11') and self._epslen == self.max_episode_steps:
      done = True
      self._score = np.where(
        self._left_score < self._right_score, -1, 
        self._left_score > self._right_score)
      self._score[self.n_left_units:] *= -1
    else:
      self._score = diff_score
    dones = np.tile(done, self.n_units)

    info = {
      'score': self._score,
      'dense_score': self._dense_score,
      'left_score': self._left_score,
      'right_score': self._right_score,
      'diff_score': diff_score,
      'win_score': diff_score > 0,
      'checkpoint_score': self._ckpt_score,
      'epslen': self._epslen,
      'game_over': done
    }

    return self._get_obs(obs), reward, dones, info

  def render(self):
    if not self.to_render:
      self.env.render(mode='rgb_array')
      self.to_render = True
    obs = self.raw_state()[0]
    return obs['frame']

  def close(self):
    return self.env.close()

  def _get_obs(self, obs):
    if self.representation == Representation.SIMPLE115:
      obs = dict(
        obs=obs, 
        global_state=obs
      )
    else:
      obs = dict(
        obs=np.concatenate([self.get_state(u >= self.n_left_units, u) for u in range(self.n_units)], 0), 
        global_state=np.concatenate([self.get_state(u >= self.n_left_units) for u in range(self.n_units)], 0), 
      )

    return obs

  def get_state(self, side=0, uid=None):
    # adapted from imple115StateWrapper.convert_observation
    raw_state = self.env.raw_state(side)

    def do_flatten(obj):
      """Run flatten on either python list or numpy array."""
      if type(obj) == list:
        return np.array(obj).flatten()
      return obj.flatten()

    s = []
    for i, name in enumerate(
      ["left_team", "left_team_direction", "right_team", "right_team_direction"]
    ):
      s.extend(do_flatten(raw_state[0][name]))
      # If there were less than 11vs11 players we backfill missing values
      # with -1.
      # if len(s) < (i + 1) * 22:
      #   s.extend([-1] * ((i + 1) * 22 - len(s)))
    # ball position
    s.extend(raw_state[0]["ball"])
    # ball direction
    s.extend(raw_state[0]["ball_direction"])
    # one hot encoding of which team owns the ball
    if raw_state[0]["ball_owned_team"] == -1:
      s.extend([1, 0, 0])
    if raw_state[0]["ball_owned_team"] == 0:
      s.extend([0, 1, 0])
    if raw_state[0]["ball_owned_team"] == 1:
      s.extend([0, 0, 1])
    if uid is None:
      for obs in raw_state:
        active = [0] * 11
        if obs["active"] != -1:
          active[obs["active"]] = 1
        s.extend(active)
    else:
      active = [0] * 11
      active[uid] = 1
      s.extend(active)
    game_mode = [0] * 7
    game_mode[raw_state[0]["game_mode"]] = 1
    s.extend(game_mode)
    
    s = np.array(s, dtype=f32)
    return np.expand_dims(s, 0)
  
  def _get_reward(self, reward, info):
    def ckpt_reward(side):
      n_units = self.n_left_units if side == 0 else self.n_right_units
      reward = np.zeros(n_units, dtype=f32)
      assert side in [0, 1], side
      if (info['score_reward'] == 1 and side == 0) or \
          (info['score_reward'] == -1 and side == 1):
        reward += self._checkpoint_reward * (
            self._num_checkpoints - self._collected_checkpoints[side])
        self._collected_checkpoints[side] = self._num_checkpoints
      else:
        o = self.raw_state(side)[0]
        if 'ball_owned_team' not in o or o['ball_owned_team'] != side:
          return reward
        d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5
        while self._collected_checkpoints[side] < self._num_checkpoints:
          threshold = (.99 - .8 / (self._num_checkpoints - 1)
            * self._collected_checkpoints[side])
          if d > threshold:
            break
          reward += self._checkpoint_reward
          self._collected_checkpoints[side] += 1

      return reward

    if self.n_right_units == 0:
      reward = ckpt_reward(0)
      self._ckpt_score += reward
      reward = reward + info['score_reward'] * self.score_reward_scale
      # np.testing.assert_allclose(reward, np.mean(reward))
    else:
      reward = [None, None]
      left_ckpt = ckpt_reward(0)
      right_ckpt = ckpt_reward(1)
      reward[0] = left_ckpt
      reward[1] = right_ckpt
      self._ckpt_score[:self.n_left_units] += reward[0]
      self._ckpt_score[self.n_left_units:] += reward[1]
      reward[0] = reward[0] + info['score_reward'] * self.score_reward_scale
      reward[1] = reward[1] - info['score_reward'] * self.score_reward_scale
      # np.testing.assert_allclose(reward[0], np.mean(reward[0]))
      # np.testing.assert_allclose(reward[1], np.mean(reward[1]))
      reward = np.concatenate(reward)

    return reward

  def raw_state(self, side=None):
    obs = self.env.raw_state(side)
    return obs

  def seed(self, seed):
    self.env.unwrapped.seed(seed)
    return seed

def parse_args():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--left', '-l', type=bool, default=True)
  parser.add_argument('--right', '-r', type=bool, default=False)
  parser.add_argument('--step', '-s', type=int, default=1000)
  parser.add_argument('--unit', '-u', action='store_true')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()
  config = {
      'env_name': 'academy_counterattack_hard',
      'representation': 'simple115v2',
      'write_full_episode_dumps': False, 
      'write_video': False, 
      'render': False, 
      'rewards': 'scoring,checkpoints', 
      'control_left': args.left,
      'control_right': args.right,
      'shared_policy': True, 
      'use_action_mask':False, 
      'uid2aid': None,
      'use_idx': True,
      'use_hidden': False, 
      'agentwise_global_state': False, 
      'render': False, 
      'selected_agents': False, 
      'seed': 1
  }

  import random
  random.seed(0)
  import numpy as np
  np.random.seed(10)
  # env = GRF(**config)
  import gfootball.env as football_env
  env = football_env.create_environment('academy_counterattack_hard', representation=Representation.SIMPLE115, number_of_left_players_agent_controls=4)
  env.unwrapped.seed(1)
  obs = env.reset()
  for i in range(args.step):
    a = np.random.randint(0, 19, size=4)
    obs, rew, done, info = env.step(a)
    if isinstance(env, GRF):
      raw_obs = env.raw_state()
    else:
      raw_obs = env.unwrapped.observation()
  
    id = [o['active'] for o in raw_obs]
    print(i, 'active', id)
    if np.all(done):
      print(info)
      # print('Done ball_owned_team', [o['ball_owned_team'] for o in obs])
      break
