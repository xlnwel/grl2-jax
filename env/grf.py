import numpy as np
import gym
f32 = np.float32

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
    shared_policy=False, 
    shared_policy_among_agents=True, 
    score_reward_scale=1, 
    # required configs for grl
    max_episode_steps=3000,
    use_action_mask=False,
    use_sample_mask=False, 
    uid2aid=None,
    uid2gid=None, 
    seed=None, 
    use_idx=False, 
    **kwargs,
  ):
    self.name = env_name
    self.representation = representation
    self.to_render = render
    self.score_reward_scale = score_reward_scale

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
    if uid2gid is None:
      if shared_policy:
        uid2gid = tuple(np.zeros(self.n_left_units, dtype=np.int32)) \
          + tuple(np.ones(self.n_right_units, dtype=np.int32))
      else:
        uid2gid = tuple(np.arange(self.n_units, dtype=np.int32))

    if uid2aid is None:
      if shared_policy_among_agents:
        uid2aid = tuple(np.zeros(self.n_left_units, dtype=np.int32)) \
          + tuple(np.ones(self.n_right_units, dtype=np.int32))
      else:
        uid2aid = tuple(np.arange(self.n_units, dtype=np.int32))

    self.uid2gid = uid2gid
    self.uid2aid = uid2aid
    self.aid2uids = compute_aid2uids(self.uid2aid)
    self.gid2uids = compute_aid2uids(self.uid2gid)
    self.aid2gids = compute_aid2gids(uid2aid, uid2gid)
    self.n_agents = len(self.aid2uids)

    self.max_episode_steps = max_episode_steps

    self.use_action_mask = use_action_mask  # if action mask is used
    self.use_sample_mask = use_sample_mask              # if life mask is used
    self.use_idx = use_idx

    self.action_space = [
      self.env.action_space
      if isinstance(self.env.action_space, gym.spaces.MultiDiscrete) 
      else self.env.action_space 
      for _ in range(self.n_agents)
    ]
    self.action_shape = [() for _ in self.action_space]
    self.action_dim = [19 for _ in range(self.n_agents)]
    self.action_dtype = [np.int32 for _ in self.action_space]
    self.is_action_discrete = [True for _ in self.action_space]

    self.observation_space = self.env.observation_space
    self.reward_range = self.env.reward_range
    self.metadata = self.env.metadata
    obs = self.reset()
    self.obs_shape = [{k: v.shape[-1:] for k, v in o.items()} for o in obs]
    self.obs_dtype = [{k: v.dtype for k, v in o.items()} for o in obs]

    # The following stats should be updated in self.step and be reset in self.reset
    # The episodic score we use to evaluate agent's performance. It excludes shaped rewards
    self._score = np.zeros(self.n_units, dtype=f32)
    # The accumulated episodic rewards we give to the agent. It includes shaped rewards
    self._dense_score = np.zeros(self.n_units, dtype=f32)
    # The length of the episode
    self._epslen = 0
    self._left_score = np.zeros(self.n_units, dtype=f32)
    self._right_score = np.zeros(self.n_units, dtype=f32)

    self._prev_action = [-1 for _ in self.action_dim]
    self._consecutive_action = np.zeros(self.n_units, bool)

    self._checkpoint_reward = .1
    self._num_checkpoints = 10
    self._collected_checkpoints = [0, 0]

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

    self._prev_action = [-1 for _ in self.action_dim]
    self._consecutive_action = np.zeros(self.n_units, bool)
    self._collected_checkpoints = [0, 0]

    # return [{'obs': o[None], 'global_state': o[None]} for o in obs]
    return self._get_obs()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)

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
      self._score = diff_score > 0
    dones = np.tile(done, self.n_units)

    self._prev_action = action
    info = {
      'score': self._score,
      'dense_score': self._dense_score,
      'left_score': self._left_score,
      'right_score': self._right_score,
      'diff_score': diff_score,
      'win_score': diff_score > 0,
      # 'non_loss_score': diff_score >= 0,
      # 'consecutive_action': self._consecutive_action,
      'checkpoint_score': self._ckpt_score,
      'epslen': self._epslen,
      'game_over': done
    }

    agent_obs = self._get_obs()
    # agent_obs = [{'obs': o[None], 'global_state': o[None]} for o in obs]
    agent_rewards = [np.reshape(reward[uids], -1) for uids in self.gid2uids]
    agent_dones = [np.reshape(dones[uids], -1) for uids in self.gid2uids]

    return agent_obs, agent_rewards, agent_dones, info

  def render(self):
    if not self.to_render:
      self.env.render(mode='rgb_array')
      self.to_render = True
    obs = self.raw_state()[0]
    return obs['frame']

  def close(self):
    return self.env.close()

  def _get_obs(self):
    agent_obs = [dict(
      obs=np.concatenate([self.get_state(u >= self.n_left_units, u) for u in uids], 0), 
      global_state=np.concatenate([self.get_state(u >= self.n_left_units) for u in uids], 0), 
    ) for uids in self.gid2uids]
    if self.use_idx:
      for o, uids in zip(agent_obs, self.gid2uids):
        o['idx'] = np.eye(len(uids), dtype=f32)

    return agent_obs

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
    active = [0] * 11
    if uid is None:
      for obs in raw_state:
        if obs["active"] != -1:
          active[obs["active"]] = 1
    else:
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
      np.testing.assert_allclose(reward, np.mean(reward))
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
      np.testing.assert_allclose(reward[0], np.mean(reward[0]))
      np.testing.assert_allclose(reward[1], np.mean(reward[1]))
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
      'env_name': 'academy_pass_and_shoot_with_keeper',
      'representation': 'simple115v2',
      'write_full_episode_dumps': True, 
      'write_video': True, 
      'render': True, 
      'rewards': 'scoring,checkpoints', 
      'control_left': args.left,
      'control_right': args.right,
      'shared_policy': True, 
      'use_action_mask':True, 
      'uid2aid': None,
      'use_idx': True,
      'use_hidden': False, 
      'agentwise_global_state': False, 
      'render': False, 
      'selected_agents': False, 
      'seed': 1
  }

  import numpy as np
  np.random.seed(0)
  env = GRF(**config)
  left = args.left
  obs_left = []
  obs_right = []
  obs = env.reset()
  env.seed(1)
  obs = env.raw_state()
  ids = np.array([o['active'] for o in obs])
  for i in range(args.step):
    a = env.random_action()
    obs, rew, done, info = env.step(a)
    print(i, 'reward', rew, 'done', done)
    obs = env.raw_state()
    new_ids = np.array([o['active'] for o in obs])
    # new_ids = np.array([o['obs'][0, -13+i] for i, o in enumerate(obs)])
    # np.testing.assert_equal(ids, new_ids)
    # print('ball_owned_team', [o['ball_owned_team'] for o in obs])
    # print(obs[0]['obs'].reshape(-1, 5))
    if np.all(done):
      print(info)
      # print('Done ball_owned_team', [o['ball_owned_team'] for o in obs])
      break
