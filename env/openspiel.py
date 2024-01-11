import random
import numpy as np
import gym

from open_spiel.python import rl_environment
from env.utils import *


class OpenSpiel:
  def __init__(
    self, 
    env_name, 
    uid2aid=[0, 1], 
    uid2gid=[0, 1], 
    **kwargs, 
  ):
    self.env = env = rl_environment.Environment(
      env_name, 
      enable_legality_check=True
    )
    self.env.seed(kwargs['seed'])
    self.game = self.env.game
    self.uid2aid = uid2aid
    self.uid2gid = uid2gid
    self.aid2uids = compute_aid2uids(self.uid2aid)
    self.gid2uids = compute_aid2uids(self.uid2gid)
    self.aid2gids = compute_aid2gids(uid2aid, uid2gid)
    self.n_units = len(self.uid2aid)
    self.n_agents = len(self.aid2uids)

    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    self.action_space = [
      gym.spaces.Discrete(num_actions) for _ in self.aid2uids
    ]
    self.action_shape = [a.shape for a in self.action_space]
    self.action_dim = [a.n for a in self.action_space]
    self.action_dtype = [np.int32 for _ in self.action_space]
    self.is_action_discrete = [isinstance(a, gym.spaces.Discrete) for a in self.action_space]

    self.obs_shape = [dict(
      obs=(state_size, ), 
      global_state=(state_size, ), 
      action_mask=(ad, )
    ) for ad in self.action_dim]
    self.obs_dtype = [dict(
      obs=np.float32, 
      global_state=np.float32, 
      action_mask=bool
    ) for _ in range(self.n_agents)]

    self.use_action_mask = True

  @property
  def current_player(self):
    return self._current_player

  def random_action(self):
    action = [random.choice(self._time_step.observations['legal_actions'][self._current_player])]
    assert action[0] in self._time_step.observations['legal_actions'][self._current_player], \
      (action, self._time_step.observations['legal_actions'][self._current_player])
    return action
    

  def seed(self, seed=None):
    self.env.seed(seed)

  def reset(self):
    self._time_step = self.env.reset()
    self._current_player = self._time_step.observations['current_player']

    return self.get_obs(self._time_step)

  def step(self, action):
    action = action[0]['action']
    assert action[0] in self._time_step.observations['legal_actions'][self._current_player], \
      (action, self._time_step.observations['legal_actions'][self._current_player])
    self._time_step = self.env.step(action)
    if self._time_step.observations['current_player'] >= 0:
      self._current_player = self._time_step.observations['current_player']

    rewards = np.array(self._time_step.rewards)
    discounts = np.array(self._time_step.discounts, dtype=np.float32)

    if self._time_step.last():
      assert np.all(discounts == 0), discounts

    info = {
      'game_over': self._time_step.last()
    }

    obs = self.get_obs(self._time_step)
    assert obs['action_mask'].shape[-1] == 3, obs['action_mask'].shape

    return obs, rewards, discounts, info

  def get_obs(self, time_step):
    uid = max(time_step.observations['current_player'], 0)
    info_state = np.array(time_step.observations['info_state'][uid], dtype=np.float32)
    info_state = np.expand_dims(info_state, 0)
    action_mask = np.zeros((1, self.action_dim[uid]), bool)
    legal_action = time_step.observations['legal_actions'][uid]
    action_mask[0, legal_action] = 1
    obs = dict(
      obs=info_state, 
      global_state=info_state, 
      action_mask=action_mask,
      uid=uid
    )

    return obs

  def close(self):
    return


if __name__ == "__main__":
  config = dict(
    env_name='spiel-leduc_poker',
    squeeze_keys=['uid'], 
    uid2aid=[0, 1],
    seed=0, 
    n_envs=2
  )
  from env.func import create_env
  random.seed(config['seed'])
  env = create_env(config)
  done = np.zeros((config['n_envs'], len(config['uid2aid'])), bool)
  reward = np.zeros((config['n_envs'], len(config['uid2aid'])))
  for i in range(10):
    print('iteration', i)
    a = env.random_action()
    agent_outs = env.step(a)
    print('action\t', np.squeeze(a))
    for pid, out in enumerate(agent_outs):
      if len(out.obs) == 0:
        continue
      print(out.obs)
      print(out.reward)
      