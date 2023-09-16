import random
import numpy as np
import gym

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from env.utils import compute_aid2uids
from core.typing import AttrDict2dict


REW_SHAPING_PARAMS = {
  "PLACEMENT_IN_POT_REW": 1,
  "DISH_PICKUP_REWARD": 0,
  "SOUP_PICKUP_REWARD": 0,
  "DISH_DISP_DISTANCE_REW": 0,
  "POT_DISTANCE_REW": 0,
  "SOUP_DISTANCE_REW": 0
}


def compute_pots_ingradients(terrain):
  pots_pos = []
  n_pots = 0
  n_onions = 0
  n_tomatoes = 0
  for y, row in enumerate(terrain):
    for x, i in enumerate(row):
      if i == 'P':
        n_pots += 1
        pots_pos.append((x, y))
      elif i == 'O':
        n_onions += 1
      elif i == 'T':
        n_tomatoes += 1
  assert len(pots_pos) == n_pots
  return pots_pos, n_pots, n_onions, n_tomatoes


class Overcooked:
  def __init__(self, config):
    config = AttrDict2dict(config)
    self.name = config['env_name'].split('-', 1)[-1]
    self._mdp = OvercookedGridworld.from_layout_name(
      layout_name=self.name, **config.get('layout_params', {}), rew_shaping_params=REW_SHAPING_PARAMS)
    self._env = OvercookedEnv.from_mdp(self._mdp, horizon=config['max_episode_steps'], info_level=0)

    self.max_episode_steps = config['max_episode_steps']
    self.dense_reward = config.get('dense_reward', False)
    self._add_goal = config.get('add_goal', False)
    self._featurize = config.get('featurize', False)
    if self._featurize:
      mlp = MediumLevelActionManager.from_pickle_or_compute(self._mdp, NO_COUNTERS_PARAMS)
      self.featurize_fn = lambda x: np.stack(self._mdp.featurize_state(x, mlp))

    self.pots_pos, self.n_pots, self.n_onions, self.n_tomatoes = compute_pots_ingradients(self._mdp.terrain_mtx)
    self.goal_size = self.n_pots * 2

    self.uid2aid = config.get('uid2aid', [0, 1])
    self.aid2uids = compute_aid2uids(self.uid2aid)
    self.n_units = len(self.uid2aid)
    self.n_agents = len(self.aid2uids)

    self.action_space = [gym.spaces.Discrete(len(Action.ALL_ACTIONS)) 
      for _ in range(self.n_agents)]
    self.action_shape = [a.shape for a in self.action_space]
    self.action_dim = [a.n for a in self.action_space]
    self.action_dtype = [np.int32
      for _ in range(self.n_agents)]
    self.is_action_discrete = [True for _ in range(self.n_agents)]

    self.obs_shape = self._get_observation_shape()
    self.obs_dtype = self._get_observation_dtype()

  def seed(self, seed):
    random.seed(seed)
    np.random.seed(seed)

  def get_screen(self, **kwargs):
    """
    Standard way to view the state of an esnvironment programatically
    is just to print the Env object
    """
    return self._env.__repr__()
  
  def random_action(self):
    return np.random.randint(0, self.action_dim, self.n_units)

  def _get_observation_shape(self):
    dummy_state = self._mdp.get_standard_start_state()
    if self._featurize:
      shape = self._env.featurize_state_mdp(dummy_state)[0].shape
      obs_shape = [dict(
        obs=shape,
        global_state=shape, 
      ) for _ in self.aid2uids]
    else:
      shape = self._env.lossless_state_encoding_mdp(dummy_state)[0].shape
      obs_shape = [dict(
        obs=shape,
      ) for _ in self.aid2uids]
    if self._add_goal:
      for shape in obs_shape:
        shape['goal'] = (self.goal_size,)
    return obs_shape

  def _get_observation_dtype(self):
    if self._featurize:
      obs_dtype = [dict(
        obs=np.float32,
        global_state=np.float32,
      ) for _ in self.aid2uids]
    else:
      obs_dtype = [dict(
        obs=np.float32,
      ) for _ in self.aid2uids]
    if self._add_goal:
      for dtype in obs_dtype:
        dtype['goal'] = np.float32
    return obs_dtype

  def reset(self):
    self._env.reset()
    obs = self._get_obs(self._env.state)
    self._score = np.zeros(self.n_units, dtype=np.float32)
    self._dense_score = np.zeros(self.n_units, dtype=np.float32)
    self._epslen = 0

    return obs

  def step(self, action):
    assert len(action) == 2, action
    real_action = Action.ALL_ACTIONS[action[0]], Action.ALL_ACTIONS[action[1]]
    state, reward, done, info = self._env.step(real_action)
    rewards = reward * np.ones(self.n_units, np.float32)
    self._score += rewards
    self._epslen += 1
    if self.dense_reward:
      dense_reward = max(info['shaped_r_by_agent'])
      rewards += dense_reward * np.ones(self.n_units, np.float32)
    # else:
    #   print(reward, info['sparse_r_by_agent'])
    self._dense_score += rewards
    obs = self._get_obs(state, action)
    dones = done * np.ones(self.n_units, np.float32)
    info = dict(
      score=self._score,
      epslen=self._epslen,
      dense_score=self._dense_score,
      game_over=done,
    )

    rewards = [rewards[uids] for uids in self.aid2uids]
    dones = [dones[uids] for uids in self.aid2uids]

    return obs, rewards, dones, info

  def _get_obs(self, state, action=None):
    if self._featurize:
      obs = self._env.featurize_state_mdp(state)
      obs = [np.expand_dims(o, 0).astype(np.float32) for o in obs]
      obs = [dict(
        obs=o, 
        global_state=o, 
      ) for o in obs]
    else:
      obs = [dict(
        obs=np.expand_dims(o, 0).astype(np.float32)) 
        for o in self._env.lossless_state_encoding_mdp(state)]
    if self._add_goal:
      goal = self._get_pots_status()
      for o in obs:
        o['goal'] = np.expand_dims(goal, 0)
    return obs

  def _get_pots_status(self):
    goal = np.ones(self.goal_size, np.float32)
    for i, pos in enumerate(self.pots_pos):
      if pos in self._env.state.objects:
        soup = self._env.state.objects[pos]
        for x in soup.ingredients:
          if x == 'tomato':
            goal[2*i] -= 1
          elif x == 'onion':
            goal[2*i+1] -= 1
    return goal

  def close(self):
    pass


if __name__ == '__main__':
  def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--interactive', '-i', action='store_true')
    args = parser.parse_args()
    return args
  args = parse_args()
  config = dict(
    env_name=args.env,
    max_episode_steps=400,
    dense_reward=True,
    featurize=True,
    add_goal=False
    # layout_params={
    #   'onion_time': 1,
    # }
  )
  def action2char(action):
    dic = {
      'w': (0, -1),
      's': (0, 1),
      'a': (-1, 0),
      'd': (1, 0),
      'q': (0, 0),
      'e': 'interact',
    }
    a1, a2 = dic[action[0]], dic[action[1]]
    return Action.ACTION_TO_CHAR[a1], Action.ACTION_TO_CHAR[a2]
  def action2array(action):
    dic = {
      'w': 0,
      's': 1,
      'a': 3,
      'd': 2,
      'q': 4,
      'e': 5,
    }
    return np.array([[dic[action[0]]], [dic[action[1]]]])

  env = Overcooked(config)
  obs = env.reset()
  print(obs[0]['obs'].max(), obs[0]['obs'].min())
  d = False
  while not np.all(d):
    # print(env.get_screen())
    if args.interactive:
      a = input('action: ').split(' ')
    else:
      a = env.random_action()
    o, r, d, i = env.step(a)
    # print(o['goal'])
    print("Curr reward: (sparse)", i['sparse_reward'], "\t(dense)", i['dense_reward'])
    print('Reward', r)
