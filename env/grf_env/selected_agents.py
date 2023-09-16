import numpy as np
import gym
import gfootball.env as football_env


class Representation:
  RAW='raw'
  CUSTOM='custom'
  MAT='mat'
  SIMPLE115='simple115v2'


class SelectedAgents(gym.Wrapper):
  def __init__(
    self, 
    env_name,
    representation=Representation.SIMPLE115,
    rewards='scoring,checkpoints',
    write_goal_dumps=False,
    write_full_episode_dumps=False,
    render=False,
    write_video=False,
    dump_frequency=1000,
    logdir='results/grf',
    extra_players=None,
    control_left=True, 
    control_right=False, 
  ):
    n_left_units = 0
    n_right_units = 0
    self._left_controlled_units = []
    self._right_controlled_units = []
    if env_name.endswith('single_agent'):
      if control_left:
        n_left_units = 1
        self._left_controlled_units = [0]
      if control_right:
        n_right_units = 1
        self._right_controlled_units = [0]
      env_name = env_name.replace('_single_agent', '')
    else:
      if env_name == 'academy_pass_and_shoot_with_keeper':
        if control_left:
          self._left_controlled_units = [1, 2]
          n_left_units = 3
        if control_right:
          self._right_controlled_units = [0, 1]
          n_right_units = 2
      elif env_name == 'academy_run_pass_and_shoot_with_keeper':
          self._controlled_units = [1, 2]
          n_left_units = 3
      elif env_name == 'academy_3_vs_1_with_keeper':
          self._controlled_units = [1, 2, 3]
          n_left_units = 4
      elif env_name == 'academy_corner':
          self._controlled_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
          n_left_units = 11
      else:
          self._controlled_units = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
          n_left_units = 11
    self.env_name = env_name

    other_config_options = {'action_set':'v2'}
    self.env = football_env.create_environment(
      env_name, 
      representation=representation,
      rewards=rewards,
      write_goal_dumps=write_goal_dumps,
      write_full_episode_dumps=write_full_episode_dumps,
      render=render,
      write_video=write_video,
      dump_frequency=dump_frequency,
      logdir=logdir,
      extra_players=extra_players,
      number_of_left_players_agent_controls=n_left_units,
      number_of_right_players_agent_controls=n_right_units,
      other_config_options=other_config_options, 
    )
    super().__init__(self.env)

    self.action_dim = 20
    self.n_left_units = n_left_units
    self.n_right_units = n_right_units
    self.n_units = n_left_units + n_right_units
    self.n_left_controlled_units = len(self.left_controlled_units)
    self.n_right_controlled_units = len(self.right_controlled_units)
    self.n_controlled_units = self.n_left_controlled_units + self.n_right_controlled_units

  @property
  def left_controlled_units(self):
    if self._left_controlled_units is None:
      states = self.raw_state(0)
      return [states[i]['active'] for i in range(self.n_left_units)]
    else:
      return self._left_controlled_units

  @property
  def right_controlled_units(self):
    if self._right_controlled_units is None:
      states = self.raw_state(1)
      return [states[i]['active'] for i in range(self.n_right_units)]
    else:
      return self._right_controlled_units

  def random_action(self):
    actions = []
    for _ in range(self.n_left_controlled_units):
      actions.append(np.random.randint(self.action_dim))
    for _ in range(self.n_right_controlled_units):
      actions.append(np.random.randint(self.action_dim))
    return actions

  def reset(self):
    obs = self.env.reset()
    obs = self.get_controlled_players_data(obs)
    return obs

  def step(self, action):
    assert len(action) == self.n_controlled_units, (action, self.n_controlled_units)
    action = self.fill_actions(action)
    obs, reward, done, info = self.env.step(action)
    obs = self.get_controlled_players_data(obs)
    reward = self.get_controlled_players_data(reward)
    
    return obs, reward, done, info

  def fill_actions(self, action):
    actions = []
    cid = 0
    for i in range(self.n_left_units):
      if i in self._left_controlled_units:
        actions.append(action[cid])
        cid += 1
      else:
        actions.append(19)
    for i in range(self.n_right_units):
      if i in self._right_controlled_units:
        actions.append(action[cid])
        cid += 1
      else:
        actions.append(19)
    return actions

  def get_controlled_players_data(self, data):
    assert len(data) == self.n_units, (len(data), self.n_units)
    if self.n_right_units > 0:
      left_data = data[:self.n_left_units]
      right_data = data[self.n_left_units:]
      left_data = np.asarray([left_data[i] for i in self.left_controlled_units])
      right_data = np.asarray([right_data[i] for i in self.right_controlled_units])
      return [left_data, right_data]
    else:
      data = np.asarray([data[i] for i in self.left_controlled_units])
    return data

  def raw_state(self, side=None):
    s = self.env.unwrapped.observation()
    assert len(s) == self.n_units, (len(s), self.n_units)
    if side is None:
      return s
    elif side == 0:
      return [s[i] for i in range(self.n_left_units)]
    elif side == 1:
      return [s[i] for i in range(self.n_left_units, self.n_units)]
    else:
      raise ValueError(f'Unknown side: {side}')
