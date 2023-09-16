import numpy as np
from core.typing import AttrDict

from tools.feature import xy2tri


def do_flatten(obj):
  """Run flatten on either python list or numpy array."""
  if isinstance(obj, (list, tuple)):
    return np.array(obj).flatten()
  return obj.flatten()


class FeatureEncoder:
  OBS_SHAPE = (242,)
  HIDDEN_SHAPE = (240,)
  ACTION_DIM = 19

  def __init__(
    self, 
    aid2uids, 
    use_idx, 
    use_hidden, 
    use_event, 
    use_action_mask, 
    agentwise_global_state
  ):
    self.aid2uids = aid2uids
    assert len(self.aid2uids) in (1, 2), len(self.aid2uids)
    self.num_left = len(self.aid2uids[0])
    self.use_idx = use_idx
    self.use_hidden = use_hidden
    self.use_event = use_event
    self.use_action_mask = use_action_mask
    self.agentwise_global_state = agentwise_global_state

  def get_obs_shape(self, n_agents):
    shape = []
    for i in range(n_agents):
      s = dict(
        obs=self.OBS_SHAPE, 
        global_state=self.OBS_SHAPE if self.agentwise_global_state else self.HIDDEN_SHAPE, 
        prev_reward=(), 
        prev_action=(self.ACTION_DIM,), 
      )
      if self.use_action_mask:
        s['action_mask'] = (self.ACTION_DIM,)
      if self.use_idx:
        s['idx'] = (len(self.aid2uids[i]),)
      if self.use_hidden:
        s['hidden_state'] = self.HIDDEN_SHAPE
      if self.use_event:
        s['event'] = (3,)
      shape.append(s)

    return shape

  def get_obs_dtype(self, n_agents):
    dtype = []
    for _ in range(n_agents):
      d = dict(
        obs=np.float32, 
        global_state=np.float32, 
        prev_reward=np.float32, 
        prev_action=np.float32, 
      )
      if self.use_action_mask:
        d['action_mask'] = bool
      if self.use_idx:
        d['idx'] = np.float32
      if self.use_hidden:
        d['hidden_state'] = np.float32
      if self.use_event:
        d['event'] = np.float32
      dtype.append(d)

    return dtype

  def construct_observations(self, observations, action, reward):
    obs_array = []
    hidden_array = []
    idx_array = []
    for i, o in enumerate(observations):
      side = i >= self.num_left
      uid = i - side * self.num_left
      obs_array.append(self.get_obs(o, uid))
      if self.use_hidden or not self.agentwise_global_state:
        hidden_array.append(self.get_hidden_state(o))
      if self.use_idx:
        idx_array.append(self._encode_idx(side, uid))
    obs_array = np.stack(obs_array)
    final_obs = [dict(
      obs=obs_array[uids], 
      prev_reward=reward[aid], 
      prev_action=action[aid]
    ) for aid, uids in enumerate(self.aid2uids)]
    if self.use_idx:
      for obs, uids in zip(final_obs, self.aid2uids):
        obs['idx'] = np.array([idx_array[i] for i in uids])
    if self.use_hidden or not self.agentwise_global_state:
      hidden_array = np.stack(hidden_array)
    if self.agentwise_global_state:
      for obs in final_obs:
        obs['global_state'] = obs['obs'].copy()
    else:
      for uids, obs in zip(self.aid2uids, final_obs):
        obs['global_state'] = hidden_array[uids].copy()
    if self.use_hidden:
      for uids, obs in zip(self.aid2uids, final_obs):
        obs['hidden_state'] = hidden_array[uids]

    return final_obs      

  def get_obs(self, obs, player_num, flatten=True):
    # if player_num is None:
    #   player_num = obs["active"]
    # # print(f'aid: {side}, uid: {player_num}')
    # assert player_num == obs['active'], (player_num, obs['active'], obs['left_team_active'])

    # player info
    observation = AttrDict()
    team = 'left_team'
    player_pos_x, player_pos_y = obs[team][player_num]
    player_pos = obs[team][player_num]   # 2
    player_direction = obs[f"{team}_direction"][player_num] # 2
    player_role = obs["left_team_roles"][player_num]
    player_role = self._encode_role(player_role)  # 10
    player_tired = obs[f"{team}_tired_factor"][player_num]
    is_dribbling = obs["sticky_actions"][9]
    is_sprinting = obs["sticky_actions"][8]
    player_prop = np.array([player_tired, is_dribbling, is_sprinting])  # 3
    observation.player_pos = player_pos
    observation.player_speed = player_direction
    observation.player_role = player_role
    observation.player_property = player_prop
    # assert len(player_state) == 18, len(player_state)

    ball_x, ball_y, ball_z = obs["ball"]
    ball_x_relative = ball_x - player_pos_x
    ball_y_relative = ball_y - player_pos_y
    ball_rel_pos = np.array([ball_y_relative, ball_x_relative]) # 3
    ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
    ball_speed = np.array([ball_y_speed, ball_x_speed]) # 3
    ball_owned_team = self._encode_ball_owned_team(obs)
    ball_owned_player = obs['ball_owned_player'] == player_num
    ball_zone = self._encode_ball_zone(ball_x, ball_y)  # 6
    observation.ball_pos = ball_rel_pos
    observation.ball_speed = ball_speed
    observation.ball_owned_team = ball_owned_team
    observation.ball_owned_player = [ball_owned_player]
    observation.ball_zone = ball_zone

    obs_left_team = np.delete(obs["left_team"], player_num, axis=0)
    obs_left_team_direction = np.delete(
      obs["left_team_direction"], player_num, axis=0
    )
    left_team_tired = np.delete(
      obs["left_team_tired_factor"], player_num, axis=0
    )
    left_team_pos = obs_left_team - player_pos
    observation.left_team_pos = left_team_pos.reshape(-1)
    observation.left_team_speed = obs_left_team_direction.reshape(-1)
    observation.left_team_tired = left_team_tired

    obs_right_team = np.array(obs["right_team"])
    obs_right_team_direction = np.array(obs["right_team_direction"])
    right_team_tired = np.array(obs["right_team_tired_factor"])
    right_team_pos = obs_right_team - player_pos
    observation.right_team_pos = right_team_pos.reshape(-1)
    observation.right_team_speed = obs_right_team_direction.reshape(-1)
    observation.right_team_tired = right_team_tired

    observation.game_mode = self._encode_game_mode(obs)
    
    if flatten:
      keys = sorted(observation)
      observation = np.concatenate([observation[k] for k in keys])

    return observation

  def get_hidden_state(self, obs, flatten=True):
    # player info
    hidden_state = AttrDict()

    hidden_state.left_team_pos = obs['left_team'].reshape(-1)
    hidden_state.right_team_pos = obs['right_team'].reshape(-1)
    hidden_state.left_team_speed = obs['left_team_direction'].reshape(-1)
    hidden_state.right_team_speed = obs['right_team_direction'].reshape(-1)
    hidden_state.left_team_tired = obs['left_team_tired_factor']
    hidden_state.right_team_tired = obs['right_team_tired_factor']
    
    ball = obs['ball'][:2]
    hidden_state.ball_pos = ball
    hidden_state.ball_zone = self._encode_ball_zone(*ball)
    hidden_state.ball_speed = self._encode_ball_speed(obs)
    hidden_state.ball_owned_team = self._encode_ball_owned_team(obs)
    hidden_state.ball_owned_player = self._encode_ball_owned_player(obs)

    hidden_state.game_mode = self._encode_game_mode(obs)
    # from tools.display import print_dict_info
    # print_dict_info(hidden_state)
    if flatten:
      keys = sorted(hidden_state)
      hidden_state = np.concatenate([hidden_state[k] for k in keys])

    return hidden_state

  def _encode_ball_speed(self, obs):
    ball_x_speed, ball_y_speed = obs['ball_direction'][:2]
    ball_speed = xy2tri(ball_y_speed, ball_x_speed, return_length=True)
    return np.array(ball_speed)

  def _encode_ball_owned_team(self, obs):
    ball_owned_team = np.zeros(3)
    ball_owned_team[obs['ball_owned_team']] = 1
    return ball_owned_team

  def _encode_ball_owned_player(self, obs):
    ball_owned_player = np.zeros(12)
    ball_owned_player[obs['ball_owned_player']] = 1
    return ball_owned_player

  def _encode_ball_zone(self, ball_x, ball_y):
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
      -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
      ball_zone = [1.0, 0, 0, 0, 0, 0]
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
      -END_Y < ball_y and ball_y < END_Y
    ):
      ball_zone = [0, 1.0, 0, 0, 0, 0]
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
      -END_Y < ball_y and ball_y < END_Y
    ):
      ball_zone = [0, 0, 1.0, 0, 0, 0]
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
      -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
      ball_zone = [0, 0, 0, 1.0, 0, 0]
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
      -END_Y < ball_y and ball_y < END_Y
    ):
      ball_zone = [0, 0, 0, 0, 1.0, 0]
    else:
      ball_zone = [0, 0, 0, 0, 0, 1.0]
    return np.array(ball_zone)

  def _encode_score(self, obs):
    score_diff = np.clip(obs['score'][0] - obs['score'][1], -2, 2) + 2
    score = np.zeros(5)
    score[score_diff] = 1
    return score

  def _encode_steps_left(self, obs):
    steps_left = [0] * 5
    steps_left[-1 if obs['steps_left'] else obs['steps_left'] // 600 ] = 1
    return steps_left

  def _encode_game_mode(self, obs):
    game_mode = np.zeros(7)
    game_mode[obs['game_mode']] = 1
    return game_mode

  def _encode_idx(self, side, player_num):
    role = np.zeros(len(self.aid2uids[side]))
    role[player_num] = 1.0
    return role

  def _encode_role(self, role_num):
    result = np.zeros(10)
    result[role_num] = 1.0
    return np.array(result)

  def get_avail_action(self, obs, ball_distance):
    avail = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    (
      NO_OP,
      LEFT,
      TOP_LEFT,
      TOP,
      TOP_RIGHT,
      RIGHT,
      BOTTOM_RIGHT,
      BOTTOM,
      BOTTOM_LEFT,
      LONG_PASS,
      HIGH_PASS,
      SHORT_PASS,
      SHOT,
      SPRINT,
      RELEASE_MOVE,
      RELEASE_SPRINT,
      SLIDE,
      DRIBBLE,
      RELEASE_DRIBBLE,
    ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

    if obs["ball_owned_team"] == 1:  # opponents owning ball
      (
        avail[LONG_PASS],
        avail[HIGH_PASS],
        avail[SHORT_PASS],
        avail[SHOT],
        avail[DRIBBLE],
      ) = (0, 0, 0, 0, 0)
      if ball_distance > 0.03:
        avail[SLIDE] = 0
    elif (
      obs["ball_owned_team"] == -1
      and ball_distance > 0.03
      and obs["game_mode"] == 0
    ):  # Ground ball  and far from me
      (
        avail[LONG_PASS],
        avail[HIGH_PASS],
        avail[SHORT_PASS],
        avail[SHOT],
        avail[DRIBBLE],
        avail[SLIDE],
      ) = (0, 0, 0, 0, 0, 0)
    else:  # my team owning ball
      avail[SLIDE] = 0
      if ball_distance > 0.03:
        (
          avail[LONG_PASS],
          avail[HIGH_PASS],
          avail[SHORT_PASS],
          avail[SHOT],
          avail[DRIBBLE],
        ) = (0, 0, 0, 0, 0)

    # Dealing with sticky actions
    sticky_actions = obs["sticky_actions"]
    if sticky_actions[8] == 0:  # sprinting
      avail[RELEASE_SPRINT] = 0

    if sticky_actions[9] == 1:  # dribbling
      avail[SLIDE] = 0
    else:
      avail[RELEASE_DRIBBLE] = 0

    if np.sum(sticky_actions[:8]) == 0:
      avail[RELEASE_MOVE] = 0

    # if too far, no shot
    ball_x, ball_y, _ = obs["ball"]
    if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
      avail[SHOT] = 0
    elif (0.64 <= ball_x and ball_x <= 1.0) and (
      -0.27 <= ball_y and ball_y <= 0.27
    ):
      avail[HIGH_PASS], avail[LONG_PASS] = 0, 0

    if obs["game_mode"] == 2 and ball_x < -0.7:  # Our GoalKick
      avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
      return np.array(avail)

    elif obs["game_mode"] == 4 and ball_x > 0.9:  # Our CornerKick
      avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
      return np.array(avail)

    elif obs["game_mode"] == 6 and ball_x > 0.6:  # Our PenaltyKick
      avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      avail[SHOT] = 1
      return np.array(avail)

    return np.array(avail)


if __name__ == '__main__':
  import gfootball.env as football_env
  from tools.display import print_dict_info, print_dict

  left = 3
  right = 1
  env = football_env.create_environment(
    'academy_3_vs_1_with_keeper', 
    representation='raw', 
    number_of_left_players_agent_controls=left, 
    number_of_right_players_agent_controls=right
  )
  obs = env.reset()
  encoder = FeatureEncoder(
    [[0, 1, 2], [2]], 
    True, 
    True,
    True, 
    True, 
    True,
  )
  for _ in range(29):
    a = np.random.randint(0, 19, left+right)
    o, _, _, _ = env.step(a)
    
  for i, oo in enumerate(o):
    if i == 3:
      print_dict(oo, 'old')
      side = i >= left
      i = i - side * left
      oo = encoder.get_obs(oo, i, False)
      print_dict(oo, 'new')
    # print(i)
    # print('left', [oo['left_team'].shape for oo in o])
    # print('right', [oo['right_team'].shape for oo in o])

  # for i, o in enumerate(obs):
  #   side = i >= left
  #   i = i - side * left
  #   o = encoder.get_obs(o, i, False)
  #   print_dict_info(o)
