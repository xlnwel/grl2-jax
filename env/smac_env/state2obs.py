import jax
import jax.numpy as jnp
import enum
from core.log import do_logging

"""
这部分代码提供将SMAC环境中的状态映射到单个智能体观测的函数。
函数中使用了部分外部的、和环境有关的、不由当前状态完全控制的信息，因此写成
了类的形式并提供相关属性，建议使用时保留主方法，其他内容整合到目标类当中。
"""

class Direction(enum.IntEnum):
  NORTH = 0
  SOUTH = 1
  EAST = 2
  WEST = 3

class StateProjection:
  def __init__(
    self, **env_kwargs
  ):
    self.sanity_check(env_kwargs)
    # load the env configs
    self.obs_all_health = env_kwargs['obs_all_health']
    self.obs_own_health = env_kwargs['obs_own_health']
    self.obs_last_action = env_kwargs['obs_last_action']
    self.obs_pathing_grid = env_kwargs['obs_pathing_grid']
    self.obs_terrain_height = env_kwargs['obs_terrain_height']
    self.obs_timestep_number = env_kwargs['obs_timestep_number']
    self.obs_instead_of_state = env_kwargs['obs_instead_of_state']
    self.state_last_action = env_kwargs['state_last_action']
    self.state_timestep_number = env_kwargs['state_timestep_number']
    self._move_amount = env_kwargs['move_amount']
    # load the env information
    self.shield_bits_ally = env_kwargs['shield_bits_ally']
    self.shield_bits_enemy = env_kwargs['shield_bits_enemy']
    self.unit_type_bits = env_kwargs['unit_type_bits']
    self.sight_range = env_kwargs['sight_range']
    self.shoot_range = env_kwargs['shoot_range']
    self.n_agents, self.n_enemies = env_kwargs['n_agents'], env_kwargs['n_enemies']
    self.n_actions = env_kwargs['n_actions']
    self.map_x, self.map_y = env_kwargs['map_x'], env_kwargs['map_y']
    self.max_distance_x, self.max_distance_y = env_kwargs['max_distance_x'], env_kwargs['max_distance_y']
    self.pathing_grid = env_kwargs['pathing_grid']
    self.map_type = env_kwargs['map_type']

    self.n_actions_move = 4

    if self.map_type == "MMM":
      do_logging('Current map type is MMM. There may exist bugs')

  def parse_state(self, state_vec):
    # parse the state information
    state_nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
    state_nf_en = 4 + self.shield_bits_enemy + self.unit_type_bits
    last_action, state_timestep = None, None

    al_state = state_vec[:state_nf_al * self.n_agents]
    ind = state_nf_al * self.n_agents
    en_state = state_vec[ind:ind + state_nf_en * self.n_enemies]
    ind += state_nf_en * self.n_enemies
    if self.state_last_action:
      last_action = state_vec[ind:ind + self.n_actions * self.n_agents]
      ind += self.n_actions * self.n_agents
    if self.state_timestep_number:
      state_timestep = state_vec[ind:ind + 1]
    al_feats = al_state.split(self.n_agents)  # [n_agents, ally_feats]
    en_feats = en_state.split(self.n_enemies)   # [n_enemies, enemy_feats]
    return {
      'al_feats': al_feats,
      'en_feats': en_feats,
      'last_action': last_action,
      'state_timestep': state_timestep,
    }

  def project(self, state_vec):
    # parse the state information
    state_dict = self.parse_state(state_vec)
    return [self.project_agent(state_dict, agent_id) for agent_id in range(self.n_agents)]
  
  def project_agent(self, state_dict, agent_id):    
    # define some properties about the obersvation
    obs_nf_en = obs_nf_al = 4 + self.unit_type_bits
    if self.obs_all_health:
      obs_nf_al += 1 + self.shield_bits_ally
      obs_nf_en += 1 + self.shield_bits_enemy
    
    if self.obs_last_action:
      obs_nf_al += self.n_actions
    
    obs_nf_own = self.unit_type_bits
    if self.obs_own_health:
      obs_nf_own += 1 + self.shield_bits_ally

    move_feats_len = self.n_actions_move
    if self.obs_pathing_grid:
      move_feats_len += self.n_obs_pathing
    if self.obs_terrain_height:
      move_feats_len += self.n_obs_height

    move_feats = jnp.zeros(move_feats_len, dtype=jnp.float32)
    enemy_feats = jnp.zeros((self.n_enemies, obs_nf_en), dtype=jnp.float32)
    ally_feats = jnp.zeros((self.n_agents - 1, obs_nf_al), dtype=jnp.float32)
    own_feats = jnp.zeros(obs_nf_own, dtype=jnp.float32)

    agent_state_feat = state_dict['al_feats'][agent_id]
    health, weapon, rel_x, rel_y  = agent_state_feat[:4]
    pos_x, pos_y = self.recover_pos(rel_x, rel_y)
    if self.shield_bits_ally > 0:
      own_shield = agent_state_feat[4]
    if self.unit_type_bits > 0:
      own_unit_type = agent_state_feat[-self.unit_type_bits:]

    if health > 0:
      # compute movement feat
      if self.can_move(pos_x, pos_y, Direction.NORTH):
        move_feats[0] = 1
      if self.can_move(pos_x, pos_y, Direction.SOUTH):
        move_feats[1] = 1
      if self.can_move(pos_x, pos_y, Direction.EAST):
        move_feats[2] = 1
      if self.can_move(pos_x, pos_y, Direction.WEST):
        move_feats[3] = 1
      
      if self.obs_pathing_grid or self.obs_terrain_height:
        raise NotImplementedError
      
      # compute enemy features
      for e_id in range(self.n_enemies):
        enemy_state_feat = state_dict['en_feats'][e_id]
        en_health, en_rel_x, en_rel_y = enemy_state_feat[:3]
        if self.shield_bits_enemy > 0:
          en_shield = enemy_state_feat[3]
        if self.unit_type_bits > 0:
          en_unit_type = enemy_state_feat[-self.unit_type_bits:]
        en_pos_x, en_pos_y = self.recover_pos(en_rel_x, en_rel_y)
        dist = self.distance(pos_x, pos_y, en_pos_x, en_pos_y)

        if (
          dist < self.sight_range and en_health > 0
        ):
          # visible and alive
          enemy_feats[e_id, 0] = float(dist <= self.shoot_range)
          enemy_feats[e_id, 1] = dist / self.sight_range
          enemy_feats[e_id, 2] = (en_pos_x - pos_x) / self.sight_range
          enemy_feats[e_id, 3] = (en_pos_y - pos_y) / self.sight_range

          ind = 4
          if self.obs_all_health:
            enemy_feats[e_id, ind] = en_health
            ind += 1
            if self.shield_bits_enemy > 0:
              enemy_feats[e_id, ind] = en_shield
              ind += 1
          if self.unit_type_bits > 0:
            enemy_feats[e_id, ind:] = en_unit_type
      
      # compute ally features
      al_ids = [
        al_id for al_id in range(self.n_agents) if al_id != agent_id
      ]
      for i, al_id in enumerate(al_ids):
        ally_state_feat = state_dict['al_feats'][al_id]
        al_health, al_weapon, al_rel_x, al_rel_y = ally_state_feat[:4]
        al_pos_x, al_pos_y = self.recover_pos(al_rel_x, al_rel_y)
        if self.shield_bits_ally > 0:
          al_shield = ally_state_feat[4]
        if self.unit_type_bits > 0:
          al_unit_type = ally_state_feat[-self.unit_type_bits:]
        dist = self.distance(pos_x, pos_y, al_pos_x, al_pos_y)
        if (
          dist < self.sight_range and al_health > 0
        ):
          ally_feats[i, 0] = 1
          ally_feats[i, 1] = dist / self.sight_range
          ally_feats[i, 2] = (al_pos_x - pos_x) / self.sight_range
          ally_feats[i, 3] = (al_pos_y - pos_y) / self.sight_range

          ind = 4
          if self.obs_all_health:
            ally_feats[i, ind] = al_health
            ind += 1
            if self.shield_bits_ally > 0:
              ally_feats[i, ind] = al_shield
              ind += 1
          if self.unit_type_bits > 0:
            ally_feats[al_id, ind:] = al_unit_type
      
      # own features
      ind = 0
      if self.obs_own_health:
        own_feats[ind] = health
        ind += 1
        if self.shield_bits_ally > 0:
          own_feats[ind] = own_shield
          ind += 1
      if self.unit_type_bits > 0:
        own_feats[ind:] = own_unit_type

    agent_obs = jnp.concatenate(
      (
        move_feats.flatten(),
        enemy_feats.flatten(),
        ally_feats.flatten(),
        own_feats.flatten(),
      )
    )    
    
    if self.obs_timestep_number:
      assert state_dict['state_timestep'] is not None
      agent_obs = jnp.concatenate((agent_obs, state_dict['state_timestep']))

    return agent_obs

  def can_move(self, pos_x, pos_y, direction):
    m = self._move_amount / 2
    
    if direction == Direction.NORTH:
      x, y = int(pos_x), int(pos_y + m)
    elif direction == Direction.SOUTH:
      x, y = int(pos_x), int(pos_y - m)
    elif direction == Direction.EAST:
      x, y = int(pos_x + m), int(pos_y)
    else:
      x, y = int(pos_x - m), int(pos_y)

    check_bounds = lambda x, y: (0 <= x < self.map_x and 0 <= y < self.map_y)
    if check_bounds(x, y) and self.pathing_grid[x, y]:
      return True

    return False
  
  def recover_pos(self, rel_x, rel_y):
    center_x, center_y = self.map_x / 2, self.map_y / 2
    return rel_x * self.max_distance_x + center_x, rel_y * self.max_distance_y + center_y

  @staticmethod 
  def distance(x1, y1, x2, y2):
    return jnp.hypot(x2 - x1, y2 - y1)

  def sanity_check(self, env_kwargs):
    assert env_kwargs['obs_all_health']
    assert env_kwargs['obs_own_health']
    assert env_kwargs['obs_last_action']
    assert not env_kwargs['obs_pathing_grid']
    assert not env_kwargs['obs_terrain_height']
    assert not env_kwargs['obs_timestep_number']
    assert not env_kwargs['obs_instead_of_state']
    assert env_kwargs['state_last_action']
    assert not env_kwargs['state_timestep_number']
  
    
    
