"""
!/usr/bin/python3
-*- coding: utf-8 -*-
@FileName: environment.py
@Time: 2024/4/15 下午3:54
@Author: ZhengtaoCao
@Description: None
"""
# -*- coding: utf-8 -*-
import math
import random
import collections
import numpy as np
import gym

from core.names import DEFAULT_ACTION
from tools.utils import batch_dicts
from tools.display import print_dict, print_dict_info
from envs.battle5v5.config import ADDRESS
from envs.battle5v5.env.env_runner import EnvRunner
from envs.utils import *
from envs.battle5v5.config import config, Agent, BLUE_INFO, RED_INFO, BLUE_FIRE_INFO, RED_FIRE_INFO


def get_info_id(pinfos, id, check_existence=False):
  pids = [j for j, item in enumerate(pinfos) if item['ID'] == id]
  if check_existence and len(pids) == 0:
    return None
  assert len(pids) == 1, pids
  pid = pids[0]
  return pid


def get_uid(infos, id):
  uids = [k for k, item in infos.items() if item['ID'] == id]
  assert len(uids) == 1, uids
  uid = uids[0]
  return uid


class Reason:
  TIMEOUT = 'done for timeout'
  WIN = 'done for win'
  LOSE = 'done for lose'
  OOR = 'done for out of region'
  IN_PROGRESS = 'in progress'


class HuaRu5v5(EnvRunner):
  def __init__(
    self,
    env_name, 
    eid, 
    uid2aid=[0, 0, 0, 0, 0], 
    uid2gid=[0, 0, 0, 0, 0], 
    max_episode_steps=2000, 
    ip='127.0.0.1',  # ADDRESS['ip'] + ":" + str(ADDRESS['port']),
    port=None,  # ADDRESS['ip'] + ":" + str(ADDRESS['port']),
    mode='host', 
    frame_skip=15, 
    lock_reward_scale=1, 
    damage_reward_scale=1, 
    escape_reward_scale=1, 
    attack_reward_scale=1, 
    miss_reward_scale=1, 
    distance_reward_scale=1, 
    border_reward_scale=1, 
    win_reward=10, 
    shared_reward=False,
    **kwargs
  ):
    if eid is None:
      eid = 1
    if port is None:
      port = str(eid)
    address = f'{ip}:{port}'
    self.address = address
    super().__init__(config['agents'], address=address, mode=mode)
    self.lock_reward_scale = lock_reward_scale
    self.damage_reward_scale = damage_reward_scale
    self.escape_reward_scale = escape_reward_scale
    self.attack_reward_scale = attack_reward_scale
    self.miss_reward_scale = miss_reward_scale
    self.distance_reward_scale = distance_reward_scale
    self.border_reward_scale = border_reward_scale
    self.win_reward = win_reward

    self.frame_skip = frame_skip
    
    self.shared_reward = shared_reward

    self.uid2aid = uid2aid  # Unit ID到Aagent ID的映射
    self.uid2gid = uid2gid  # Unit ID到Group ID的映射
    self.aid2uids = compute_aid2uids(self.uid2aid)    # Agent ID到Unit ID的映射
    self.gid2uids = compute_aid2uids(self.uid2gid)    # Group ID到Unit ID的映射
    self.aid2gids = compute_aid2gids(uid2aid, uid2gid)  # Agent ID到Group ID的映射
    self.n_units = len(self.uid2aid)  # Unit个数
    self.n_groups = len(self.gid2uids)   # Group个数
    self.n_agents = len(self.aid2uids)  # Agent个数

    # 观测/动作空间相关定义
    self.observation_space = [{
      'obs': gym.spaces.Box(high=float('inf'), low=0, shape=(151,)),
      'global_state': gym.spaces.Box(high=float('inf'), low=0, shape=(354,)), 
      'sample_mask': gym.spaces.Discrete(2)
    } for _ in range(self.n_groups)]
    self.obs_shape = [{
      k: v.shape for k, v in obs.items()
    } for obs in self.observation_space]
    self.obs_dtype = [{
      k: v.dtype for k, v in obs.items()
    } for obs in self.observation_space]
    n_move_actions = 10
    n_attack_actions = 5
    n_speed_actions = 2
    self.action_space = [{
      DEFAULT_ACTION: gym.spaces.Discrete(n_move_actions + n_attack_actions), 
    } for _ in range(self.n_groups)]
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
    self.use_sample_mask = True
    self.use_action_mask = [{DEFAULT_ACTION: True}]

    self._epslen = 0    # 回合步长
    self._score = np.zeros(self.n_units)  # 最终胜负
    self._dense_score = np.zeros(self.n_units)  # 稠密奖励累积

    self.max_episode_steps = max_episode_steps  # 最大回合步长

    # 初始化智能体，红方智能体是用于转换仿真的指令；蓝方智能体适用于利用代码规则
    # 存取本episode训练的其他信息
    self.shoot_interval_step = 20  # 50000米发射，导弹1000m/s

  def seed(self, seed):
    pass

  def random_action(self):
    action = []
    for i in range(self.n_units):
      avail_actions = self.get_avail_actions(self.msg, i)
      action.append(random.choice([i for i, v in enumerate(avail_actions) if v]))
      assert avail_actions[action[-1]], (avail_actions, action[-1])
    action = {DEFAULT_ACTION: np.stack(action)}
    action = [{k: v[uids] for k, v in action.items()} for uids in self.aid2uids]
    return action

  def reset(self, if_test=False, args=None, cur_time=None):
    """重置仿真环境, 返回初始帧obs"""
    self._epslen = 0
    self._score = np.zeros(self.n_units)
    self._dense_score = np.zeros(self.n_units)
    self._scores = collections.defaultdict(float)
    self._rewards = {}

    for side, agent in self.agents.items():
      agent.reset()
    super().reset()
    msg = super().step([])

    self.red_alive_mask = []
    self.blue_alive_mask = []
    self.shoot_interval = np.zeros([self.n_units, 5])
    
    self.red_agent_loc = {}
    self.agents_speed = {}
    self.agent_speed_history = collections.defaultdict(list)

    if msg["sim_time"] == 0.0:
      msg = super().step([])  # 推动拿到第一帧的obs信息
    parsed_msg = {'agent_pre_loc': self.red_agent_loc,
           'blue_info': BLUE_INFO,
           'red_info': RED_INFO,
           'sim_time': msg['sim_time'],
           'agent_speed': self.agents_speed # 记录当前帧每个智能体的速度，如果已经死亡，则速度为-1
    }

    cmd_list = []
    cmd_list.extend(self.agents[Agent.RED].make_init_cmd())
    cmd_list.extend(self.get_blue_cmd(self.agents[Agent.BLUE], msg))

    self.last_msg = msg
    self.msg = super().step(cmd_list)
    # 将前200步的规则操作都移到reset()中，不输入到算法中，减小探索空间
    # 前200将动作为0传给alo_agent
    rule_actions = [0]*5  # 加一个强规则，前150帧像前飞，减小探索空间
    while self.msg["sim_time"] < 200:
      cmd_list = []
      pinfos = self.msg[Agent.RED]['platforminfos']
      for i in range(5):
        pinfo = pinfos[i]
        self.red_agent_loc[i] = {
          'X': pinfo['X'], 'Y': pinfo['Y'], 'Z': pinfo['Alt'], 
          'heading': pinfo['Heading'], 'pitch': pinfo['Pitch']}
        self.agents_speed[i] = pinfo['Speed']
      parsed_msg = {'agent_pre_loc': self.red_agent_loc,
            'blue_info': BLUE_INFO,
            'red_info': RED_INFO,
            'sim_time': self.msg['sim_time'],
            'agent_speed': self.agents_speed # 记录当前帧每个智能体的速度，如果已经死亡，则速度为-1
      }
      if self.msg["sim_time"] % self.frame_skip == 0:
        cmd_list.extend(self.agents["red"].make_actions(rule_actions, parsed_msg))  # parse_msg里的sim_time应该没更新
        self.last_msg = self.msg  # 2024/06/24 修改，15帧决策时，上一帧的数据应该是20帧之前
      else:
        pass
      cmd_list.extend(self.get_blue_cmd(self.agents["blue"], self.msg))
      self.msg = super().step(cmd_list)

    obs = self.make_obs(self.msg)

    return obs   # list, np

  def step(self, actions):
    """
      逐帧推进仿真引擎, 发送actions, 返回obs
      :param actions: [[0, 2],
              [1, 3],
              ...
              ] 是一个二维数组，规模是(5, 2) 第1列是移动动作，动作空间是9；第2列是攻击动作，动作空间是6
    """
    done_reward, done, done_reason = self.take_action(actions)
    obs = self.make_obs(self.msg)
    reward = self.get_reward(self.last_msg, self.msg)
    reward = reward + done_reward
    # self.reward计算的是累积奖励，没有什么用, cur_reward才是当前帧的奖励
    self._dense_score += reward
    self._score += np.sign(done_reward)
    self._epslen += 1
    for k, v in self._rewards.items():
      self._scores[k] += v

    reward = np.ones(self.n_units) * reward
    done = np.ones(self.n_units) * done

    info = {
      'score': self._score,
      'dense_score': self._dense_score, 
      'epslen': self._epslen, 
      'game_over': np.any(done), 
      'red_left_missile': sum([p['LeftWeapon'] for p in self.msg[Agent.RED]['platforminfos']]), 
      'blue_left_missile': sum([p['LeftWeapon'] for p in self.msg[Agent.BLUE]['platforminfos']]), 
      'timeout_times': done_reason == Reason.TIMEOUT, 
      'win_times': done_reason == Reason.WIN, 
      'lose_times': done_reason == Reason.LOSE, 
      'inprogress_times': done_reason == Reason.IN_PROGRESS, 
      **{f'{k}_speed': np.mean(v) for k, v in self.agent_speed_history.items()}, 
      **{f'{k}_score': v for k, v in self._scores.items()}
    }

    # 结束环境
    return obs, reward, done, info  # self.info

  def get_blue_cmd(self, blue_agents, msg):
    """获取蓝方当前的cmd_list"""
    cur_time = msg["sim_time"]
    cmd_list = super()._agent_step(blue_agents, cur_time, msg[Agent.BLUE])

    return cmd_list

  def take_action(self, action):
    action = action[0][DEFAULT_ACTION]

    for i in range(self.n_units):
      # 拿取每个agent的攻击动作
      am = self.get_avail_actions(self.msg, i)
      assert am[action[i]], (action[i], am)
      if 8 < action[i] < 14:
        assert self.shoot_interval[i, action[i] - 9] == 0, self.shoot_interval[i, action[i] - 9]
        self.shoot_interval[i, action[i] - 9] = self.shoot_interval_step
    non_zero_indices = self.shoot_interval != 0
    self.shoot_interval[non_zero_indices] -= 1
    # 生成蓝方Agents的cmd list
    blue_cmd_list = self.get_blue_cmd(self.agents[Agent.BLUE], self.msg)
    # 将网络输出动作转换为仿真执行指令,给self.red_agents
    cmd_list = []
    parsed_msg = {
      'agent_pre_loc': self.red_agent_loc,
      'blue_info': BLUE_INFO,
      'red_info': RED_INFO,
      'sim_time': self.msg['sim_time'],
      'agent_speed': self.agents_speed # 记录当前帧每个智能体的速度，如果已经死亡，则速度为-1
    }
    cmd_list.extend(self.agents[Agent.RED].make_actions(action, parsed_msg)) # Agents的仿真指令
    # print(f'cmd list: {cmd_list}')
    cmd_list.extend(blue_cmd_list)

    self.last_msg = self.msg
    for _ in range(self.frame_skip):
      self.msg = super().step(cmd_list)
      done, reward, reason = self.get_done(self.msg)
      if done:
        break
      else:
        cmd_list = self.get_blue_cmd(self.agents[Agent.BLUE], self.msg)

    return reward, done, reason

  def make_obs(self, msg):
    obs = {
      'obs': [], 
      'global_state': [], 
      'action_mask': [], 
    }
    for i in range(self.n_units):
      obs['obs'].append(self.get_obs(msg, i))
      obs['global_state'].append(self.get_global_state(msg, i))
      obs['action_mask'].append(self.get_avail_actions(msg, i))
    for k, v in obs.items():
      obs[k] = np.stack(v)
    obs['sample_mask'] = np.array(self.red_alive_mask)
    obs = [{k: v[uids] for k, v in obs.items()} for uids in self.aid2uids]
    for o in obs:
      o['action_mask'] = {DEFAULT_ACTION: o['action_mask']}

    return obs

  def get_obs(self, msg, uid):
    """对原始态势信息进行解析处理，构造状态"""
    pinfos = msg[Agent.RED]['platforminfos']
    agent_id = [0] * self.n_units
    agent_id[uid] = 1
    if not self.red_alive_mask[uid]:
      obs = np.array(agent_id + [0] * (self.obs_shape[0]['obs'][0] - self.n_units))
      self.red_agent_loc[uid] = {
        'X': None, 'Y': None, 'Z': None, 
        'heading': None, 'pitch': None, 'roll': None, 
      }
      self.agents_speed[uid] = 0
      return obs

    """当前Agent还存活"""
    obs = agent_id
    # 获取当前Agent在['platforminfos']中的索引
    pid = get_info_id(pinfos, RED_INFO[uid]['ID'])
    # 拿这个Agent本身的信息
    ptype = pinfos[pid]['Type']
    ptype_oh = [0, 0]
    ptype_oh[ptype-1] = 1
    x = pinfos[pid]['X']
    y = pinfos[pid]['Y']
    z = pinfos[pid]['Alt']
    heading = pinfos[pid]['Heading']
    pitch = pinfos[pid]['Pitch']
    # roll = pinfos[pid]['Roll']
    speed = pinfos[pid]['Speed']
    is_locked = pinfos[pid]['IsLocked']
    leftweapon = pinfos[pid]['LeftWeapon']

    obs += [x, y, z, heading, pitch, speed, leftweapon, is_locked, *ptype_oh]
    self.red_agent_loc[uid] = {
      'X': x, 'Y': y, 'Z': z, 
      'heading': heading, 'pitch': pitch
    }
    self.agents_speed[uid] = speed
    self.agent_speed_history[uid].append(speed)

    # 拿其他队友的信息，对于某个Agent，它的队友固定为1，2，3，4顺序, 除了agent_order以外
    for i in range(5):
      if i == uid:
        continue
      # 先查找这个队友还存在
      if self.red_alive_mask[i]:
        # 说明这个队友还活着，拿到这个队友的索引
        al_id = get_info_id(pinfos, RED_INFO[i]['ID'])
        al_type = pinfos[al_id]['Type']
        al_type_oh = [0, 0]
        al_type_oh[al_type-1] = 1
        al_x = pinfos[al_id]['X'] - x
        al_y = pinfos[al_id]['Y'] - y
        al_z = pinfos[al_id]['Alt'] - z
        al_dist = np.linalg.norm(np.array([al_x, al_y, al_z]))
        al_speed = pinfos[al_id]['Speed']
        al_islocked = pinfos[al_id]['IsLocked']
        al_leftweapon = pinfos[al_id]['LeftWeapon']
        obs += [al_x, al_y, al_z, al_dist, al_speed, al_leftweapon, al_islocked] + al_type_oh
      else:
        # 说明这个队友已经死亡，那么只需要在对应位置上设为0即可
        # 同时将这个队友的死亡ID记录
        # print(f'死亡的队友有: {self.red_death}')
        obs += [0.] * 9
    pinfos = msg[Agent.BLUE]['platforminfos']
    # 拿敌人的状态信息
    for i in range(5):
      if self.blue_alive_mask[i]:
        # 说明这个敌人还活着
        e_id = get_info_id(pinfos, BLUE_INFO[i]['ID'])
        # 找到这个敌人在trackinfos_list中的索引位置
        e_type = pinfos[e_id]['Type']
        e_type_oh = [0, 0]
        e_type_oh[e_type-1] = 1
        e_x = pinfos[e_id]['X'] - x
        e_y = pinfos[e_id]['Y'] - y
        e_z = pinfos[e_id]['Alt'] - z
        e_dist = np.linalg.norm(np.array([e_x, e_y, e_z]))
        e_speed = pinfos[e_id]['Speed']
        e_islocked = pinfos[e_id]['IsLocked']
        obs += [e_x, e_y, e_z, e_dist, e_speed, e_islocked] + e_type_oh
      else:
        obs += [0.] * 8

    # 拿敌人的导弹信息
    # 这里只取对自己有威胁的导弹
    minfos = msg[Agent.RED]['missileinfos']
    if minfos == []:
      obs += [0.] * 60
    else:
      for i in range(12):
        # 一共是12枚导弹，
        # 分别查看这12枚导弹是否出现了
        minfo = list(filter(lambda x: x['Name'] == BLUE_FIRE_INFO[i]['Name'], minfos))
        if not minfo:
          obs += [0.] * 5
          continue
        minfo = minfo[0]
        # 如果出现了，查看是否锁定了自己？
        if minfo['EngageTargetID'] == RED_INFO[uid]['ID']:
          # 说明这枚弹已经出现了
          # 拿到这枚弹的信息
          # 锁定了自己
          m_x = minfo['X'] - x
          m_y = minfo['Y'] - y
          m_z = minfo['Alt'] - z
          m_dist = np.linalg.norm(np.array([m_x, m_y, m_z]))
          m_speed = minfo['Speed']
          obs += [m_x, m_y, m_z, m_dist, m_speed]
        else:
          # 说明这枚弹没有出现
          obs += [0.] * 5

    return obs

  def get_global_state(self, msg, uid):
    def get_ego_info(msg, uid):
      pinfos = msg[Agent.RED]['platforminfos']
      pid = get_info_id(pinfos, RED_INFO[uid]['ID'])
      ptype = pinfos[pid]['Type']
      ptype_oh = [0, 0]
      ptype_oh[ptype-1] = 1
      x = pinfos[pid]['X']
      y = pinfos[pid]['Y']
      z = pinfos[pid]['Alt']
      heading = pinfos[pid]['Heading']
      pitch = pinfos[pid]['Pitch']
      # roll = pinfos[pid]['Roll']
      assert abs(pitch) < 3.15, pitch
      speed = pinfos[pid]['Speed']
      is_locked = pinfos[pid]['IsLocked']
      leftweapon = pinfos[pid]['LeftWeapon']
      assert abs(x) < 1.6e5, x
      assert abs(y) < 1.6e5, y
      assert abs(z) < 2e4, z
      assert abs(heading) < 3.15, heading
      assert abs(speed) <= 410, speed
      assert abs(is_locked) <= 1, is_locked
      assert abs(leftweapon) <= 4, leftweapon
      gs = [x, y, z, heading, pitch, speed, leftweapon, is_locked, *ptype_oh]
      return gs, x, y, z

    def get_info(msg, side, x, y, z):
      pinfos = msg[side]['platforminfos']
      gs = []

      info_dict = RED_INFO if side == Agent.RED else BLUE_INFO
      alive_mask = self.red_alive_mask if side == Agent.RED else self.blue_alive_mask
      # 获取当前Agent在['platforminfos']中的索引
      for i in range(5):
        if side == Agent.RED and i == uid:
          continue
        # 先查找这个队友还存在
        if alive_mask[i]:
          # 说明这个队友还活着，拿到这个队友的索引
          pid = get_info_id(pinfos, info_dict[i]['ID'])
          ptype = pinfos[pid]['Type']
          ptype_oh = [0, 0]
          ptype_oh[ptype-1] = 1
          px = pinfos[pid]['X'] - x
          py = pinfos[pid]['Y'] - y
          pz = pinfos[pid]['Alt'] - z
          dist = np.linalg.norm(np.array([px, py, pz]))
          heading = pinfos[pid]['Heading']
          pitch = pinfos[pid]['Pitch']
          speed = pinfos[pid]['Speed']
          islocked = pinfos[pid]['IsLocked']
          leftweapon = pinfos[pid]['LeftWeapon']
          gs += [px, py, pz, dist, heading, pitch, speed, leftweapon, islocked] + ptype_oh
        else:
          # 说明这个队友已经死亡，那么只需要在对应位置上设为0即可
          # 同时将这个队友的死亡ID记录
          # print(f'死亡的队友有: {self.red_death}')
          gs += [0.] * 11
      return gs

    def get_miss_info(msg, side):
      gs = []
      info_dict = RED_INFO if side == Agent.RED else BLUE_INFO
      fire_dict = BLUE_FIRE_INFO if side == Agent.RED else RED_FIRE_INFO
      pinfos = msg[side]['platforminfos']
      minfos = msg[side]['missileinfos']
      for i in range(12):
        # 一共是12枚导弹，
        # 分别查看这12枚导弹是否出现了
        minfo = list(filter(lambda x: x['Name'] == fire_dict[i]['Name'], minfos))
        if not minfo:
          gs += [0.] * 10
          continue
        minfo = minfo[0]
        tid = minfo['EngageTargetID']
        tid = get_uid(info_dict, tid)
        tid_oh = [0] * 5
        tid_oh[tid] = 1
        pids = [j for j, item in enumerate(pinfos) if item['ID'] == tid]
        if not pids:
          gs += [0.] * 10
          continue
        pid = pids[0]
        x = pinfos[pid]['X']
        y = pinfos[pid]['Y']
        z = pinfos[pid]['Alt']
        m_x = minfo['X'] - x
        m_y = minfo['Y'] - y
        m_z = minfo['Alt'] - z
        m_speed = minfo['Speed']
        m_dist = np.linalg.norm(np.array([m_x, m_y, m_z]))
        gs += tid_oh + [m_x, m_y, m_z, m_dist, m_speed]
      return gs

    agent_id = [0] * self.n_units
    agent_id[uid] = 1
    gs = agent_id

    if self.red_alive_mask[uid]:
      ego_gs, x, y, z = get_ego_info(msg, uid)
      gs += ego_gs
      gs += get_info(msg, Agent.RED, x, y, z)
      gs += get_info(msg, Agent.BLUE, x, y, z)
      gs += get_miss_info(msg, Agent.RED)
      gs += get_miss_info(msg, Agent.BLUE)
    else:
      gs += [0] * (self.obs_shape[self.uid2aid[uid]]['global_state'][0] - self.n_units)

    return gs

  def get_avail_actions(self, msg, uid):
    """
      获得可执行动作列表
      :return available_actions size = (5, 2, 5)
    """
    action_mask = np.ones((self.action_dim[0][DEFAULT_ACTION],), bool)
    # 先判断这个Agent是否已经死亡
    if not self.red_alive_mask[uid]:
      action_mask[:-1] = 0
      return action_mask
    
    action_mask[-1] = 0
    pinfos = msg[Agent.RED]['platforminfos']
    # 如果没有死亡，那么移动就可以全部是1,需要判断能不能攻击具体到某个敌方，要进行弹药数量判断和距离判断
    pid = get_info_id(pinfos, RED_INFO[uid]['ID'])

    if pinfos[pid]['LeftWeapon'] == 0:
      action_mask[9:14] = 0
    else:
      a_pos = np.array([pinfos[pid]['X'], pinfos[pid]['Y'], pinfos[pid]['Alt']])
      # 进行地方距离判断
      # avai_mask[i][1] 每个index上对应着敌方固定的实体
      cur_ava = np.zeros(5)
      # 分别计算敌方实体的距离
      for i in range(5):
        # 查找蓝方实体的位置，要先判断这个蓝方是不是已经死掉了
        if self.blue_alive_mask[i]:
          # 这个蓝方实体还没有死掉
          e_id = BLUE_INFO[i]['ID']
          e_ids = [j for j, item in enumerate(msg[Agent.RED]['trackinfos']) if item['ID'] == e_id]
          if e_ids:
            assert len(e_ids) == 1, e_ids
            e_id = e_ids[0]
            enemy_info = msg[Agent.RED]['trackinfos'][e_id]
            e_pos = np.array([enemy_info['X'], enemy_info['Y'], enemy_info['Alt']])
            # 判断距离
            cur_ava[i] = np.linalg.norm(a_pos - e_pos) <= 80000

      # 加入射击间隔
      agent_shoot_interval = self.shoot_interval[uid]
      agent_shoot_legal = np.ones_like(agent_shoot_interval)
      agent_shoot_legal[agent_shoot_interval != 0] = 0
      # print(f'agent_shoot_legal', agent_shoot_legal)
      # print(f'original cur_ava: {cur_ava}')
      new_cur_ava = np.zeros_like(cur_ava)
      new_cur_ava[(cur_ava == 1) & (agent_shoot_legal == 1)] = 1
      # print(f'cur_ava', new_cur_ava)
      # cur_ava = np.append(cur_ava, 1)
      action_mask[9:14] = new_cur_ava

    return action_mask

  def get_reward(self, last_msg, msg):
    reward = np.zeros(self.n_units)           # 最终奖励
    rewards = {}

    """战场没有结束，只需要统计占损奖励"""
    last_red_msg = last_msg[Agent.RED]
    last_blue_msg = last_msg[Agent.BLUE]
    red_msg = msg[Agent.RED]
    blue_msg = msg[Agent.BLUE]
    last_red_pinfos = last_red_msg['platforminfos']
    last_blue_pinfos = last_blue_msg['platforminfos']
    red_pinfos = red_msg['platforminfos']
    blue_pinfos = blue_msg['platforminfos']
    last_red_weapon = [m for m in last_blue_msg["missileinfos"] if m["Identification"] == "红方"]
    last_blue_weapon = [m for m in last_red_msg["missileinfos"] if m["Identification"] == "蓝方"]
    red_weapon = [m for m in blue_msg["missileinfos"] if m["Identification"] == "红方"]
    blue_weapon = [m for m in red_msg["missileinfos"] if m["Identification"] == "蓝方"]
    # 1. 统计上一帧中，红方战机的数量 & 存在的导弹剩余数量
    n_last_red_planes = len(last_red_pinfos)
    n_last_red_weapon = len(last_red_weapon)
    # 2. 统计下一帧中，红方战机的数量 & 存在的导弹剩余数量
    n_red_planes = len(red_pinfos)
    n_red_weapon = len(red_weapon)
    # 3. 统计上一帧中，蓝方战机的数量 & 存在的导弹剩余数量
    n_last_blue_planes = len(last_blue_pinfos)
    n_last_blue_weapon = len(last_blue_weapon)
    # 4. 统计下一帧中，蓝方战机的数量 & 存在的导弹剩余数量
    n_blue_planes = len(blue_pinfos)
    n_blue_weapon = len(blue_weapon)

    # 计算锁定奖励
    red_locked = np.zeros(self.n_units)
    blue_locked = np.zeros(self.n_units)
    for p in red_pinfos:
      if not p['IsLocked'] and any([
          True for le in last_red_pinfos if le['ID'] == p['ID'] and le['IsLocked']]):
        if self.shared_reward:
          red_locked -= 1
        else:
          uid = get_uid(RED_INFO, p['ID'])
          red_locked[uid] -= 1
    # for p in blue_pinfos:
    #   if p['IsLocked']:
    #     blue_locked += 1
    rewards['lock'] = self.lock_reward_scale * (blue_locked - red_locked)
    reward += rewards['lock']

    # 计算伤害奖励
    rewards['damage'] = np.zeros(self.n_units)
    rewards['escape'] = np.zeros(self.n_units)
    if n_last_red_planes - n_red_planes > 0:
      # 说明上一轮红方飞机更多，红方飞机有伤亡
      if self.shared_reward:
        rewards['damage'] = -self.damage_reward_scale * (n_last_red_planes - n_red_planes)
      else:
        for p in last_red_pinfos:
          if any([True for pp in red_pinfos if p['ID'] == pp['ID']]):
            uid = get_uid(RED_INFO, p['ID'])
            rewards['damage'][uid] = -self.damage_reward_scale
    elif n_last_blue_weapon - n_blue_weapon > 0:
      # 说明红方飞机数量没有变化, 躲蛋成功获得奖励
      if self.shared_reward:
        rewards['escape'] = self.escape_reward_scale * (n_last_blue_weapon - n_blue_weapon)
      else:
        for w in last_blue_weapon:
          if any([True for ww in blue_weapon if w['ID'] == ww['ID']]):
            continue
          uid = get_uid(RED_INFO, w['EngageTargetID'])
          rewards['escape'][uid] = self.escape_reward_scale
    reward += rewards['damage']
    reward += rewards['escape']

    # 计算攻击奖励
    rewards['attack'] = np.zeros(self.n_units)
    rewards['miss'] = np.zeros(self.n_units)
    if n_last_blue_planes - n_blue_planes > 0:
      # 说明上一帧蓝方飞机要更多
      # 有意用共享reward, 没做ablation study
      rewards['attack'] = self.attack_reward_scale * (n_last_blue_planes - n_blue_planes)
    elif n_last_red_weapon - n_red_weapon > 0:
      # 说明蓝方飞机数量没有变化，被蓝方躲避，浪费弹药
      if self.shared_reward:
        rewards['miss'] = -self.miss_reward_scale * (n_last_red_weapon - n_red_weapon)
      else:
        for w in last_red_weapon:
          if any([True for ww in red_weapon if w['ID'] == ww['ID']]):
            continue
          uid = get_uid(RED_INFO, w['LauncherID'])
          rewards['miss'][uid] = -self.miss_reward_scale
    reward += rewards['attack']
    reward += rewards['miss']

    # 计算距离奖励
    distance_reward = np.zeros(self.n_units)
    for rid in range(self.n_units):
      if not self.red_alive_mask[rid]:
        continue
      id = RED_INFO[rid]['ID']
      pid = get_info_id(red_pinfos, id)
      if not self.red_alive_mask[rid] or red_pinfos[pid]['IsLocked']:
        continue
      # 当前的坐标
      cur_x = red_pinfos[pid]['X']
      cur_y = red_pinfos[pid]['Y']
      cur_distance = np.linalg.norm(np.array([cur_x/10000, cur_y/10000]))
      # 上一个坐标
      last_pids = [j for j, item in enumerate(last_red_pinfos) if item['ID'] == id]
      if len(last_pids) == 0:
        continue
      assert len(last_pids) == 1, last_pids
      last_pid = last_pids[0]
      last_x = last_red_pinfos[last_pid]['X']
      last_y = last_red_pinfos[last_pid]['Y']
      last_distance = np.linalg.norm(np.array([last_x/10000, last_y/10000]))
      # 这个值是很大的
      # distance_reward += 1 if (last_distance - cur_distance) > 0 else -1
      if self.shared_reward:
        distance_reward += last_distance - cur_distance
      else:
        distance_reward[rid] = last_distance - cur_distance
    rewards['distance'] = self.distance_reward_scale * distance_reward
    reward += rewards['distance']

    # 计算边缘惩罚
    border_reward = np.zeros(self.n_units)
    for rid in range(self.n_units):
      if not self.red_alive_mask[rid]:
        continue
      # 当前的坐标
      cur_pid = get_info_id(red_pinfos, RED_INFO[rid]['ID'])
      cur_x = red_pinfos[cur_pid]['X']
      cur_y = red_pinfos[cur_pid]['Y']
      r = 0
      if abs(cur_x) > 140000:
        if abs(cur_x) >= 150000:
          print(f"红方飞机{rid}自杀了！")
        # print(f"红方飞机{rid}距离x轴飞出边界还剩{150000 - abs(cur_x)}！")
        r -= 1
      if abs(cur_y) > 140000:
        if abs(cur_y) >= 150000:
          print(f"红方飞机{rid}自杀了！")
        # print(f"红方飞机{agent_order}距离y轴飞出边界还剩{150000 - abs(cur_y)}！")
        r -= 1
      if self.shared_reward:
        border_reward += r
      else:
        border_reward[rid] += r
    rewards['border'] = self.border_reward_scale * border_reward
    reward += rewards['border']

    self._rewards = rewards.copy()

    return reward

  def get_done(self, msg):
    """
    推演是否结束
    @param obs: 环境状态信息
    @return: done列表信息
    """
    done = False
    reward = 0
    reason = None

    cur_time = msg["sim_time"]
    red_pinfos = msg[Agent.RED]["platforminfos"]
    blue_pinfos = msg[Agent.BLUE]["platforminfos"]
    # print("get_done cur_time:", cur_time)
    if not self.red_alive_mask[0]:
      # print("红方有人机阵亡")
      done = True
      reward = - self.win_reward
      reason = Reason.LOSE
      self._rewards['win'] = reward
    elif not self.blue_alive_mask[0]:
      # print("蓝方有人机阵亡")
      done = True
      reward = self.win_reward
      reason = Reason.WIN
      self._rewards['win'] = reward
    elif cur_time >= 20 * 60 - 1:
      done = True
      reward = len(red_pinfos) - len(blue_pinfos)
      reason = Reason.TIMEOUT
      self._rewards['timeout'] = reward
    else:
      red_ammo_free = all([pinfo["LeftWeapon"] == 0 for pinfo in red_pinfos])
      blue_ammo_free = all([pinfo["LeftWeapon"] == 0 for pinfo in blue_pinfos])
      red_fly_missile = len([m for m in msg[Agent.BLUE]["missileinfos"] if m["Identification"] == "红方"]) > 0
      blue_fly_missile = len([m for m in msg[Agent.RED]["missileinfos"] if m["Identification"] == "蓝方"]) > 0
      if red_ammo_free and blue_ammo_free and not red_fly_missile and not blue_fly_missile:
        done = True
        reward = -self.win_reward if red_ammo_free else self.win_reward
        reason = Reason.LOSE if red_ammo_free else Reason.WIN
        self._rewards['win'] = reward
      else:
        if self.red_agent_loc[0]['X'] >= 150000 or self.red_agent_loc[0]['Y'] >= 150000:
          done = True
          reward = -self.win_reward
          reason = Reason.OOR
        else:
          done = False
          reward = 0
          reason = Reason.IN_PROGRESS
        self._rewards['win'] = reward

    return done, reward, reason

  def close(self):
    """关闭仿真引擎"""
    print(self.address, 'simulation is closed')
    super().close()


if __name__ == '__main__':
  """测试环境类"""
  print('testing environment')
  ADDRESS['port'] = 1234
  address = ADDRESS['ip'] + ":" + str(ADDRESS['port'])
  print('address:', address)
  env = HuaRu5v5(
    env_name='5v5', 
    eid=ADDRESS['port'], 
    ip=ADDRESS['ip'], 
    port=ADDRESS['port'])
  obs_list = []
  obs = env.reset()
  np.random.seed(0)
  obs_list += obs
  i = 0
  dones = []
  actions = []
  epslens = []
  while True:
    i += 1
    action = env.random_action()
    actions.append(action)
    obs, rew, done, info = env.step(action)
    print_dict_info(obs)
    dones.append(done)
    if np.any(done):
      epslens.append(env._epslen)
      env.reset()
    obs_list += obs
  env.close()
  obs = batch_dicts(obs_list)
  from tools.display import print_dict_info
  print_dict_info(obs)
