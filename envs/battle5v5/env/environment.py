"""
!/usr/bin/python3
-*- coding: utf-8 -*-
@FileName: environment.py
@Time: 2024/4/15 下午3:54
@Author: ZhengtaoCao
@Description: None
"""

from .env_runner import EnvRunner
import numpy as np
import gym
import torch as th
from time import sleep
from envs.battle5v5.config import config, Agent, BLUE_INFO, RED_INFO, BLUE_FIRE_INFO, RED_FIRE_INFO
# from .multiagentenv import MultiAgentEnv
import math
import logging

# logger = logging.getLogger('Battle_5v5')
# logger.setLevel(logging.DEBUG)


def get_info_id(pinfos, id, check_existence=False):
  pids = [j for j, item in enumerate(pinfos) if item['ID'] == id]
  if check_existence and len(pids) == 0:
    return None
  assert len(pids) == 1, pids
  pid = pids[0]
  return pid


def get_uid(infos, id):
  uids = [eval(k) for k, item in infos.items() if item['ID'] == id]
  assert len(uids) == 1, uids
  uid = uids[0]
  return uid


class Agent:
  BLUE = 'blue'
  RED = 'red'

class HuaRuBattleEnvWrapper(EnvRunner):
  def __init__(self,
         map_name='battle5v5',
         num_agents=5,
         time_step=0,
         seed=123,
         episode_limit=80,
         agents=config['agents'],
         address=None,  # ADDRESS['ip'] + ":" + str(ADDRESS['port']),
         mode='host',
         logger=None
    ):
    EnvRunner.__init__(self, agents, address, mode)
    # MultiAgentEnv.__init__(self)
    self.map_name = map_name
    self.address = address
    self.n_agents = num_agents

    # 动作空间
    # self.action_space_1 = 10 + 10   # 9个方向 + 每个智能体可以选择跟随一个实体(9)和跟随自己(区域巡航)
    # self.action_space_2 = 6   # 5个攻击动作加不攻击
    # self.action_space_3 = 3  # 加速、减速、速度保持不变 

    # single head
    self.action_space = 9 + 5 + 2 + 1  # 9个方向 + 5攻击动作 + 加速减速 + 不采取任何动作
    # 状态空间
    self.observation_space = 105  # [spaces.Box(low=-1, high=1, shape=(40,))]  # shapes1.box  shared_obs_size: 105 * 5 for Qmix, 359 for MAPPO(把所有实体的roll都去掉了，没有用到) 

    self.ori_message = None
    self.episode_limit = episode_limit
    self.red_death = None  # 存入红方死亡的实体ID
    self.blue_death = None  # 存入蓝方死亡实体ID

    # 记录每一帧每个飞机实体的位置
    self.red_agent_loc = None
    self.agents_speed = None

    # 存储上一帧的原始战场态势数据
    self.last_ori_message = None
    # 存取本episode训练的其他信息
    self.info = None
    self.forward_step = None
    self.logger = logger
    self.done = None
    self.already_done = False

    self.reward = None
    self.shoot_interval = None  # 每个Agent的设计间隔
    self.shoot_interval_step = 0  # 50000米发射，导弹1000m/s

    # 记录胜率
    self.battles_won = 0.
    self.battles_game = 0.

    # 记录各个部分的奖励
    self.reward_attack = None
    self.reward_damage = None
    self.reward_distance = None
    self.reward_final = None

    # 模拟修改想定步长
    self.change_time_step = 15  # 每次做出动作之后，要空推15帧
    
    self.abundant_info = True

  def reset(self, if_test=False, args=None, cur_time=None):
    """重置仿真环境, 返回初始帧obs"""
    for side, agent in self.agents.items():
      agent.reset()
    super().reset()
    self.ori_message = super().step([])
    self.update_is_locked(self.ori_message)
    self.update_alive_mask(self.ori_message)
    # self.logger.console_logger.info('Reset, Restart Engine!')
    self.forward_step = 0
    self.red_death = []
    self.blue_death = []

    self.red_alive_mask = []
    self.blue_alive_mask = []

    self.red_agent_loc = {'0': None, '1': None, '2': None, '3': None, '4': None}
    self.agents_speed = {'0': None, '1': None, '2': None, '3': None, '4': None}
    self.info = {'Battle Won': 2}  # 2代表是平局, 0红方胜利，1蓝方胜利
    self.obs = None
    self.shoot_interval = np.zeros([self.n_agents, 5])
    self.reward_attack = 0.
    self.reward_damage = 0.
    self.reward_distance = 0.
    self.reward_final = 0.

    # 重置奖励函数
    self.reward = 0.
    self.already_done = False
    # 重置训练环境

    if self.ori_message["sim_time"] == 0.0:
      self.ori_message = super().step([])  # 推动拿到第一帧的obs信息
      self.update_is_locked(self.ori_message)
      self.update_alive_mask(self.ori_message)
    self.obs = self.get_obs()
    self.state = self.get_state()
    # 准备将自己已经知道的所有信息一起发送给Agent
    parse_msg_ = {'agent_pre_loc': self.red_agent_loc,
           'blue_info': BLUE_INFO,
           'red_info': RED_INFO,
           'sim_time': self.ori_message['sim_time'],
           'agent_speed': self.agents_speed # 记录当前帧每个智能体的速度，如果已经死亡，则速度为-1
    }

    cmd_list = []
    cmd_list.extend(self.agents["red"].make_init_cmd())
    cmd_list.extend(self.get_blue_cmd(self.agents["blue"]))

    self.last_ori_message = self.ori_message
    self.ori_message = super().step(cmd_list)
    self.update_is_locked(self.ori_message)
    self.update_alive_mask(self.ori_message)
    

    ## 将前150步的规则操作都移到reset()中，不输入到算法中，减小探索空间
    # 前150将动作为0传给alo_agent
    rule_actions = [[0], [0], [0], [0], [0]]  # 加一个强规则，前150帧像前飞，减小探索空间
    while self.ori_message["sim_time"] < 200:
      cmd_list = []
      if self.ori_message["sim_time"] % 20 == 0:
        cmd_list.extend(self.agents["red"].make_actions(rule_actions, parse_msg_))  # parse_msg里的sim_time应该没更新
        self.last_ori_message = self.ori_message  # 2024/06/24 修改，15帧决策时，上一帧的数据应该是20帧之前
      else:
        pass
      cmd_list.extend(self.get_blue_cmd(self.agents["blue"]))
      # self.last_ori_message = self.ori_message
      self.ori_message = super().step(cmd_list)
      self.update_is_locked(self.ori_message)
      self.update_alive_mask(self.ori_message)

      # 获取RL Obs & State
      self.obs = self.get_obs()
      self.state = self.get_state()

    return self.obs, self.state   # list, np

  def step(self, actions=None, if_test=False):
    """
      逐帧推进仿真引擎, 发送actions, 返回obs
      :param actions: [[0, 2],
              [1, 3],
              ...
              ] 是一个二维数组，规模是(5, 2) 第1列是移动动作，动作空间是9；第2列是攻击动作，动作空间是6
    """
    # print(f'step: {self.forward_step}, actions: {actions}')
    # single head 不修改
    actions = np.array([[element] for element in actions]) # for parallel
    for index in range(len(actions)):
      # 拿取每个agent的攻击动作
      cur_shoot_action = int(actions[index][0])
      if cur_shoot_action >= 9 and cur_shoot_action < 14:  # 射击间隔
        # self.shoot_interval[index][cur_shoot_action] = self.shoot_interval_step
        # 修改为，朝着所有敌方发射都存在间隔
        for shoot_target in range(5):
          self.shoot_interval[index][shoot_target] = self.shoot_interval_step
    non_zero_indices = self.shoot_interval != 0
    self.shoot_interval[non_zero_indices] -= 1
    self.forward_step += 1

    if_done = False
    # # 准备将自己已经知道的所有信息一起发送给Agent
    parse_msg_ = {'agent_pre_loc': self.red_agent_loc,
           'blue_info': BLUE_INFO,
           'red_info': RED_INFO,
           'sim_time': self.ori_message['sim_time'],
           'agent_speed': self.agents_speed # 记录当前帧每个智能体的速度，如果已经死亡，则速度为-1
    }
    
    # 生成蓝方Agents的cmd list
    blue_cmd_list = self.get_blue_cmd(self.agents["blue"])
    # 将网络输出动作转换为仿真执行指令,给self.red_agents
    cmd_list = []
    cmd_list.extend(self.agents["red"].make_actions(actions, parse_msg_)) # Agents的仿真指令
    # print(f'cmd list: {cmd_list}')
    cmd_list.extend(blue_cmd_list)  ## 注释掉蓝方
    # print(f'当前生成的指令包括：{cmd_list}')
    self.last_ori_message = self.ori_message
    # 将仿真指令，发送回仿真，并且拿到下一帧的状态
    self.ori_message = super().step(cmd_list)
    self.update_is_locked(self.ori_message)
    self.update_alive_mask(self.ori_message)
    # print(super().get_done(self.ori_message))
    self.done, flag_no = super().get_done(self.ori_message)
    
    if not self.done[0]:
      ## 如果修改想定步长，则是15帧之内空推，不产生任务指令
      forward_step = 1
      while forward_step % self.change_time_step != 0:
        blue_cmd_list = self.get_blue_cmd(self.agents["blue"])
        cmd_list = []
        cmd_list.extend(blue_cmd_list)
        # self.last_ori_message = self.ori_message   # 2024/06/24 修改，15帧决策时，上一帧的数据应该是15帧之前
        self.ori_message = super().step(cmd_list)
        self.update_is_locked(self.ori_message)
        self.update_alive_mask(self.ori_message)
        self.done, flag_no = super().get_done(self.ori_message)
        if self.done[0]:
          break
        forward_step+=1
    
    # 解析得到下一帧的Agents Obs
    self.obs = self.get_obs()
    self.state = self.get_state()  # 更新全局state

    information = {"battle_win": False}

    if self.done[0] or self.forward_step >= self.episode_limit:  # 或者步数超过episode_limit
      if not self.done[0]:
        # 超过了步数
        # 说明当前episode的训练已经结束
        self.logger.console_logger.info('当前Episode训练已经结束(超过指定训练步数)！')
        self.info['Battle Won'] = 2
        self.logger.console_logger.info("平  局")
        # 结束的时候不需要kill env，只是下个episode开始的时候，reset()一下Agents就可以了
        information["episode_limit"] = True
        if_done = True
        # self.close_environ()
        # super().reset()
      else:
        # 说明当前episode的训练已经结束
        self.logger.console_logger.info('当前Episode训练已经结束！')
        if (self.done[1] == 1 and self.done[2] == 0):
          self.logger.console_logger.info("红 方 胜!")
          self.info['Battle Won'] = 0
          information['battle_win'] = True  # self.get_stats()['win_rate']
          self.battles_won += 1
        elif (self.done[1] == 0 and self.done[2] == 1):
          self.logger.console_logger.info("蓝 方 胜!")
          self.info['Battle Won'] = 1
        else:
          self.logger.console_logger.info("平  局")
          self.info['Battle Won'] = 2
        # 结束的时候不需要kill env，只是下个episode开始的时候，reset()一下Agents就可以了
        if_done = True
        # self.close_environ()
        # super().reset()

      self.battles_game += 1
      # self.logger.console_logger.info(f'取得的胜率：{self.get_stats()["win_rate"]}')
      # 计算红方有人，无人
      red_total_num = len(self.ori_message['red']['platforminfos'])
      if 1 in self.red_death:
        red_av_num = 0
        red_uav_num = red_total_num
      else:
        red_av_num = 1
        red_uav_num = red_total_num - 1

      blue_total_num = len(self.ori_message['blue']['platforminfos'])
      if 6 in self.blue_death:
        blue_av_num = 0
        blue_uav_num = blue_total_num
      else:
        blue_av_num = 1
        blue_uav_num = blue_total_num - 1

      self.logger.console_logger.info(
        f"红方剩余兵力：有人机剩余 {red_av_num} 架，无人机剩余: {red_uav_num} 架; 红方剩余弹药情况：{sum([entity['LeftWeapon'] for entity in self.ori_message['red']['platforminfos']])}")
      self.logger.console_logger.info(
        f"蓝方剩余兵力：有人机剩余 {blue_av_num} 架，无人机剩余: {blue_uav_num} 架; 蓝方剩余弹药情况：{sum([entity['LeftWeapon'] for entity in self.ori_message['blue']['platforminfos']])}")

    # cur_reward = self.get_reward_adjust(self.last_ori_message, self.ori_message, if_done, flag_no)
    cur_reward = self.get_reward_new(self.last_ori_message, self.ori_message, if_done, flag_no)
    # cur_reward = self.get_reward_test(self.last_ori_message, self.ori_message, if_done, flag_no)
    # self.reward计算的是累积奖励，没有什么用, cur_reward才是当前帧的奖励
    self.reward += cur_reward

    if self.done[0]:
      self.logger.console_logger.info(f'本Episode获得的最后一帧奖励是: {cur_reward}')
      self.logger.console_logger.info(f'本Episode获得的累积奖励: {self.reward}')
      self.logger.console_logger.info(f'本Episode获得的<距离>累积奖励: {self.reward_distance}')
      self.logger.console_logger.info(f'本Episode获得的<伤害>累积惩罚: {self.reward_damage}')
      self.logger.console_logger.info(f'本Episode获得的<攻击>累积奖励: {self.reward_attack}')
      self.logger.console_logger.info(f'本Episode获得的<结局>累积奖励: {self.reward_final}')

    if self.forward_step % 1 == 0:
      self.logger.console_logger.info('Forward step: %d' % self.forward_step)
      self.logger.console_logger.info(f'Cur frame reward: {cur_reward}')
      self.logger.console_logger.info(f'Total reward: {self.reward}')

    # 结束环境
    return cur_reward, if_done, information  # self.info

  def get_blue_cmd(self, blue_agents):
    """获取蓝方当前的cmd_list"""
    cur_time = self.ori_message["sim_time"]
    cmd_list = super()._agent_step(blue_agents, cur_time, self.ori_message["blue"])

    return cmd_list

  def get_reward(self, last_obs=None, next_obs=None, if_done=False):
    """
      每帧计算reward值，使用团队奖励，当敌方的某个战机被击落后，产生正奖励，当己方的一个飞机被击落，产生负奖励；
      到最后，即if_done==True，说明episode结束，开始进行结局奖励计算。
      :param last_obs: 上一帧的战场局势(原始数据形式)
      :param next_obs: 下一帧的战场局势(原始数据形式)
      :param if_done: 本episode是否已经结束？如果结束的话，要进行战场判定
    """
    reward = 0.
    """战场没有结束，只需要统计占损奖励"""
    # 1. 统计上一帧中，红方战机的数量 & 存在的导弹剩余数量
    last_red_agent_num = len(last_obs['red']['platforminfos'])
    last_red_weapon_num = 0.
    for entity in last_obs['red']['platforminfos']:
      last_red_weapon_num += entity['LeftWeapon']
    # 2. 统计下一帧中，红方战机的数量 & 存在的导弹剩余数量
    next_red_agent_num = len(next_obs['red']['platforminfos'])
    next_red_weapon_num = 0.
    for entity in next_obs['red']['platforminfos']:
      next_red_weapon_num += entity['LeftWeapon']
    # 3. 统计上一帧中，蓝方战机的数量 & 存在的导弹剩余数量
    last_blue_agent_num = len(last_obs['blue']['platforminfos'])
    last_blue_weapon_num = 0.
    for entity in next_obs['blue']['platforminfos']:
      last_blue_weapon_num += entity['LeftWeapon']
    # 4. 统计下一帧中，蓝方战机的数量 & 存在的导弹剩余数量
    next_blue_agent_num = len(next_obs['blue']['platforminfos'])
    next_blue_weapon_num = 0.
    for entity in next_obs['blue']['platforminfos']:
      next_blue_weapon_num += entity['LeftWeapon']

    # 计算占损战耗奖励
    reward = (-1.2) * (10 * (last_red_agent_num - next_red_agent_num)) - 0.5 * (last_red_weapon_num - next_red_weapon_num) \
         + 1.2 * (10 * (last_blue_agent_num - next_blue_agent_num) + 0.5 * (last_blue_weapon_num - next_blue_weapon_num))
    # reward = (-1.5) * (10 * (last_red_agent_num - next_red_agent_num)) \
    #      + 2.0 * (50 * (last_blue_agent_num - next_blue_agent_num) + 0.5 * (last_blue_weapon_num - next_blue_weapon_num))

    distance_reward = 0.
    if not if_done:
      # 设置一个奖励：计算每一帧红方每个飞机离中心点的距离，距离中心点越近得分越高
      for agent_order in range(len(self.ori_message['red']['platforminfos'])):
        cur_id = self.ori_message['red']['platforminfos'][agent_order]['ID']
        if cur_id in self.red_death:
          distance_reward += 0.
        else:
          # 当前的坐标
          cur_x = self.ori_message['red']['platforminfos'][agent_order]['X']
          cur_y = self.ori_message['red']['platforminfos'][agent_order]['Y']
          cur_z = self.ori_message['red']['platforminfos'][agent_order]['Alt']
          cur_distance = math.sqrt(cur_x ** 2 + cur_y ** 2 + cur_z ** 2)
          # 上一个坐标
          last_x = self.last_ori_message['red']['platforminfos'][agent_order]['X']
          last_y = self.last_ori_message['red']['platforminfos'][agent_order]['Y']
          last_z = self.last_ori_message['red']['platforminfos'][agent_order]['Alt']
          last_distance = math.sqrt(last_x ** 2 + last_y ** 2 + last_z ** 2)

        distance_reward += 0.002 if (last_distance - cur_distance) >= 0 else -0.002

    else:
      """战场已经结束, 需要额外统计结局奖励"""
      if self.info['Battle Won'] == 0: # 红方胜利
        reward += 600
      elif self.info['Battle Won'] == 1:
        reward -= 600
      else:
        reward += 0

    reward += distance_reward

    return reward

  def get_reward_new(self, last_obs=None, next_obs=None, if_done=False, flag_no=False):

    reward = 0.  # 最终奖励
    distance_reward = 0.  # 距离奖励, 距离中心点越近越胜利，越来越低
    attack_reward = 0.  # 攻击奖励，攻击敌方越多伤害越高，毁伤奖励应该是越来越高的
    damaged_reward = 0.  # 毁伤奖励， 自己造成的毁伤越多得到的负奖励越高
    locked_reward = 0.   # 锁定奖励，锁定敌方飞机，获取奖励
    protect_reward = 0.  # 保护红方有人机

    """战场没有结束，只需要统计占损奖励"""
    # 1. 统计上一帧中，红方战机的数量 & 存在的导弹剩余数量
    last_red_agent_num = len(last_obs['red']['platforminfos'])
    last_red_weapon_num = 0.
    for entity in last_obs['red']['platforminfos']:
      last_red_weapon_num += entity['LeftWeapon']
    # 2. 统计下一帧中，红方战机的数量 & 存在的导弹剩余数量
    next_red_agent_num = len(next_obs['red']['platforminfos'])
    next_red_weapon_num = 0.
    for entity in next_obs['red']['platforminfos']:
      next_red_weapon_num += entity['LeftWeapon']
    # 3. 统计上一帧中，蓝方战机的数量 & 存在的导弹剩余数量
    last_blue_agent_num = len(last_obs['blue']['platforminfos'])
    last_blue_weapon_num = 0.
    for entity in last_obs['blue']['platforminfos']:
      last_blue_weapon_num += entity['LeftWeapon']
    # 4. 统计下一帧中，蓝方战机的数量 & 存在的导弹剩余数量
    next_blue_agent_num = len(next_obs['blue']['platforminfos'])
    next_blue_weapon_num = 0.
    for entity in next_obs['blue']['platforminfos']:
      next_blue_weapon_num += entity['LeftWeapon']

    # 计算锁定奖励
    last_red_locked_num = 0
    for entity in last_obs['red']['platforminfos']:
      last_red_locked_num += 1 if entity['IsLocked'] else 0
    next_red_locked_num = 0
    for entity in next_obs['red']['platforminfos']:
      next_red_locked_num += 1 if entity['IsLocked'] else 0

    last_blue_locked_num = 0
    for entity in last_obs['blue']['platforminfos']:
      last_blue_locked_num +=1 if entity['IsLocked'] else 0

    next_blue_locked_num = 0
    for entity in next_obs['blue']['platforminfos']:
      next_blue_locked_num += 1 if entity['IsLocked'] else 0

    if last_red_locked_num > 0:
      if next_red_locked_num > 0:
        locked_reward -= 1
        self.logger.console_logger.info(f'红方被持续锁定! -1')
      else:
        locked_reward += 10
        self.logger.console_logger.info(f'红方脱离锁定! +10')

    if last_blue_locked_num > 0:
      if next_blue_locked_num > 0:
        locked_reward += 1
        self.logger.console_logger.info(f'蓝方被持续锁定! +1')
      else:
        locked_reward -= 10
        self.logger.console_logger.info(f'蓝方脱离锁定! -10')

    # # new
    # if last_blue_locked_num - next_blue_locked_num > 0:
    #   locked_reward -= 2
    #   self.logger.console_logger.info(f'蓝方脱离锁定! -2')
    # else:
    #   locked_reward += 1
    #   self.logger.console_logger.info(f'蓝方被持续锁定! +1')

    reward += locked_reward
    # # 计算保护奖励
    # last_red_x_positions = []
    # last_red_y_positions = []
    # last_red_av_x = None
    # last_red_av_y = None
    # last_red_av_x_protected = False
    # last_red_av_y_protected = False
    # last_red_av_protected = False

    # for entity in last_obs['red']['platforminfos']:
    #   if entity['ID'] == 1:
    #     last_red_av_x = entity['X']
    #     last_red_av_y = entity['Y']
    #   else:
    #     last_red_x_positions.append(entity['X'])
    #     last_red_y_positions.append(entity['Y'])

    # if last_red_av_x and last_red_av_y and len(last_red_x_positions) > 0 and len(last_red_y_positions):
    #   if last_red_av_x < max(last_red_x_positions) and last_red_av_x > min(last_red_x_positions):
    #     last_red_av_x_protected = True
    #   if last_red_av_y < max(last_red_y_positions) and last_red_av_y > min(last_red_y_positions):
    #     last_red_av_y_protected = True

    #   if last_red_av_x_protected and last_red_av_y_protected:
    #     last_red_av_protected = True

    # next_red_x_positions = []
    # next_red_y_positions = []
    # next_red_av_x = None
    # next_red_av_y = None
    # next_red_av_x_protected = False
    # next_red_av_y_protected = False
    # next_red_av_protected = False

    # for entity in next_obs['red']['platforminfos']:
    #   if entity['ID'] == 1:
    #     next_red_av_x = entity['X']
    #     next_red_av_y = entity['Y']
    #   else:
    #     next_red_x_positions.append(entity['X'])
    #     next_red_y_positions.append(entity['Y'])

    # if next_red_av_x and next_red_av_y and len(next_red_x_positions) > 0 and len(next_red_y_positions) > 0:
    #   if next_red_av_x < max(next_red_x_positions) and next_red_av_x > min(next_red_x_positions):
    #     next_red_av_x_protected = True
    #   if next_red_av_y < max(next_red_y_positions) and next_red_av_y > min(next_red_y_positions):
    #     next_red_av_y_protected = True

    #   if next_red_av_x_protected and next_red_av_y_protected:
    #     next_red_av_protected = True

    # if not last_red_av_protected and next_red_av_protected:
    #   protect_reward += 20
    #   self.logger.console_logger.info(f'保护红方有人机! +20')
    # if last_red_av_protected and not next_red_av_protected:
    #   protect_reward -= 50
    #   self.logger.console_logger.info(f'红方有人机脱离保护! -50')
    # if last_red_av_protected and next_red_av_protected:
    #   protect_reward += 0.1
    #   # self.logger.console_logger.info(f'持续保护红方有人机! +0.1')
    # if not last_red_av_protected and not next_red_av_protected:
    #   protect_reward -= 0.1
    #   # self.logger.console_logger.info(f'没有保护红方有人机! -0.1')

    # next_x_y_too_far = 0
    # if next_red_av_x:
    #   next_x_y_too_far += (np.abs(np.array(next_red_x_positions) - next_red_av_x) > 75000).sum()
    # if next_red_av_y:
    #   next_x_y_too_far += (np.abs(np.array(next_red_y_positions) - next_red_av_y) > 75000).sum()

    # if next_x_y_too_far > 0:
    #   protect_reward -= 0.1 * next_x_y_too_far
    #   self.logger.console_logger.info(f'无人机离有人机过远！{-0.5 * next_x_y_too_far}')
    # else:
    #   self.logger.console_logger.info(f'所有无人机不离有人机过远！')

    # reward += protect_reward

    # 计算伤害奖励
    if last_red_agent_num - next_red_agent_num > 0:
      # 说明上一轮红方飞机更多，红方飞机有伤亡
      if RED_INFO['0']['ID'] in self.red_death:
        # 红方有人机被打掉，红方输掉了
        # damaged_reward -= 600
        pass # 算在最终输赢奖励里
      else:
        # 红方无人机被打掉，减小分
        damaged_reward -= 200 * (last_red_agent_num - next_red_agent_num)
        # damaged_reward -= 200 * (last_red_agent_num - next_red_agent_num)  # 地方是不是不对，如果同一帧，死掉两个实体呢？这里应该是 *
    if last_blue_weapon_num - next_blue_weapon_num > 0:
      # 说明红方飞机数量没有变化, 躲蛋成功获得，奖励
      damaged_reward += 100 * (last_blue_weapon_num - next_blue_weapon_num)
      # self.logger.console_logger.info(f'红方躲蛋成功! +{damaged_reward}')

    reward += damaged_reward
    self.reward_damage += damaged_reward
    # 计算攻击奖励

    if last_blue_agent_num - next_blue_agent_num > 0:
      # 说明上一帧蓝方飞机要更多
      if BLUE_INFO['0']['ID'] in self.blue_death:
        # 蓝方有人机被打掉，红方胜利
        # attack_reward += 600
        pass # 算在最终输赢奖励里
      else:
        attack_reward += 200 * (last_blue_agent_num - next_blue_agent_num)
    if last_red_weapon_num - next_red_weapon_num > 0:
      # 说明蓝方飞机数量没有变化， 被蓝方躲避，浪费弹药
      attack_reward -= 100 * (last_red_weapon_num - next_red_weapon_num)
      # self.logger.console_logger.info(f'蓝方躲蛋成功! 红方浪费弹药 {attack_reward}')

    reward += attack_reward
    self.reward_attack += attack_reward
    # 计算距离奖励
    for agent_order in range(len(self.ori_message['red']['platforminfos'])):
      cur_id = self.ori_message['red']['platforminfos'][agent_order]['ID']
      if cur_id in self.red_death:
        distance_reward += 0.
      else:
        # 当前的坐标
        cur_x = next_obs['red']['platforminfos'][agent_order]['X']
        cur_y = next_obs['red']['platforminfos'][agent_order]['Y']
        cur_distance = math.sqrt((cur_x / 10000) ** 2 + (cur_y / 10000) ** 2)
        # 上一个坐标
        last_x = last_obs['red']['platforminfos'][agent_order]['X']
        last_y = last_obs['red']['platforminfos'][agent_order]['Y']
        last_distance = math.sqrt((last_x / 10000) ** 2 + (last_y / 10000) ** 2)
        # 这个值是很大的
        # distance_reward += 1 if (last_distance - cur_distance) > 0 else -1
        distance_reward += 0.5 if (last_distance - cur_distance) > 0 else -0.5

    reward += (distance_reward * self.decay_coefficient(step=self.forward_step))  # 1.0是衰减系数
    self.reward_distance += distance_reward
    # # 计算边缘惩罚
    # border_reward = 0
    # if next_obs['sim_time'] > 180:
    #   for agent_order in range(len(self.ori_message['red']['platforminfos'])):
    #     cur_id = self.ori_message['red']['platforminfos'][agent_order]['ID']
    #     if cur_id in self.red_death:
    #       border_reward += 0.
    #     else:
    #       # 当前的坐标
    #       cur_x = next_obs['red']['platforminfos'][agent_order]['X']
    #       cur_y = next_obs['red']['platforminfos'][agent_order]['Y']
    #       if abs(cur_x) > 120000.0:
    #         if 150000 - abs(cur_x) <= 0:
    #           print(f"红方飞机{agent_order}自杀了！")
    #         # print(f"红方飞机{agent_order}距离x轴飞出边界还剩{150000 - abs(cur_x)}！")
    #         border_reward += 3 * np.exp(1 - (abs(cur_x) - 120000) / 30000)
    #       if abs(cur_y) > 120000.0:
    #         if 150000 - abs(cur_y) <= 0:
    #           print(f"红方飞机{agent_order}自杀了！")
    #         # print(f"红方飞机{agent_order}距离y轴飞出边界还剩{150000 - abs(cur_y)}！")
    #         border_reward += 3 * np.exp(1 - (abs(cur_y) - 120000) / 30000)

    # reward -= border_reward

    # 胜负奖励
    winner = 0
    if if_done:
      if self.info['Battle Won'] == 0:
        winner = 1
      elif self.info['Battle Won'] == 1:
        winner = 2
    elif next_red_weapon_num == 0: # 无弹惩罚 （无弹时已经输了）
      missile_locking_human = None
      for missile in next_obs['red']['missileinfos']:
        if missile['Identification'] == '红方' and missile['EngageTargetID'] == BLUE_INFO['0']['ID']: # 场上没有正在追击敌方有人机的导弹 已经输了
          missile_locking_human = missile
          break
      if missile_locking_human is None:
        winner = 2
    elif next_blue_weapon_num == 0: # 无弹提前胜利
      missile_locking_human = None
      for missile in next_obs['blue']['missileinfos']:
        if missile['Identification'] == '蓝方' and missile['EngageTargetID'] == RED_INFO['0']['ID']:
          missile_locking_human = missile
          break
      if missile_locking_human is None:
        winner = 1

    if not self.already_done:
      if winner == 1: # 红方胜
        self.already_done = True
        reward += 2400
        self.reward_final += 2400
      elif winner == 2: # 蓝方胜 (极端情况： 红射4发导弹，击落蓝方全部无人机，蓝射光全部导弹，击落红方有人机，此时红方差值200*4+100*8=1600)
        self.already_done = True
        reward -= 2400
        self.reward_final += (-2400)
      else: # 尚未结束，或平局
        reward += 0
        self.reward_final += 0.

    return reward

  def get_reward_adjust(self, last_obs=None, next_obs=None, if_done=False, flag_no=False):
    """
      每帧计算reward值，使用团队奖励，当敌方的某个战机被击落后，产生正奖励，当己方的一个飞机被击落，产生负奖励；
      到最后，即if_done==True，说明episode结束，开始进行结局奖励计算。
      :param last_obs: 上一帧的战场局势(原始数据形式)
      :param next_obs: 下一帧的战场局势(原始数据形式)
      :param if_done: 本episode是否已经结束？如果结束的话，要进行战场判定
    """
    reward = 0.  # 最终奖励
    distance_reward = 0.  # 距离奖励, 距离中心点越近越胜利，越来越低
    attack_reward = 0.  # 攻击奖励，攻击敌方越多伤害越高，毁伤奖励应该是越来越高的
    damaged_reward = 0.  # 毁伤奖励， 自己造成的毁伤越多得到的负奖励越高

    """战场没有结束，只需要统计占损奖励"""
    # 1. 统计上一帧中，红方战机的数量 & 存在的导弹剩余数量
    last_red_agent_num = len(last_obs['red']['platforminfos'])
    last_red_weapon_num = 0.
    for entity in last_obs['red']['platforminfos']:
      last_red_weapon_num += entity['LeftWeapon']
    # 2. 统计下一帧中，红方战机的数量 & 存在的导弹剩余数量
    next_red_agent_num = len(next_obs['red']['platforminfos'])
    next_red_weapon_num = 0.
    for entity in next_obs['red']['platforminfos']:
      next_red_weapon_num += entity['LeftWeapon']
    # 3. 统计上一帧中，蓝方战机的数量 & 存在的导弹剩余数量
    last_blue_agent_num = len(last_obs['blue']['platforminfos'])
    last_blue_weapon_num = 0.
    for entity in next_obs['blue']['platforminfos']:
      last_blue_weapon_num += entity['LeftWeapon']
    # 4. 统计下一帧中，蓝方战机的数量 & 存在的导弹剩余数量
    next_blue_agent_num = len(next_obs['blue']['platforminfos'])
    next_blue_weapon_num = 0.
    for entity in next_obs['blue']['platforminfos']:
      next_blue_weapon_num += entity['LeftWeapon']

    # 计算伤害奖励
    if last_red_agent_num - next_red_agent_num > 0:
      # 说明上一轮红方飞机更多，红方飞机有伤亡
      if RED_INFO['0']['ID'] in self.red_death:
        # 红方有人机被打掉，红方输掉了
        damaged_reward -= 12
      else:
        # 红方无人机被打掉，减小分
        damaged_reward = damaged_reward - (2 * (last_red_agent_num - next_red_agent_num))
        # damaged_reward -= 200 * (last_red_agent_num - next_red_agent_num)  # 地方是不是不对，如果同一帧，死掉两个实体呢？这里应该是 *
    else:
      # 说明红方飞机数量没有变化
      damaged_reward -= 0.

    reward += damaged_reward

    # 计算攻击奖励
    if last_blue_agent_num - next_blue_agent_num > 0:
      # 说明上一帧蓝方飞机要更多
      if BLUE_INFO['0']['ID'] in self.blue_death:
        attack_reward += 12
      else:
        attack_reward += 2 * (last_blue_agent_num - next_blue_agent_num)
    else:
      # 说明蓝方飞机数量没有变化
      attack_reward += 0.

    reward += attack_reward

    # 计算距离奖励
    for agent_order in range(len(self.ori_message['red']['platforminfos'])):
      cur_id = self.ori_message['red']['platforminfos'][agent_order]['ID']
      if cur_id in self.red_death:
        distance_reward += 0.
      else:
        # 当前的坐标
        cur_x = self.ori_message['red']['platforminfos'][agent_order]['X']
        cur_y = self.ori_message['red']['platforminfos'][agent_order]['Y']
        cur_distance = math.sqrt(cur_x ** 2 + cur_y ** 2)
        # 上一个坐标
        last_x = self.last_ori_message['red']['platforminfos'][agent_order]['X']
        last_y = self.last_ori_message['red']['platforminfos'][agent_order]['Y']
        last_distance = math.sqrt(last_x ** 2 + last_y ** 2)
        # 这个值是很大的
        # distance_reward += 1 if (last_distance - cur_distance) > 0 else -1
        distance_reward += 0.001 if (last_distance - cur_distance) > 0 else -0.001

    reward += (distance_reward * self.decay_coefficient(step=self.forward_step))  # 1.0是衰减系数

    return reward

  def get_reward_test(self, last_obs=None, next_obs=None, if_done=False, flag_no=False):

    reward = 0.
    victory_reward = 0.
    distance_reward = 0.
    attack_reward = 0.
    damage_reward = 0.
    missile_reward = 0.
    protection_reward = 0.
    center_radius = 40000

    # 1. 统计上一帧中，红方战机的数量 & 存在的导弹剩余数量
    last_red_agent_num = len(last_obs['red']['platforminfos'])
    last_red_weapon_num = 0.
    for entity in last_obs['red']['platforminfos']:
      last_red_weapon_num += entity['LeftWeapon']
    # 2. 统计下一帧中，红方战机的数量 & 存在的导弹剩余数量
    next_red_agent_num = len(next_obs['red']['platforminfos'])
    next_red_weapon_num = 0.
    for entity in next_obs['red']['platforminfos']:
      next_red_weapon_num += entity['LeftWeapon']
    # 3. 统计上一帧中，蓝方战机的数量 & 存在的导弹剩余数量
    last_blue_agent_num = len(last_obs['blue']['platforminfos'])
    last_blue_weapon_num = 0.
    for entity in next_obs['blue']['platforminfos']:
      last_blue_weapon_num += entity['LeftWeapon']
    # 4. 统计下一帧中，蓝方战机的数量 & 存在的导弹剩余数量
    next_blue_agent_num = len(next_obs['blue']['platforminfos'])
    next_blue_weapon_num = 0.
    for entity in next_obs['blue']['platforminfos']:
      next_blue_weapon_num += entity['LeftWeapon']

    # victory reward
    if if_done:
      if self.info['Battle Won'] == 0:  # 红方胜利
        # remaining_missiles = sum([entity['LeftWeapon'] for entity in next_obs['red']['platforminfos']])
        victory_reward = 40  # + 2 * remaining_missiles
        print('红方胜利', victory_reward)
      elif self.info['Battle Won'] == 1:  # 蓝方胜利
        used_missiles = sum([entity['LeftWeapon'] for entity in next_obs['red']['platforminfos']])
        # used_missiles = 12 - used_missiles
        victory_reward = -40  # - 2 * used_missiles
        print('蓝方胜利', victory_reward)
      else:  # 平局
        victory_reward = 0

    reward += victory_reward

    # distance_reward
    for agent_order in range(len(self.ori_message['red']['platforminfos'])):
      cur_id = self.ori_message['red']['platforminfos'][agent_order]['ID']
      if cur_id in self.red_death:
        distance_reward += 0.
      else:
        # 当前的坐标
        cur_x = self.ori_message['red']['platforminfos'][agent_order]['X']
        cur_y = self.ori_message['red']['platforminfos'][agent_order]['Y']
        cur_distance = math.sqrt(cur_x ** 2 + cur_y ** 2)  # 计算当前距离

        # 上一个坐标
        last_x = self.last_ori_message['red']['platforminfos'][agent_order]['X']
        last_y = self.last_ori_message['red']['platforminfos'][agent_order]['Y']
        last_distance = math.sqrt(last_x ** 2 + last_y ** 2)  # 计算上一帧的距离

        if cur_distance < center_radius:  # 如果红方单位在中心区域内
          distance_reward += 1 / 500  # 每多存活一帧，得到一个奖励
        else:  # 如果红方单位在中心区域外
          if cur_distance < last_distance:
            distance_reward += 1 / 500  # 向中心区域移动，得到正向奖励
          else:
            distance_reward += -1 / 500  # 远离中心区域，得到负向奖励

    reward += distance_reward

    # attack_reward
    if last_blue_agent_num - next_blue_agent_num > 0:
      # 说明上一帧蓝方飞机要更多
      if BLUE_INFO['0']['ID'] in self.blue_death:
        attack_reward += 50
      #    print('蓝方有人机被击落', attack_reward)
      else:
        attack_reward += 10 * (last_blue_agent_num - next_blue_agent_num)
    #    print('蓝方wu人机被击落', attack_reward)
    else:
      # 说明蓝方飞机数量没有变化
      attack_reward += 0.

    reward += attack_reward

    if last_red_agent_num - next_red_agent_num > 0:
      # 说明上一帧红方飞机要更多
      if RED_INFO['0']['ID'] in self.red_death:
        damage_reward -= 50
      #    print('红方有人机被击落', damage_reward)
      else:
        damage_reward -= 10 * (last_red_agent_num - next_red_agent_num)
    #    print('红方wu人机被击落', damage_reward)
    else:
      # 说明蓝方飞机数量没有变化
      damage_reward += 0.

    reward += damage_reward

    # missile reward
    last_red_missile_num = sum([entity['LeftWeapon'] for entity in last_obs['red']['platforminfos']])
    next_red_missile_num = sum([entity['LeftWeapon'] for entity in next_obs['red']['platforminfos']])
    missile_reward += (last_red_missile_num - next_red_missile_num) * -1

    reward += missile_reward

    # protection_reward
    if not if_done:
      # 获取有人机的位置
      manned_aircraft_position = None
      manned_aircraft_heading = None
      for entity in self.ori_message['red']['platforminfos']:
        if entity['ID'] == RED_INFO['0']['ID']:  # 假设有人机的ID为RED_INFO中的'0'
          manned_aircraft_position = [entity['X'], entity['Y'], entity['Alt']]
          manned_aircraft_heading = entity['Heading']
          break

      if manned_aircraft_position is not None and manned_aircraft_heading is not None:
        # 计算无人机与有人机的距离，并根据距离计算保护奖励
        for i in range(1, 5):
          unmanned_aircraft_position = None
          for entity in self.ori_message['red']['platforminfos']:
            if entity['ID'] == RED_INFO[str(i)]['ID']:
              unmanned_aircraft_position = [entity['X'], entity['Y'], entity['Alt']]
              break

          if unmanned_aircraft_position is not None:
            distance = self.distance_computation(manned_aircraft_position, unmanned_aircraft_position)
            # 假设保护距离阈值为40000
            if distance < 40000:
              protection_reward += 1.0 / 500
            else:
              protection_reward -= 1.0 / 500
            # 如果有人机在无人机前面，给予负向奖励

    reward += protection_reward
    return reward

  def decay_coefficient(self, step=None):
    total_steps = self.episode_limit
    # total_steps = self.episode_limit
    decay_rate = -np.log(0.01) / total_steps  # 衰减速率，这里假设最终值为0.01
    return np.exp(-decay_rate * step)

  def get_simple_reward(self, last_obs=None, next_obs=None, if_done=False, flag_no=False):
    """获取一个简单奖励，胜利：1，平局：0，失败：-1

    :param last_obs: 上一局的态势
    :param next_obs: 本局的态势
    :param if_done: 是否本episode已经结束
    :param flag_no: 用来判断是否红方是通过长时间占领中心区域获胜的

    :return reward: float
    """
    reward = 0.
    if not if_done:
      pass
    else:
      if self.info['Battle Won'] == 0: # 红方胜利
        reward = 1
        if flag_no:
          reward = 0
      elif self.info['Battle Won'] == 1:
        reward = -1
      else:
        reward = 0

    return reward

  def get_obs(self):
    """获取所有Agent的obs = [obs1, obs2, ..., obs5]"""
    agent_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
    return agent_obs

  def get_obs_agent(self, agent_order):
    """获取每个Agent自己的obs"""
    # 已经有一个全局属性self.ori_message
    assert self.ori_message is not None
    # 通过解析这个原始数据构造每个Agent的obs，先通过这个id来判断是否已经死亡，就是判断字典中是否有这个实体
    self.cur_agent_name = RED_INFO[str(agent_order)]['Name']  # 当前Agent的名字
    self.cur_agent_id = RED_INFO[str(agent_order)]['ID']  # 当前Agent的id
    # 查找返回的信息中是否包含当前Agent，如果没有的话说明当前Agent已经死亡
    cur_agent_exist = any(filter(lambda x: x['ID'] == self.cur_agent_id, self.ori_message['red']['platforminfos']))
    if not cur_agent_exist:
      """说明这个Agent已经不存在了"""
      self.red_agent_loc[str(agent_order)] = None
      self.agents_speed[str(agent_order)] = -1
      self.red_death.append(self.cur_agent_id)
    """当前Agent的obs"""
    agent_obs = self.parse_msg(str(agent_order), cur_agent_exist)
    return np.array(agent_obs)

  def parse_msg(self, agent_order, exist=None):
    """对原始态势信息进行解析处理，构造状态"""
    cur_agent_obs = []
    platforminfos_list = self.ori_message['red']['platforminfos']
    if exist:
      """当前Agent还存活"""
      # 获取当前Agent在['platforminfos']中的索引
      cur_index = next(((i, item) for i, item in enumerate(platforminfos_list) if item['ID'] == RED_INFO[str(agent_order)]['ID']), (-1, None))[0]
      ## 拿这个Agent本身的信息
      # identification_own = 0 if platforminfos_list[cur_index]['Identification'] == '红方' else 1
      type_own = platforminfos_list[cur_index]['Type'] / 3.0
      # availability_own = platforminfos_list[cur_index]['Availability']  
      x_own = (platforminfos_list[cur_index]['X'] + 150000) / 300000
      y_own = (platforminfos_list[cur_index]['Y'] + 150000) / 300000
      # alt_own = platforminfos_list[cur_index]['Alt']
      z_own = platforminfos_list[cur_index]['Alt'] / 15000
      # accmag_own = platforminfos_list[cur_index]['AccMag']
      heading_own = platforminfos_list[cur_index]['Heading'] / (2*math.pi) # [0, 2pi]

      # new add
      pitch_own = (platforminfos_list[cur_index]['Pitch'] + (math.pi/2)) / (math.pi)
      # roll_own = platforminfos_list[cur_index]['Roll'] / (2*math.pi)
      speed_own = platforminfos_list[cur_index]['Speed'] / 400  # 根据智能体类型进行归一化
      # status_own = platforminfos_list[cur_index]['Status'] # status返回有问题
      leftweapon_own = platforminfos_list[cur_index]['LeftWeapon'] / 4

      # own_list = [identification_own, type_own, availability_own, x_own, y_own,z_own, accmag_own,
      #       heading_own, pitch_own, roll_own, speed_own, 0., leftweapon_own]
      own_list = [type_own, x_own, y_own, z_own,
            heading_own, pitch_own, speed_own, leftweapon_own]
      cur_agent_obs.extend(own_list)

      self.red_agent_loc[str(agent_order)] = {'X': platforminfos_list[cur_index]['X'], 'Y': platforminfos_list[cur_index]['Y'], 
                          'Z': platforminfos_list[cur_index]['Alt'], 'heading': platforminfos_list[cur_index]['Heading'], 'pitch': platforminfos_list[cur_index]['Pitch']}
      self.agents_speed[str(agent_order)] = platforminfos_list[cur_index]['Speed'] # 存取当前速度
      team_list = []
      ## 拿其他队友的信息，对于某个Agent，它的队友固定为1，2，3，4顺序, 除了agent_order以外
      for i in range(5):
        if i != int(agent_order):
          team_ID = RED_INFO[str(i)]['ID']
          # 先查找这个队友还存在
          team_exist = any(filter(lambda x: x['ID'] == team_ID, platforminfos_list))
          if team_exist:
            # 说明这个队友还活着，拿到这个队友的索引
            team_index = next(((j, item) for j, item in enumerate(platforminfos_list) if item['ID'] == team_ID), (-1, None))[0]
            # identification_team = 0 if platforminfos_list[team_index]['Identification'] == '红方' else 1
            type_team = platforminfos_list[team_index]['Type'] / 3
            # availability_team = platforminfos_list[team_index]['Availability']
            x_team = (platforminfos_list[team_index]['X'] + 150000) / 300000 - x_own
            y_team =(platforminfos_list[team_index]['Y'] + 150000) / 300000 - y_own
            z_team = (platforminfos_list[team_index]['Alt'] / 15000) - z_own
            speed_team = (platforminfos_list[team_index]['Speed'] - 100) / 300 - speed_own
            leftweapon_team = platforminfos_list[team_index]['LeftWeapon'] / 4
            # team_ones = [identification_team, type_team, availability_team,
            #        x_team, y_team, z_team, speed_team]
            team_ones = [type_team, x_team, y_team, z_team, speed_team, leftweapon_team]
            team_list.extend(team_ones)
          else:
            # 说明这个队友已经死亡，那么只需要在对应位置上设为0即可
            # 同时将这个队友的死亡ID记录
            # print(f'死亡的队友有: {self.red_death}')
            self.red_death.append(team_ID) if team_ID not in self.red_death else None
            team_ones = [0., 0., 0., 0., 0., 0.]
            team_list.extend(team_ones)
      cur_agent_obs.extend(team_list)
      trackinfos_list = self.ori_message['red']['trackinfos']
      ## 拿敌人的状态信息
      enemy_list = []
      for i in range(5):
        enemy_ID = BLUE_INFO[str(i)]['ID']
        enemy_exist = any(filter(lambda x: x['ID'] == enemy_ID, trackinfos_list))
        if enemy_exist:
          # 说明这个敌人还活着
          enemy_index = next(((j, item) for j, item in enumerate(trackinfos_list) if item['ID'] == enemy_ID), (-1, None))[0]
          # 找到这个敌人在trackinfos_list中的索引位置
          # identification_enemy = 0 if trackinfos_list[enemy_index]['Identification'] == '红方' else 1
          type_enemy = trackinfos_list[enemy_index]['Type'] / 3
          # availability_enemy = trackinfos_list[enemy_index]['Availability']
          x_enemy = (trackinfos_list[enemy_index]['X'] + 150000) / 300000 - x_own
          y_enemy = (trackinfos_list[enemy_index]['Y'] + 150000) / 300000 - y_own
          # z_enemy = self.llh_to_ecef(trackinfos_list[enemy_index]['Lon'], trackinfos_list[enemy_index]['Lat'],
          #          trackinfos_list[enemy_index]['Alt'])['Z']
          z_enemy = trackinfos_list[enemy_index]['Alt'] / 15000 - z_own
          speed_enemy = (trackinfos_list[enemy_index]['Speed'] - 100) / 300 - speed_own
          # enemy_ones = [identification_enemy, type_enemy, availability_enemy,
          #        x_enemy, y_enemy, z_enemy, speed_enemy]
          enemy_ones = [type_enemy, x_enemy, y_enemy, z_enemy, speed_enemy]
          enemy_list.extend(enemy_ones)

        else:
          # 说明这个敌人已经死掉了
          self.blue_death.append(enemy_ID)
          enemy_ones = [0., 0., 0., 0., 0.]
          enemy_list.extend(enemy_ones)
      cur_agent_obs.extend(enemy_list)

      ## 拿敌人的导弹信息
      # 这里只取对自己有威胁的导弹
      missileinfos_list = self.ori_message['red']['missileinfos']
      cur_fire = []
      for i in range(12):
        # 一共是12枚导弹，
        # 分别查看这12枚导弹是否出现了
        cur_blue_fire_exist = any(filter(lambda x: x['Name'] == BLUE_FIRE_INFO[str(i)]['Name'], missileinfos_list))
        cur_blue_fire_index = next(((j, item) for j, item in enumerate(missileinfos_list) if item['Name'] == BLUE_FIRE_INFO[str(i)]['Name']), (-1, None))[0]
        # 如果出现了，查看是否锁定了自己？
        if cur_blue_fire_exist:
          # 说明这枚弹已经出现了
          # 拿到这枚弹的信息
          cur_blue_fire_info = missileinfos_list[cur_blue_fire_index]
          if cur_blue_fire_info['EngageTargetID'] == RED_INFO[str(agent_order)]['ID']:
            # 锁定了自己
            # cur_blue_fire_Z = self.llh_to_ecef(cur_blue_fire_info['Lon'], cur_blue_fire_info['Lat'],cur_blue_fire_info['Alt'])['Z']
            cur_blue_fire_Z = cur_blue_fire_info['Alt']
            cur_fire.extend([(cur_blue_fire_info['X'] + 150000) / 300000 - x_own,
                     (cur_blue_fire_info['Y'] + 150000) / 300000 - y_own,
                     (cur_blue_fire_Z - 2000) / 30000 - z_own,
                     (cur_blue_fire_info['Speed'] - 400) / 1000 - speed_own]
            )
          else:
            # 未锁定自己
            cur_fire.extend([0., 0., 0., 0.])
        else:
          # 说明这枚弹没有出现
          cur_fire.extend([0., 0., 0., 0.])

      cur_agent_obs.extend(cur_fire)

      # # 对每一个obs_value都进行归一化
      cur_agent_obs = cur_agent_obs - np.mean(cur_agent_obs)
      cur_agent_obs = cur_agent_obs / np.max(np.abs(cur_agent_obs))

    else:
      """当前Agent已经死亡"""
      cur_agent_obs.extend([0.0 for _ in range(105)])

    return cur_agent_obs

  def get_global_state(self):
    """
      重新设计global state
      适用于Qmix版本：
        所有Agent的state都是相同的
      适用于MAPPO版本：
        所有Agent的state都是不同的
    """
    # # for mappo 
    # 为每个智能体制作state
    all_state = []
    def get_ego_info(msg, uid):
      pinfos = msg[Agent.RED]['platforminfos']
      pid = get_info_id(pinfos, RED_INFO[str(uid)]['ID'])
      ptype = pinfos[pid]['Type']
      ptype_oh = [0, 0]
      ptype_oh[ptype-1] = 1
      x = pinfos[pid]['X']
      y = pinfos[pid]['Y']
      z = pinfos[pid]['Alt']
      heading = pinfos[pid]['Heading']
      if self.abundant_info:
        pitch = pinfos[pid]['Pitch']
        # roll = pinfos[pid]['Roll']
        assert abs(pitch) < 3.15, pitch
        # assert abs(roll) < 3.15, roll
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
      gs = [((x+150000)/300000), ((y+150000)/300000), z/15000,
          heading/(2*math.pi), speed/400, leftweapon/4, is_locked, *ptype_oh]
      if self.abundant_info:
        gs += [((pitch+ (math.pi/2))/(math.pi))]
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
          id = info_dict[str(i)]['ID']
          # 说明这个目标还活着，拿到这个目标的索引
          pid = get_info_id(pinfos, id)
          if side == Agent.BLUE:
            tid = get_info_id(msg[side]['trackinfos'], id, check_existence=True)
            if tid is None:
              gs += [0.] * (13 if self.abundant_info else 11)
              continue
            visibility = 1
            gs.append(visibility)
          ptype = pinfos[pid]['Type']
          ptype_oh = [0, 0]
          ptype_oh[ptype-1] = 1
          px = (pinfos[pid]['X']+150000)/300000 - x
          py = (pinfos[pid]['Y']+150000)/300000 - y
          pz = (pinfos[pid]['Alt']/15000) - z
          dist = np.linalg.norm(np.array([pinfos[pid]['X'] - x, pinfos[pid]['Y'] - y, pinfos[pid]['Alt'] - z])) / 424499.705
          heading = pinfos[pid]['Heading']/(2*math.pi)
          speed = pinfos[pid]['Speed']/400
          islocked = pinfos[pid]['IsLocked']
          leftweapon = pinfos[pid]['LeftWeapon']/4
          gs += [px, py, pz, dist, heading, speed, leftweapon, islocked] + ptype_oh
          if self.abundant_info:
            pitch = (pinfos[pid]['Pitch']+ (math.pi/2)) / (math.pi)
            # roll = pinfos[pid]['Roll']
            gs += [pitch]
        else:
          # 说明这个队友已经死亡，那么只需要在对应位置上设为0即可
          # 同时将这个队友的死亡ID记录
          # print(f'死亡的队友有: {self.red_death}')
          gs += [0.] * (12 if self.abundant_info else 10)
          if side == Agent.BLUE:
            gs.append(0)
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
        minfo = list(filter(lambda x: x['Name'] == fire_dict[str(i)]['Name'], minfos))
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
        m_x = (minfo['X']+150000)/300000-x
        m_y = (minfo['Y']+150000)/300000-y
        m_z = (minfo['Alt']/ 15000) - z
        m_speed = minfo['Speed'] / 400
        m_dist = np.linalg.norm(np.array([minfo['X'] - x, minfo['Y'] - y, minfo['Alt'] - z])) / 424499.705
        gs += tid_oh + [m_x, m_y, m_z, m_dist, m_speed]
      return gs
    
    for uid in range(len(self.n_agents)):
      agent_id = [0] * self.n_agents
      agent_id[uid] = 1
      gs = agent_id

      if self.red_alive_mask[uid]:
        ego_gs, x, y, z = get_ego_info(self.ori_message, uid)
        gs += ego_gs
        gs += get_info(self.ori_message, Agent.RED, x, y, z)
        gs += get_info(self.ori_message, Agent.BLUE, x, y, z)
        gs += get_miss_info(self.ori_message, Agent.RED)
        gs += get_miss_info(self.ori_message, Agent.BLUE)
      else:
        gs += [0] * (359 - self.n_agents)

      # return gs
      all_state.append(np.array(gs))
    
    return all_state
    # # for Qmix 
    return np.array(self.obs).flatten()

  def get_state(self):
    """获取全局状态"""
    return self.get_global_state()

  def get_avail_actions(self):
    """
      获得可执行动作列表
      :return available_actions size = (5, 2, 5)
    """
    from copy import deepcopy
    # np1 = np.array([1 for _ in range(20)], dtype=object)  # 9个方向
    # # np1[10:] = 0  # 不采用跟随动作
    # np2 = np.array([1 for _ in range(6)], dtype=object)  # 5个攻击动作+1个不采用攻击动作
    # np3 = np.array([1 for _ in range(3)], dtype=object)
    # avai_cow = np.array([np1, np2, np3], dtype=object)
    avai_cow = np.array([1 for _ in range(17)])
    # avai_mask = np.tile(avai_cow, (5, 1)) # 出现问题，会共享内存
    avai_mask = np.vstack([deepcopy(avai_cow) for _ in range(5)])
    # 分别返回 agents的actions1，actions2
    # avai_dict = {'0': [], '1': [], '2': []}
    avai_dict = []
    for i in range(5):
      """分别遍历五个Agent给出available mask"""
      # 先判断这个Agent是否已经死亡
      if_death = RED_INFO[str(i)]['ID'] in self.red_death  # 查看这个Agent的ID是否在死亡列表self.red_death中
      if if_death:
        # avai_mask[i][0] = np.array([0 for _ in range(20)], dtype=object)
        # avai_mask[i][0][9] = 1  # 10 动作是不执行任何指令，任何情况下都是1
        # avai_mask[i][1] = np.array([0, 0, 0, 0, 0, 1], dtype=object)
        # avai_mask[i][2] = np.array([0, 0, 1], dtype=object)
        avai_mask[i] = np.array([0 for _ in range(17)], dtype=object)
        avai_mask[i][16] = 1
      else:
        # 如果没有死亡，那么移动就可以全部是1,需要判断能不能攻击具体到某个敌方，要进行弹药数量判断和距离判断
        # cur_index = next(((j, item) for j, item in enumerate(self.ori_message['red']['platforminfos']) if RED_INFO[str(i)]['ID'] == self.ori_message['red']['platforminfos'][j])['ID'], (-1, None))[0]
        # next的正确用法。
        cur_index = next(((j, item) for j, item in enumerate(self.ori_message['red']['platforminfos']) if
                  RED_INFO[str(i)]['ID'] == item.get('ID')), (-1, None))[0]

        left_weapon_num = self.ori_message['red']['platforminfos'][cur_index]['LeftWeapon']  # 先判断有没有剩余弹药
        cur_red_agent_loc = [self.ori_message['red']['platforminfos'][cur_index]['X'],
                   self.ori_message['red']['platforminfos'][cur_index]['Y'],
                   self.ori_message['red']['platforminfos'][cur_index]['Alt'],
                   ]
        # add new 跟随指令，当选择跟随的实体已经死亡时，则不能跟随；只要自己没死，则可以跟随自己(区域巡逻)
        # for follow_order in range(10):
        #   if follow_order == i: 
        #     # 说明跟随的是自己，而且自己没有死亡
        #     # 如果巡逻的范围加上自己现在的坐标超出了边界范围
        #     area_length_limit = abs(1000) / 2.0 + abs(self.ori_message['red']['platforminfos'][cur_index]['X'])
        #     area_width_limit = abs(1000) / 2.0 + abs(self.ori_message['red']['platforminfos'][cur_index]['Y'])
        #     if area_length_limit <= 150000 and area_width_limit <= 150000:
        #       avai_mask[i][follow_order+16] = 1
        #     else:
        #       # 如果巡逻的范围加上自己现在的坐标超出了边界范围
        #       avai_mask[i][follow_order+16] = 0
        #     # 一直报区域巡逻边界出界，索性把区域巡逻全部设为0
        #     pass
        #   else:
        #     # 跟随的不是自己，要判断这个实体是否已经死亡
        #     if follow_order >=0 and follow_order < 5:
        #       # 红方队友
        #       team_id = RED_INFO[str(follow_order)]["ID"]
        #       avai_mask[i][follow_order+16] = 1 if team_id not in self.red_death else 0
        #     else:
        #       # 蓝方敌人
        #       enemy_id = BLUE_INFO[str(follow_order-5)]["ID"]
        #       avai_mask[i][follow_order+16] = 1 if enemy_id not in self.blue_death else 0
        #     # print(f'当前跟随的对象不是自己，是{follow_order}')
        #     pass
        
        if left_weapon_num == 0:
          # avai_mask[i][1] = np.array([0, 0, 0, 0, 0, 1], dtype=object)
          avai_mask[i][9:14] = 0  # 
        # else:
        elif left_weapon_num > 0:
          # 进行地方距离判断
          # avai_mask[i][1] 每个index上对应着敌方固定的实体
          cur_ava = np.array([],dtype=object)
          # 分别计算敌方实体的距离
          for j in range(5):
            # 查找蓝方实体的位置，要先判断这个蓝方是不是已经死掉了
            blue_ID = BLUE_INFO[str(j)]['ID']
            if blue_ID in self.blue_death:
              cur_ava = np.append(cur_ava, 0)
            else:
              # 这个蓝方实体还没有死掉
              enemy_index = next(((i_, item) for i_, item in enumerate(self.ori_message['red']['trackinfos']) if
                 item.get('ID') == blue_ID), (-1, None))[0]
              enemy_infos = self.ori_message['red']['trackinfos'][enemy_index]
              enemy_loc = [
                enemy_infos['X'],
                enemy_infos['Y'],
                enemy_infos['Alt']
              ]
              # 判断距离
              if self.distance_computation(cur_red_agent_loc, enemy_loc) <= 80000:
                cur_ava = np.append(cur_ava, 1)
              else:
                cur_ava = np.append(cur_ava, 0)

          # # 加入射击间隔
          agent_shoot_interval = self.shoot_interval[i]
          agent_shoot_legal = np.ones_like(agent_shoot_interval)
          agent_shoot_legal[agent_shoot_interval != 0] = 0
          # # print(f'agent_shoot_legal', agent_shoot_legal)
          # # print(f'original cur_ava: {cur_ava}')
          new_cur_ava = np.zeros_like(cur_ava)
          new_cur_ava[(cur_ava == 1) & (agent_shoot_legal == 1)] = 1
          # # print(f'cur_ava', new_cur_ava)
          # # cur_ava.append(1)  # 最后一维是不执行任何攻击工作
          # cur_ava = np.append(new_cur_ava, 1)
          # # cur_ava = np.append(cur_ava, 1)
          # avai_mask[i][1] = cur_ava
          avai_mask[i][9:14] = new_cur_ava    
      
      avai_dict.append(avai_mask[i])
      # avai_mask[i][2] = np.array([1, 1, 1], dtype=object) # 加速，减速，速度不变永远都可以执行

    # for i in range(5):
    #   avai_dict['0'].append(list(avai_mask[i][0]))
    #   avai_dict['1'].append(list(avai_mask[i][1]))
    #   avai_dict['2'].append(list(avai_mask[i][2]))

    return avai_dict

  def distance_computation(self, point_1=None, point_2=None):
    """
      计算两个点之间的距离
      :param point_1 第一个点的 [X，Y，Z]
      :param point_2 第二个点的 [X，Y，Z]

      :return distance: int
    """
    return math.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2 + (point_1[2] - point_2[2])**2)

  def close_environ(self):
    """关闭仿真引擎"""
    self._end()

  def render(self):
    """渲染视频"""
    pass

  def save_replay(self):
    pass

  def get_total_actions(self):
    return self.action_space # self.action_space_1 + self.action_space_2 + self.action_space_3

  def get_obs_size(self):
    return self.observation_space # 124

  def get_state_size(self):
    return self.n_agents * self.get_obs_size()

  def get_env_info(self):
    env_info = {"state_shape": self.get_state_size(), # 620
      "obs_shape": self.get_obs_size(), # 124
      "n_actions": self.get_total_actions(), # 9 + 6 = 15
      # "n_actions_1": self.action_space_1, # 航线行进
      # "n_actions_2": self.action_space_2, # 攻击
      # "n_actions_3": self.action_space_3, # 加速减速
      "n_agents": self.n_agents, # 5
      "episode_limit": self.episode_limit
    }

    return env_info

  def render(self):
    pass

  def get_stats(self):
    stats = {
      "battles_won": self.battles_won,
      "battles_game": self.battles_game,
      "win_rate": self.battles_won / self.battles_game
    }
    return stats


if __name__ == '__main__':
  """测试环境类"""
  print('testing environment')
  address = ADDRESS['ip'] + ":" + str(ADDRESS['port'])
  print('address:', address)
  test_env = HuaRuBattleEnvWrapper(config['agents'], address)
  test_env.reset()
  # test_env.get_available_actions()
