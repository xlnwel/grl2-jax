"""
!/usr/bin/python3
-*- coding: utf-8 -*-
@FileName: alo_agent.py
@Time: 2024/4/15 下午4:14
@Author: ZhengtaoCao
@Description:
"""
from env.battle5v5.agent.agent import Agent
import torch as th
from env.battle5v5.env.env_cmd import CmdEnv
from env.battle5v5.utils.test_utils import CalNineDir, Trans

class BlueAloAgent(Agent):
    """
        自定义算法智能体，接收网络的标量动作，输出仿真能执行的指令: 移动动作 + 攻击动作
        适用于self-play模式，为蓝方Agents转换指令!
    """
    def __init__(self, name, config):
        """
        初始化信息
        :param name:阵营名称
        :param config:阵营配置信息
        """
        super(BlueAloAgent, self).__init__(name, config["side"])
        self.cur_msg = None

    def reset(self, **kwargs):
        """当引擎重置会调用,选手需要重写此方法,来实现重置的逻辑"""
        pass

    def make_actions(self, actions=None, parse_msg=None, side='blue'):
        """
            接受Environment传入的动作，生成每个self-play蓝方Agent的仿真执行指令
            :params actions Agents policy net outputs, 当前actions已经是分别对应蓝方0~4个Agent
            :params parse_msg 解析后的态势信息
            :params side Agents所属方
        """
        # print('self-play blue agents start making actions....')
        self.cur_msg = parse_msg
        blue_agents_pre_loc = parse_msg['blue_agent_pre_loc']
        cmd_list = []
        for i in range(len(actions)):
            # 当前Agent的仿真ID
            agent_id = parse_msg['blue_info'][str(i)]['ID']
            # 对于每个Agent分别产生动作
            cur_agent_pre_loc = blue_agents_pre_loc[str(i)]  # 拿到当前Agent上一帧的动作
            cur_agent_actions = actions[i]   # 拿到当前Agent的网络动作输出
            if th.is_tensor(cur_agent_actions):
                cur_agent_actions = [int(a) for a in cur_agent_actions]
            else:
                cur_agent_actions = cur_agent_actions

            move_action = cur_agent_actions[0]
            if move_action == 9:
                pass
            else:
                # 1. 每个Agent产生移动动作
                # 先拿到上一帧的X，Y，Z坐标
                try:
                    last_x, last_y, last_z, last_heading, last_pitch, last_roll = \
                        cur_agent_pre_loc['X'], cur_agent_pre_loc['Y'], cur_agent_pre_loc['Z'], cur_agent_pre_loc['heading'], cur_agent_pre_loc['pitch'], cur_agent_pre_loc['roll']
                except:
                    print(f'当前Agent坐标位置是空的.')
                    last_x, last_y, last_z, last_heading, last_pitch, last_roll = None, None, None, None, None, None
                # 得到一个新的经纬度坐标（弧度）
                move_new_loc = self.new_loc(last_x, last_y, last_z, last_heading, last_pitch, last_roll, move_action)
                cmd_list.append(self.make_move_cmd(move_new_loc, agent_id))

            # 2. 每个Agent产生攻击动作
            attack_action = cur_agent_actions[1]
            attack_cmd = self.make_attack_cmd(agent_id, attack_action)
            if attack_cmd is not None:
                cmd_list.append(attack_cmd)
        # print('blue agents have achieved some actions.......')
        return cmd_list

    def new_loc(self, x, y, z, heading, pitch, roll, action) -> dict:
        """
            计算Agent新的坐标位置，先不用考虑偏转角
            :params lon: 上一帧的经度
            :params lat: 上一帧的纬度
            :params alt: 上一帧的高度
            :params action: 采取的动作
            :return dict {'X': X, 'Y': Y, 'Z': Z}
        """
        if x == None and y == None and z == None:
            """说明当前Agent已经死掉了"""
            new_point = {'X': 0, 'Y': 0, 'Z': 0}
            return new_point

        new_loc = None
        ## 调用两个工具类，计算坐标
        # 调用utils函数，可以得到9个方向的经纬高（弧度）新坐标
        cal_dir = CalNineDir()
        all_dir_choice = cal_dir.get_all_nine_dir(heading, x, y, z)
        new_point = all_dir_choice[str(int(action))] # 拿到新的（纬度、经度、高度）

        return new_point


    def make_move_cmd(self, new_loc: dict = None, agent_id: int = None):
        """
            将agent的网络输出转换为移动指令
            :param new_loc: 新的坐标X，Y，Z
            :param agent_id: 当前Agent ID
        """
        fly_config = {
            # 有人机
            '0': {
                'move_min_speed': 150,
                'move_max_speed': 350,
                'move_max_acc': 1,
                'move_max_g': 6,
                'area_max_alt': 14000,
                'attack_range': 1,
                'launch_range': 80000
            },
            # 无人机
            '1': {
                'move_min_speed': 100,
                'move_max_speed': 300,
                'move_max_acc': 2,
                'move_max_g': 12,
                'area_max_alt': 10000,
                'attack_range': 1,
                'launch_range': 60000
            }
        }
        if agent_id == 6: # 有人机
            fly_param = fly_config['0']
        else:
            fly_param = fly_config['1']
        move_cmd = CmdEnv.make_linepatrolparam(agent_id,
                                               [new_loc],
                                               fly_param['move_max_speed'],
                                               fly_param['move_max_acc'],
                                               fly_param['move_max_g']
        )
        return move_cmd


    def make_attack_cmd(self, agent_id=None, action=None):
        """
            将agent的网络输出转换为攻击指令
            :params agent_id: 当前agent的id是哪一个
            :params action: 代表的是哪个攻击动作，就是打击敌方哪个实体i
        """
        attack_cmd = None
        if action == 5:
            pass
        else:
            # 拿到攻击敌方的ID
            target_ID = self.cur_msg['red_info'][str(int(action))]['ID']
            attack_cmd = CmdEnv.make_attackparam(agent_id, target_ID, fire_range=1)

        return attack_cmd



    def make_init_cmd(self):
        """
            制作初始化位置训练仿照HR1，有人机在最后方
            :params parse_msg: 使用红方ID
        """
        BLUE_INFO = {
            '0': {'Name': '蓝有人机', 'ID': 6},
            '1': {'Name': '蓝无人机1', 'ID': 14},
            '2': {'Name': '蓝无人机2', 'ID': 15},
            '3': {'Name': '蓝无人机3', 'ID': 16},
            '4': {'Name': '蓝无人机4', 'ID': 17},
        }
        INIT_LOC = {
            '0': {'X': 125000, 'Y': -10000, 'Z': 9000},
            '1': {'X': 125000, 'Y': 5000, 'Z': 9000},
            '2': {'X': 130000, 'Y': 5000, 'Z': 9000},
            '3': {'X': 125000, 'Y': -5000, 'Z': 9000},
            '4': {'X': 130000, 'Y': -5000, 'Z': 9000},
        }
        fly_config = {
            # 有人机
            '0': {
                'move_min_speed': 150,
                'move_max_speed': 350,
                'move_max_acc': 1,
                'move_max_g': 6,
                'area_max_alt': 14000,
                'attack_range': 1,
                'launch_range': 80000
            },
            # 无人机
            '1': {
                'move_min_speed': 100,
                'move_max_speed': 300,
                'move_max_acc': 2,
                'move_max_g': 12,
                'area_max_alt': 10000,
                'attack_range': 1,
                'launch_range': 60000
            }
        }
        cmd_list = []
        for ia in range(5):
            if ia == 0:
                speed = fly_config['0']['move_max_speed']
            else:
                speed = fly_config['1']['move_max_speed']

            cur_iagnet_id = BLUE_INFO[str(ia)]['ID']
            cmd_list.append(CmdEnv.make_entityinitinfo(cur_iagnet_id,
                                                       INIT_LOC[str(ia)]['X'],
                                                       INIT_LOC[str(ia)]['Y'],
                                                       INIT_LOC[str(ia)]['Z'],
                                                       speed,
                                                       270))

        return cmd_list
