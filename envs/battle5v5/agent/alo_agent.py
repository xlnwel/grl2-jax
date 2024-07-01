# """
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: alo_agent.py
# @Time: 2024/4/15 下午4:14
# @Author: ZhengtaoCao
# @Description:
# """

"""
!/usr/bin/python3
-*- coding: utf-8 -*-
@FileName: alo_agent.py
@Time: 2024/4/15 下午4:14
@Author: ZhengtaoCao
@Description:
"""
from envs.battle5v5.agent.agent import Agent
import torch as th
from envs.battle5v5.env.env_cmd import CmdEnv
from envs.battle5v5.utils.test_utils import CalNineDir, Trans

fly_config = {
            # 有人机
            0: {
                'move_min_speed': 150,
                'move_max_speed': 300,
                'move_max_acc': 1,
                'move_max_g': 4,
                'area_max_alt': 14000,
                'attack_range': 1,
                'launch_range': 80000
            },
            # 无人机
            1: {
                'move_min_speed': 100,
                'move_max_speed': 300,
                'move_max_acc': 2,
                'move_max_g': 3,
                'area_max_alt': 10000,
                'attack_range': 1,
                'launch_range': 60000
            }
        }

class AloAgent(Agent):
    """
        自定义算法智能体，接收网络的标量动作，输出仿真能执行的指令: 移动动作 + 攻击动作
    """
    def __init__(self, name, config):
        """
        初始化信息
        :param name:阵营名称
        :param config:阵营配置信息
        """
        super(AloAgent, self).__init__(name, config["side"])
        self.cur_msg = None
        self.cur_agents_speed = None  # 记录每个episode中的智能体速度
        # self.agents
        self.flag = True
    
    def reset(self, **kwargs):
        """当引擎重置会调用,选手需要重写此方法,来实现重置的逻辑"""
        # print(f'Init agents speed....')
        self.cur_agents_speed = [fly_config[0]['move_max_speed'], fly_config[1]['move_max_speed'], 
                                  fly_config[1]['move_max_speed'], fly_config[1]['move_max_speed'], 
                                  fly_config[1]['move_max_speed']
                                ]
        # print(self.cur_agents_speed)

    def make_actions(self, actions=None, parse_msg=None, side='red'):
        """
            接受Environment传入的动作，生成每个Agent的仿真执行指令
            :params actions Agents policy net outputs
            :params parse_msg 解析后的态势信息
            :params side Agents所属方
        """
        self.cur_msg = parse_msg
        red_agents_pre_loc = parse_msg['agent_pre_loc']
        cmd_list = []
        sim_time = parse_msg['sim_time']
        # if sim_time >= 0 and sim_time % 15 == 0:
        for i in range(len(actions)):
            # 当前Agent的仿真ID
            agent_id = parse_msg['red_info'][i]['ID']
            # 对于每个Agent分别产生动作
            cur_agent_pre_loc = red_agents_pre_loc[i]  # 拿到当前Agent上一帧的动作
            agents_speed = parse_msg["agent_speed"]  # 当前智能体的speed
            cur_agent_speed = agents_speed[i]
            # -> 转换为智能体的一个动作
            cur_agent_actions = actions[i]   # 拿到当前Agent的网络动作输出
            
            if th.is_tensor(cur_agent_actions):
                # cur_agent_actions = [int(a) for a in cur_agent_actions]
                cur_agent_actions = int(cur_agent_actions)
            else:
                cur_agent_actions = cur_agent_actions
                        # if sim_time < 150:
            #     cur_agent_actions = 0
            
            # ## 统一做动作测试
            # # if sim_time < 200:
            # #     cur_agent_actions = 0
            # # elif sim_time >= 200 and sim_time <= 230:
            # #     print('turn right, turn right, turn right, turn right,')
                
            # #     cur_agent_actions = 17 # 4 # 3 ## 2# 1 # 8
            # #     self.flag = False
            # # elif sim_time > 230 and sim_time <= 400:
            # #     print('continue ...')
            # #     cur_agent_actions = 0
            # # elif sim_time > 400 and sim_time <= 430:
            # #     # cur_agent_actions = 16 # 1 # 2 # 2 # 1 # 7
            # #     cur_agent_actions = 16 # 4 # 3 ## 2# 1 # 8
            # #     self.flag = False
            # # elif sim_time > 430:
            # #     cur_agent_actions = 0
            # # else:
            # #     return cmd_list
            # if sim_time == 240:
            #     cur_agent_actions = 3
            # else:
            #     cur_agent_actions = 0
            
            # 获取位置
            try:
                last_x, last_y, last_z, last_heading, last_pitch = \
                    cur_agent_pre_loc['X'], cur_agent_pre_loc['Y'], cur_agent_pre_loc['Z'], cur_agent_pre_loc['heading'], cur_agent_pre_loc['pitch']
            except:
                # 实体死亡，传过来是
                last_x, last_y, last_z, last_heading, last_pitch = None, None, None, None, None
            
            if 0 <= cur_agent_actions < 9:
                # 1. 每个Agent产生移动动作
                # move_action = cur_agent_actions[0]
                # 先拿到上一帧的X，Y，Z坐标
                # 此帧智能体做出机动动作
                # print(f'cur agent :{agent_id} new action: {move_action}')
                move_action = cur_agent_actions
                # if move_action == 9:
                #     pass
                # elif move_action >=0 and move_action < 9:
                    # 1）每个Agent产生方向微操控制指令
                    # 得到一个新的经纬度坐标（弧度）
                move_new_loc = self.new_loc(last_x, last_y, last_z, last_heading, last_pitch, move_action)
                cmd_list.append(self.make_move_cmd(move_new_loc, agent_id, self.cur_agents_speed[i]))
                pass
                # else:
                #     # 2）每个智能体产生跟随和区域巡航指令, i代表第i个智能体
                #     # 0：跟随红方有人机
                #     # 1: 跟随红方无人机1
                #     # 2: 跟随红方无人机2
                #     # 3: 跟随红方无人机2
                #     # 4: 跟随红方无人机3 
                #     # 5 ~ 9 跟随蓝方
                #     follow_action = int(move_action) - 10 
                #     followed_agent_id = {
                #         "0": 1, # ID
                #         "1": 2, 
                #         "2": 11,
                #         "3": 12,
                #         "4": 13,
                #         "5": 6,
                #         "6": 14,
                #         "7": 15,
                #         "8": 16,
                #         "9": 17
                #     }
                #     if i == follow_action:
                #         # follow的人是自己，则需要转换为区域巡航指令
                #         areapatrol_cmd = self.make_areapatrolparam_cmd(agent_id, [last_x, last_y, last_z])
                #         cmd_list.append(areapatrol_cmd)
                #         pass
                #     else:
                #         # 产生跟随指令
                #         global fly_config
                #         if agent_id == 1: # 有人机
                #             fly_param = fly_config['0']
                #         else:
                #             fly_param = fly_config['1']
                #         follow_cmd = self.make_follow_cmd(agent_id, followed_agent_id[str(follow_action)], fly_param)
                #         cmd_list.append(follow_cmd)
        

            # Agent产生攻击动作 9 ~ 13
            elif 9 <= cur_agent_actions < 14:
                # 此帧智能体做出攻击动作
                # attack_action = cur_agent_actions[1]
                attack_action = cur_agent_actions
                # attack_action = 9 # 发现把红方攻击目标都改成蓝方有人机后，正样本的探索效率能显著提高。
                attack_cmd = self.make_attack_cmd(agent_id, attack_action)
                if attack_cmd is not None:
                    cmd_list.append(attack_cmd)

            elif 14 <= cur_agent_actions < 16:
                # 14: 加速
                # 15：减速
                # 3. 每个智能体产生加减速动作
                # speed_con_action = cur_agent_actions[2]
                speed_con_action = cur_agent_actions
                if cur_agent_speed <= 0:
                    # 实体已经死亡
                    pass
                else:
                    # speed_con_action: 0 加速 1 减速 2 速度不变
                    # if speed_con_action == 0 or speed_con_action == 1:
                    if speed_con_action == 14:
                        speed_change = (9.8 * 10) if agent_id == 1 else (9.8 * 20) # 有人机按照加速度增加9.8 m/s， 无人机增加9.8 * 2 m/s
                    elif speed_con_action == 15:
                        speed_change = (9.8 * (-10)) if agent_id == 1 else (9.8 * (-20))
                    
                    self.cur_agents_speed[i] = self.cur_agents_speed[i] + speed_change
                    
                    # 速度边界处理
                    # 有人机：[150, 400]
                    # 无人机：[100, 300]
                    if agent_id == 1: 
                        new_speed = max(150, min(self.cur_agents_speed[i], 400))
                    else:
                        new_speed = max(100, min(self.cur_agents_speed[i], 300))

                    self.cur_agents_speed[i] = new_speed

                    speed_con_cmd = self.make_changes_speed_cmd(agent_id,
                                                                self.cur_agents_speed[i],
                                                                1.0 if agent_id == 1 else 2.0,
                                                                4 if agent_id==1 else 3)
                    cmd_list.append(speed_con_cmd)

            else:
                pass

        return cmd_list

    def new_loc(self, x, y, z, heading, pitch, action) -> dict:
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
        all_dir_choice = cal_dir.get_all_nine_dir(pitch, heading, x, y, z)
        new_point = all_dir_choice[str(int(action))] # 拿到新的（纬度、经度、高度）

        return new_point


    def make_move_cmd(self, new_loc: dict = None, agent_id: int = None, cur_agent_speed: float = None):
        """
            将agent的网络输出转换为移动指令
            :param new_loc: 新的坐标X，Y，Z
            :param agent_id: 当前Agent ID
        """
        global fly_config
        if agent_id == 1: # 有人机
            fly_param = fly_config[0]
        else:
            fly_param = fly_config[1]
        move_cmd = CmdEnv.make_linepatrolparam(agent_id,
                                               [new_loc],
                                               cur_agent_speed,
                                               fly_param['move_max_acc'],
                                               fly_param['move_max_g']
        )
        return move_cmd


    def make_follow_cmd(self, agent_id, target_id, fly_param, area_speed=None):
        """
            生成跟随指令
            :param agent_id: 当前Agent ID
            :param target_id: 跟随目标ID
        """
        """
            将agent的网络输出转换为移动指令
            :param new_loc: 新的坐标X，Y，Z
            :param agent_id: 当前Agent ID
        """
        follow_cmd = CmdEnv.make_followparam(agent_id, target_id,
                                               area_speed,
                                               fly_param['move_max_acc'],
                                               fly_param['move_max_g'])
        return follow_cmd

    def make_areapatrolparam_cmd(self, agent_id, new_point, area_speed=None):
        """
            生成区域巡逻指令, 当智能体输出对自己的跟随指令时，则转为执行区域巡逻指令
            :param receiver: Handle ID
            :param x: 区域中心坐标x坐标
            :param y: 区域中心坐标y坐标
            :param z: 区域中心坐标z坐标
            :param area_length: 区域长
            :param area_width: 区域宽
            :param cmd_speed: 指令速度
            :param cmd_accmag: 指令加速度
            :param cmd_g: 指令过载
        """
        x, y, z = new_point[0], new_point[1], new_point[2]  # list
        global fly_config
        if agent_id == 1: # 有人机
            fly_param = fly_config['0']
            z = max(2500, min(z, 14500)) # 限制巡逻高度
        else:
            fly_param = fly_config['1']
            z = max(2500, min(z, 9500)) # 限制巡逻高度
        # try
        areapatrolparam_cmd = CmdEnv.make_areapatrolparam(agent_id,
                                                          x,
                                                          y,
                                                          z,
                                                          1000, # 区域长度4000
                                                          1000, # 区域宽度
                                                          area_speed,
                                                          fly_param['move_max_acc'],
                                                          fly_param['move_max_g'])
        return areapatrolparam_cmd  # 区域巡航指令


    def make_attack_cmd(self, agent_id=None, action=None):
        """
            将agent的网络输出转换为攻击指令
            :params agent_id: 当前agent的id是哪一个
            :params action: 代表的是哪个攻击动作，就是打击敌方哪个实体i 9~13
        """
        action -= 9 # 转换为 0 ~ 4
        attack_cmd = None
        # if action == 5:
        #     pass
        # else:
        # 拿到攻击敌方的ID
        target_ID = self.cur_msg['blue_info'][int(action)]['ID']
        attack_cmd = CmdEnv.make_attackparam(agent_id, target_ID, fire_range=1)

        return attack_cmd



    def make_init_cmd(self, RED_INFO, INIT_LOC):
        """
            制作初始化位置训练仿照HR1，有人机在最后方
            :params parse_msg: 使用红方ID
        """
        global fly_config
        cmd_list = []
        for ia in range(5):
            if ia == 0:
                speed = fly_config[0]['move_max_speed']
            else:
                speed = fly_config[1]['move_max_speed']

            cur_iagnet_id = RED_INFO[ia]['ID']
            cmd_list.append(CmdEnv.make_entityinitinfo(
                cur_iagnet_id,
                INIT_LOC[ia]['X'],
                INIT_LOC[ia]['Y'],
                INIT_LOC[ia]['Z'],
                speed,
                90))

        return cmd_list
    

    def make_changes_speed_cmd(self, agent_id, cmd_speed, cmd_accmag=1.0, cmd_g=1.0):
        """
            修改飞机飞行参数指令，改变飞机速度
        """
        return CmdEnv.make_motioncmdparam(agent_id,
                                          1,
                                          cmd_speed,
                                          cmd_accmag,
                                          cmd_g)
        
