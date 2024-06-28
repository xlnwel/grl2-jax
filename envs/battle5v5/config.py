"""
@FileName：config.py
@Description：
@Time：2021/5/9 下午8:08
@Copyright：©2011-2021 北京华如科技股份有限公司
"""
from envs.battle5v5.agent.demo_agent import DemoAgent
from envs.battle5v5.agent.agent import Agent
# from .agent.MyAgent.test_agent import TestAgent
import sys

from envs.battle5v5.agent.alo_agent import AloAgent
from envs.battle5v5.agent.blue_alo_agent import BlueAloAgent
from envs.battle5v5.agent.demo_agent import DemoAgent
from envs.battle5v5.agent.MyAgent.test_agent import TestAgent
# from env.battle5v5.agent.HR1.HR1 import HR1
# 蓝方超强规则Agents
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print(sys.path)
from envs.battle5v5.agent.HR1 import HR1
# 是否启用host模式,host仅支持单个xsim
ISHOST = False

# 为态势显示工具域分组ID  1-1000
HostID = 1

IMAGE = "xsim:v8.0"

# 加速比 1-100
TimeRatio = 100

# 范围:0-100 生成的回放个数 (RTMNum + 2),后续的回放会把之前的覆盖掉.
RTMNum = 1

config = {
    "episode_time": 100,   # 训练次数
    "step_time": 1, # 想定步长
    'agents': {
            'red': AloAgent,
            # 'blue': TestAgent,
            'blue': HR1.HR1,
            'self_play_blue': BlueAloAgent
            }
}

# 进程数量
POOL_NUM = 10

# 启动XSIM的数量
XSIM_NUM = 6

ADDRESS = {
    "ip": "127.0.0.1",
    "port": 12344
}


class Agent:
    BLUE = 'blue'
    RED = 'red'


BLUE_INFO = {
    0: {'Name': '蓝有人机', 'ID': 6},
    1: {'Name': '蓝无人机1', 'ID': 14},
    2: {'Name': '蓝无人机2', 'ID': 15},
    3: {'Name': '蓝无人机3', 'ID': 16},
    4: {'Name': '蓝无人机4', 'ID': 17},
}
RED_INFO = {
    0: {'Name': '红有人机', 'ID': 1},
    1: {'Name': '红无人机1', 'ID': 2},
    2: {'Name': '红无人机2', 'ID': 11},
    3: {'Name': '红无人机3', 'ID': 12},
    4: {'Name': '红无人机4', 'ID': 13},
}
BLUE_FIRE_INFO = {
    0: {'Name': '空空导弹_1(蓝有人机_武器系统_1)', 'ID': 2147483668},
    1: {'Name': '空空导弹_2(蓝有人机_武器系统_1)', 'ID': 2147483669},
    2: {'Name': '空空导弹_3(蓝有人机_武器系统_1)', 'ID': 2147483670},
    3: {'Name': '空空导弹_4(蓝有人机_武器系统_1)', 'ID': 2147483671},
    4: {'Name': '空空导弹_1(蓝无人机1_武器系统_1)', 'ID': 2147483657},
    5: {'Name': '空空导弹_2(蓝无人机1_武器系统_1)', 'ID': 2147483659},
    6: {'Name': '空空导弹_1(蓝无人机2_武器系统_1)', 'ID': 2147483652},
    7: {'Name': '空空导弹_2(蓝无人机2_武器系统_1)', 'ID': 2147483654},
    8: {'Name': '空空导弹_1(蓝无人机3_武器系统_1)', 'ID': 2147483660},
    9: {'Name': '空空导弹_2(蓝无人机3_武器系统_1)', 'ID': 2147483662},
    10: {'Name': '空空导弹_1(蓝无人机4_武器系统_1)', 'ID': 2147483648},
    11: {'Name': '空空导弹_2(蓝无人机4_武器系统_1)', 'ID': 2147483650},
}

RED_FIRE_INFO = {
    0: {'Name': '空空导弹_1(红有人机_武器系统_1)', 'ID': 2147483668},
    1: {'Name': '空空导弹_2(红有人机_武器系统_1)', 'ID': 2147483669},
    2: {'Name': '空空导弹_3(红有人机_武器系统_1)', 'ID': 2147483670},
    3: {'Name': '空空导弹_4(红有人机_武器系统_1)', 'ID': 2147483671},
    4: {'Name': '空空导弹_1(红无人机1_武器系统_1)', 'ID': 2147483657},
    5: {'Name': '空空导弹_2(红无人机1_武器系统_1)', 'ID': 2147483659},
    6: {'Name': '空空导弹_1(红无人机2_武器系统_1)', 'ID': 2147483652},
    7: {'Name': '空空导弹_2(红无人机2_武器系统_1)', 'ID': 2147483654},
    8: {'Name': '空空导弹_1(红无人机3_武器系统_1)', 'ID': 2147483660},
    9: {'Name': '空空导弹_2(红无人机3_武器系统_1)', 'ID': 2147483662},
    10: {'Name': '空空导弹_1(红无人机4_武器系统_1)', 'ID': 2147483648},
    11: {'Name': '空空导弹_2(红无人机4_武器系统_1)', 'ID': 2147483650},
}
