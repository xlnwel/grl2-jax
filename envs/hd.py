import json
import numpy as np
from typing import Optional

from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from hddf2sim.hddf2sim import HDDF2Sim
from hddf2sim.conf import default_conf

# from agents.team_blue_raw.blue_agent_new import Agent as BlueAgent
from agents.team_blue.blue_agent_demo import Agent as BlueAgent
from agents.team_blue.blue_agent_demo import Agent as RedAgent

class HDEnv(MultiAgentEnv):
    def __init__(self, env):
        super().__init__()
        self.env = env

        with open("scen.json", "r") as fin:
            scen = json.load(fin)
        sim = HDDF2Sim(scen, use_tacview=False, save_replay=False, replay_path="replay.acmi")
        self.sim = sim

        self.num_agents = 6 * 2
        self.unit_indices = None
        self.reversed_indices = None
        self.red_agent = None # 先载入最强的规则智能体做demo
        self.blue_agent = None
        self.red_previous_alive_num = 6
        self.blue_previous_alive_num = 6
        
        self.observation_space = Box(float("-inf"), float("inf"), (4,))
        self.action_space = Discrete(2)  # 开火、不开
        self._agent_ids = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.sim.reset()
        # 假设sim.units已经是一个列表或其他容器，包含了所有映射关系 {8315: '00', 7315: '01'}
        self.unit_indices = {self.sim.units[i].ind: f'{i:02}' for i in range(self.num_agents)}
        # 反转字典：将字符串序号作为键，整数 ind 作为值
        self.reversed_indices = {v: k for k, v in self.unit_indices.items()}

        self.red_agent = RedAgent('red')
        self.blue_agent = BlueAgent('blue')
        self.red_previous_alive_num = 6
        self.blue_previous_alive_num = 6

        return self._get_obs(), {}

    def step(self, action):
        red_obs = self.sim.get_obs(side='red')
        red_cmd_dict = self.red_agent.step(red_obs)

        # 遍历red_cmd_dict中的每一个条目
        for ind, cmd in list(red_cmd_dict.items()):
            if ind in self.unit_indices:  # 确保ind存在于映射中
                unit_key = self.unit_indices[ind]  # 获取对应的00-11的序号
                if unit_key in action:
                    action_value = action[unit_key]
                    # 根据action_value的值决定如何处理cmd
                    if action_value == 0:
                        # 如果action_value为0，则只保留control属性，不要weapon属性
                        if 'control' in cmd:
                            red_cmd_dict[ind] = {'control': cmd['control']}
                        else:
                            # 如果没有control，按理说应该是不可能的
                            raise ValueError
                    # 如果action_value为1，则保留cmd的所有属性
        self.sim.send_commands(red_cmd_dict, cmd_side='red')
        
        blue_obs = self.sim.get_obs(side='blue')
        blue_cmd_dict = self.blue_agent.step(blue_obs)

        # 遍历red_cmd_dict中的每一个条目
        for ind, cmd in list(blue_cmd_dict.items()):
            if ind in self.unit_indices:  # 确保ind存在于映射中
                unit_key = self.unit_indices[ind]  # 获取对应的00-11的序号
                if unit_key in action:
                    action_value = action[unit_key]
                    # 根据action_value的值决定如何处理cmd
                    if action_value == 0:
                        # 如果action_value为0，则只保留control属性，不要weapon属性
                        if 'control' in cmd:
                            blue_cmd_dict[ind] = {'control': cmd['control']}
                        else:
                            # 如果没有control，按理说应该是不可能的
                            raise ValueError
                    # 如果action_value为1，则保留cmd的所有属性
        self.sim.send_commands(blue_cmd_dict, cmd_side='blue')

        # 开始处理gym接口的输出情况
        self.sim.step()
        obs = self._get_obs()
        rewards = {}
        terminateds = {"__all__": self.sim.done}

        # 遍历红方智能体
        for i in range(6):  # 红方对应00-05
            key = f'{i:02}'
            ind = self.reversed_indices.get(key, None)
            assert ind is not None
            if ind not in red_obs.my_planes:
                # 如果没有在red_obs中找到，表示智能体已死亡
                terminateds[key] = True
                rewards[key] = 0  # 死亡智能体的奖励为0
            else:
                terminateds[key] = False
                # 奖励为当前存活的红方智能体的数量变化
                red_current_alive_num = len(red_obs.my_planes)
                rewards[key] = red_current_alive_num - self.red_previous_alive_num
                self.red_previous_alive_num = red_current_alive_num
        
        # 遍历蓝方智能体
        for i in range(6, 12):  # 蓝方对应06-11
            key = f'{i:02}'
            ind = self.reversed_indices.get(key, None)
            assert ind is not None
            if ind not in blue_obs.my_planes:
                # 如果没有在blue_obs中找到，表示智能体已死亡
                terminateds[key] = True
                rewards[key] = 0  # 死亡智能体的奖励为0
            else:
                terminateds[key] = False
                # 奖励为当前存活的蓝方智能体的数量变化
                blue_current_alive_num = len(blue_obs.my_planes)
                rewards[key] = blue_current_alive_num - self.blue_previous_alive_num
                self.blue_previous_alive_num = blue_current_alive_num

        truncateds = dict(
            {f'{i:02}': self.sim.done for i in range(self.num_agents)}, **{"__all__": self.sim.done}
            )
        return obs, rewards, terminateds, truncateds, {}

    def render(self, mode=None) -> None:
        if mode == "human":
            print("Please use tacview!!!")

    def _get_obs(self):
        current_obs_dict = {}

        red_obs_dict = self.sim.get_obs(side='red')
        for key,value in red_obs_dict.my_planes.items():
            if key in self.unit_indices:
                obs_array = np.array([value.x * 0.0001,
                                    value.x * 0.0001,
                                    value.z * 0.001,
                                    value.yaw * 1])
                current_obs_dict[self.unit_indices[key]] = obs_array
            
        blue_obs_dict = self.sim.get_obs(side='blue')
        for key,value in blue_obs_dict.my_planes.items():
            if key in self.unit_indices:
                obs_array = np.array([value.x * 0.0001,
                                    value.x * 0.0001,
                                    value.z * 0.001,
                                    value.yaw * 1])
                current_obs_dict[self.unit_indices[key]] = obs_array

        return current_obs_dict
    
if __name__ == "__main__":
    env = HDEnv("defaults")  # 类似open_spiel的游戏内核
    obs = env.reset()
    done = {"__all__": False}
    step = 0
    while not done["__all__"]:
        # if step % 20 ==0:
        #     obs = env.reset()
        actions = {agent: env.action_space.sample() for agent in range(env.num_agents)}  # {0: 3, 1: 1}  -> 敌我12个智能体
        obs, rewards, terminateds, truncateds, info = env.step(actions)  
        print(f"Step: {step}")
        print("Observations:", obs) # {0: np.array[120], 1: np.array[120]}  -> 敌我12个智能体
        print("Rewards:", rewards)  # {0: 1.0, 1: 1.0}  -> 敌我12个智能体
        done = terminateds  # {0: False, 1: fasle, __all__: false}, no info  -> 敌我12个智能体
        step += 1
        # env.render(mode='human')
    print("Game over.")