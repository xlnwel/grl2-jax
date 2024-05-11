import numpy as np
import gym

from env.utils import *


""" 我们定义下面3种ID

1. Unit ID (0-n_units), 对应着环境中的一个可控单位
2. Group ID (0-n_groups), 对应着Group的唯一标识, 一个Group可以包含多个Unit
3. Agent ID (0-n_agents), 对应着Agent的唯一标识, 一个Agent可以控制多个Group

我们用list来做相关映射, list的index对应着源值域, 值对应着目标值域. 
如uid2aid=[0, 0, 1], 表示Unit 0, 1受Agent 0控制, Unit 2受Agent 1控制
"""
class TemplateEnv:
  def __init__(self, uid2aid=[0, 1], uid2gid=[0, 1], max_episode_steps=100, level=1, **kwargs):
    self.uid2aid = uid2aid    # Unit ID到Aagent ID的映射
    self.uid2gid = uid2gid    # Unit ID到Group ID的映射
    self.aid2uids = compute_aid2uids(self.uid2aid)      # Agent ID到Unit ID的映射
    self.gid2uids = compute_aid2uids(self.uid2gid)      # Group ID到Unit ID的映射
    self.aid2gids = compute_aid2gids(uid2aid, uid2gid)  # Agent ID到Group ID的映射
    self.n_units = len(self.uid2aid)    # Unit个数
    self.n_agents = len(self.aid2uids)  # Agent个数
    self.level = level

    # 观测/动作空间相关定义
    self.observation_space = [{
      'obs': gym.spaces.Box(high=float('inf'), low=0, shape=(6,)),
      'global_state': gym.spaces.Box(high=float('inf'), low=0, shape=(6,))
    } for _ in range(self.n_agents)]
    self.obs_shape = [{
      k: v.shape for k, v in obs.items()
    } for obs in self.observation_space]
    self.obs_dtype = [{
      k: v.dtype for k, v in obs.items()
    } for obs in self.observation_space]
    self.action_space = [{
      'action_discrete': gym.spaces.Discrete(4), 
      'action_continuous': gym.spaces.Box(low=-1, high=1, shape=(2,))
    } for _ in range(self.n_agents)]
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

    self._epslen = 0    # 回合步长
    self._score = np.zeros(self.n_units)  # 最终胜负
    self._dense_score = np.zeros(self.n_units)  # 稠密奖励累积

    self.max_episode_steps = max_episode_steps  # 最大回合步长

  def seed(self, seed=None):
    pass

  def random_action(self):
    """ 返回随机动作

    Returns:
        List[Dict]: 动作列表, 每个元素为一个字典代表一个智能体输出的动作
    """
    action = [{
      k: np.array([v.sample() for _ in self.aid2uids[i]]) for k, v in a.items()} 
      for i, a in enumerate(self.action_space)]
    return action

  def reset(self):
    """重置环境

    Returns:
      Dict: 环境重置后的观测
    """
    self._epslen = 0
    self._score = np.zeros(self.n_units)
    self._dense_score = np.zeros(self.n_units)
    env_out = self._reset_env()
    obs = self._get_obs(env_out)

    return obs
  
  def step(self, action):
    """将动作传给环境, 并从环境中取回观测, 奖励, 终止信号, 其他信息

    Args:
        action (List): 动作列表. Defaults to None.

    Returns:
        obs (List[Dict]): 每个智能体的观测
        reward (np.ndarray): 奖励
        done (np.ndarray): 终止信号
        info (Dict): 其他信息
    """
    # 下面函数仅作为一个示例
    env_out = self._env_step(action)
    obs = self._get_obs(env_out)
    reward = self._get_reward(env_out)
    done = self._get_done(env_out)


    self._dense_score += reward
    self._score = np.sign(self._dense_score)

    info = {
      'score': self._score,
      'dense_score': self._dense_score, 
      'epslen': self._epslen, 
      'game_over': np.any(done)
    }

    return obs, reward, done, info
  
  """ 下面私有函数仅为示例需要, 可自由定制需要函数 """
  def _reset_env(self):
    """ 重置环境 """
    fake_out = {}
    return fake_out
  
  def _env_step(self, action):
    """ 将动作传给仿真, 并返回仿真数据

    Args:
        action (Any): 智能体动作
    """
    fake_out = {}
    return fake_out

  def _get_obs(self, env_out):
    """ 获取观测 """
    return [{k: np.zeros((len(self.aid2uids[aid]), *v)) for k, v in o.items()} for aid, o in enumerate(self.obs_shape)]

  def _get_reward(self, env_out):
    """ 获取奖励 """
    return np.random.uniform(-1, 1, self.n_units)

  def _get_done(self, env_out):
    """ 获取终止信号 """
    return np.ones(self.n_units) * np.random.random() < 0.1
  
  def close(self):
    """ 关闭环境 """
    pass