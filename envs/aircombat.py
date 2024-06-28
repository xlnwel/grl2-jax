import numpy as np
f32 = np.float32

from envs.aircombat.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.utils import *


class AirCombat:
  def __init__(self):
    self.env = SingleCombatEnv('1v1/DodgeMissile/SelfPlay')

    self.uid2aid = (0, 1)
    self.aid2uids = compute_aid2uids(self.uid2aid)
    self.n_agents = len(self.aid2uids)
    self.n_units = len(self.uid2aid)

    self.action_space = [self.env.action_space for _ in range(self.n_agents)]
    self.action_shape = [() for _ in self.action_space]
    self.action_dim = [19 for a in self.env.action_space.nvec]
    self.action_dtype = [np.int32 for _ in self.action_space]
    self.is_action_discrete = [True for _ in self.action_space]

    self.observation_space = self.env.observation_space
    obs = self.reset()
    self.obs_shape = [{k: v.shape[-1:] for k, v in o.items()} for o in obs]
    self.obs_dtype = [{k: v.dtype for k, v in o.items()} for o in obs]

    self._score = np.zeros(self.n_units, dtype=f32)
    # The accumulated episodic rewards we give to the agent. It includes shaped rewards
    self._dense_score = np.zeros(self.n_units, dtype=f32)
    # The length of the episode
    self._epslen = 0
    self._left_score = np.zeros(self.n_units, dtype=f32)
    self._right_score = np.zeros(self.n_units, dtype=f32)

  def random_action(self):
    action = np.array([ac.sample() for ac in self.action_space])
    return action

  def reset(self):
    obs = self.env.reset()
    return obs

  def step(self, action):
    obs, reward, done, info = self.env.step(action)

    return obs, reward, done, info
  

if __name__ == "__main__":
  env = AirCombat()
  obs = env.reset()
  print(obs)

  for i in range(0):
    obs, reward, done, info = env.step(env.random_action())
    print(obs, reward, done, info, sep='\n')
