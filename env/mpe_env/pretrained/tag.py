import gym
from gym.spaces import Tuple
from .ddpg import DDPG
import torch
import os
import numpy as np

class FrozenTag(gym.Wrapper):
  """ Tag with pretrained prey agent """
  
  def __init__(self, env) -> None:
    super().__init__(env)
    # NOTE: assume the number of prey is one
    self.action_space = self.action_space[:-1]
    self.obs_shape = self.obs_shape[:-1]
    self.obs_dtype = self.obs_dtype[:-1]
    self.action_shape = self.action_shape[:-1]
    self.action_dim = self.action_dim[:-1]
    self.action_dtype = self.action_dtype[:-1]
    self.n_agents = self.n_agents - 1
    self.n_units = self.n_units - 1

  def reset(self, *args, **kwargs):
    obs = super().reset(*args, **kwargs, raw_env=True)
    obs['obs'] = np.stack(obs['obs'][:-1])
    obs['global_state'] = np.stack(obs['global_state'][:-1])
    return obs

  def step(self, action):
    # random_action = self.pt_action_space.sample()
    random_action = 0
    action = tuple(action) + (random_action,)
    obs, rew, done, info = super().step(action)
    obs['obs'] = np.stack(obs['obs'][:-1])
    obs['global_state'] = np.stack(obs['global_state'][:-1])
    rew = np.stack(rew[:-1])
    done = np.stack(done[:-1])
    return obs, rew, done, info

class RandomTag(gym.Wrapper):
  """ Tag with pretrained prey agent """

  def __init__(self, env) -> None:
    super().__init__(env)
    # NOTE: assume the number of prey is one
    self.action_space = self.action_space[:-1]
    self.obs_shape = self.obs_shape[:-1]
    self.obs_dtype = self.obs_dtype[:-1]
    self.action_shape = self.action_shape[:-1]
    self.action_dim = self.action_dim[:-1]
    self.action_dtype = self.action_dtype[:-1]
    self.n_agents = self.n_agents - 1
    self.n_units = self.n_units - 1

  def reset(self, *args, **kwargs):
    obs = super().reset(*args, **kwargs, raw_env=True)
    obs['obs'] = np.stack(obs['obs'][:-1])
    obs['global_state'] = np.stack(obs['global_state'][:-1])
    return obs

  def step(self, action):
    random_action = self.pt_action_space.sample()
    action = tuple(action) + (random_action,)
    obs, rew, done, info = super().step(action)
    obs['obs'] = np.stack(obs['obs'][:-1])
    obs['global_state'] = np.stack(obs['global_state'][:-1])
    rew = np.stack(rew[:-1])
    done = np.stack(done[:-1])
    return obs, rew, done, info


class PretrainedTag(gym.Wrapper):
  """ Tag with pretrained prey agent """

  def __init__(self, env) -> None:
    super().__init__(env)
    # NOTE: assume the number of prey is one
    self.action_space = self.action_space[:-1]
    self.obs_shape = self.obs_shape[:-1]
    self.obs_dtype = self.obs_dtype[:-1]
    self.action_shape = self.action_shape[:-1]
    self.action_dim = self.action_dim[:-1]
    self.action_dtype = self.action_dtype[:-1]
    self.n_agents = self.n_agents - 1
    self.n_units = self.n_units - 1

    self.prey = DDPG(14, 5, 50, 128, 0.01)
    param_path = os.path.join(os.path.dirname(__file__), 'prey_params.pt')
    save_dict = torch.load(param_path)
    self.prey.load_params(save_dict['agent_params'][-1])
    self.prey.policy.eval()
    self.last_prey_obs = None

  def reset(self, *args, **kwargs):
    obs = super().reset(*args, **kwargs, raw_env=True)
    self.last_prey_obs = obs['obs'][-1]
    obs['obs'] = np.stack(obs['obs'][:-1])
    obs['global_state'] = np.stack(obs['global_state'][:-1])
    return obs

  def step(self, action):
    prey_action = self.prey.step(self.last_prey_obs)
    action = tuple(action) + (prey_action,)
    obs, rew, done, info = super().step(action)
    self.last_prey_obs = obs['obs'][-1]
    obs['obs'] = np.stack(obs['obs'][:-1])
    obs['global_state'] = np.stack(obs['global_state'][:-1])
    rew = np.stack(rew[:-1])
    done = np.stack(done[:-1])
    return obs, rew, done, info