from socket import SCM_RIGHTS
from .lbf_env import ForagingEnv
from gym import spaces
import gym
import numpy as np


class LBFEnv(gym.Wrapper):
  def __init__(self, config):
    self.env = ForagingEnv(
      **config.env_args
    )
    
    self.sight = config.env_args.sight
    self._seed = config.seed
    self.env.seed(self._seed)
    self._obs = None
    self._state = None

    self.n_agents = self.env.n_agents
    self.uid2aid = list(range(self.n_agents))
    self.n_units = self.n_agents

    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space
    self.action_shape = [a.shape for a in self.action_space]
    self.action_dim = [a.n for a in self.action_space]
    self.action_dtype = [np.int32 for _ in self.action_space]
    self.is_action_discrete = [True for _ in range(self.n_agents)]

    self.obs_shape = [{
      'obs': (self.env.ob_dim, ),
      'global_state': (self.env.state_dim, ),
    } for _ in range(self.n_agents)]
    self.obs_dtype = [{
      'obs': np.float32,
      'global_state': np.float32,
    } for _ in range(self.n_agents)]
    
    self.max_episode_steps = config.max_episode_steps
    self._score = np.zeros(self.n_agents)
    self._dense_score = np.zeros(self.n_agents)
    self._epslen = 0

  def observation(self, observation):
    return tuple(
      [
        spaces.flatten(obs_space, obs)
        for obs_space, obs in zip(self.env.observation_space, observation)
      ]
    )

  def step(self, actions):
    """ Returns reward, terminated, info """
    actions = [int(a) for a in actions]
    self._obs, reward, done, info, self._state = self.env.step(actions)
    self._obs = self.observation(self._obs)
    self._state = self.observation(self._state)
    
    self._score += float(sum(reward))
    self._dense_score += float(sum(reward))
    self._epslen += 1

    info.update(
      {
        'score': self._score,
        'dense_score': self._dense_score,
        'epslen': self._epslen,
        'game_over': all(done) or self._epslen == self.max_episode_steps
      }
    )

    reward = np.array([sum(reward) for _ in range(self.n_agents)])    
    done = np.array([all(done) for _ in range(self.n_agents)])
    
    obs = {
      'obs': np.stack(self._obs),
      'global_state': np.stack(self._state),
    }

    assert len(obs) == self.n_agents, (obs, self.n_agents)
    assert len(reward) == self.n_agents, (reward, self.n_agents)
    assert len(done) == self.n_agents, (done, self.n_agents)

    return obs, reward, done, info

  def reset(self):
    self._obs, self._state = self.env.reset()
    self._obs = self.observation(self._obs)
    self._state= self.observation(self._state)

    self._score = np.zeros(self.n_agents)
    self._dense_score = np.zeros(self.n_agents)
    self._epslen = 0

    obs = {
      'obs': np.stack(self._obs),
      'global_state': np.stack(self._state),
    }
    
    return obs