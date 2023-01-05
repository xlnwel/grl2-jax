import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import numpy as np

from env.ma_mujoco_env.multiagent_mujoco.mujoco_multi import MujocoMulti

class MAMujoco(gym.Wrapper):
    def __init__(self, config):
        scenario, agent_conf = config.env_name.split('_')
        config.env_args.scenario = f'{scenario}-v2'
        config.env_args.agent_conf = agent_conf
        config.env_args.episode_limit = config.max_episode_steps

        self.env = MujocoMulti(**config)

        self.observation_space = [Box(low=np.array([-10]*self.n_agents), high=np.array([10]*self.n_agents)) for _ in range(self.n_agents)]

        self.obs_shape = [{
            'obs': (self.env.obs_size, ), 
            'global_state': (self.env.obs_size, )
        } for _ in range(self.n_agents)]
        self.obs_dtype = [{
            'obs': np.float32, 
            'global_state': np.float32
        } for _ in range(self.n_agents)]
        self.action_space = self.env.action_space

        self.n_agents = self.env.n_agents
        self.uid2aid = list(range(self.n_agents))
        self.n_units = self.n_agents
        
        self.reward_range = None
        self.metadata = None
        self.max_episode_steps = self.env.episode_limit
        self._score = np.zeros(self.n_agents)
        self._dense_score = np.zeros(self.n_agents)
        self._epslen = 0

    def random_action(self):
        action = [a.sample() for a in self.action_space]
        return action

    def step(self, actions):
        actions = np.reshape(actions, (self.n_agents, -1))
        obs, state, reward, done, _, _ = self.env.step(actions)
        reward = np.reshape(reward, -1)
        done = done[0]
        obs = get_obs(obs, state)

        self._score += reward
        self._dense_score += reward
        self._epslen += 1

        info = {
            'score': self._score, 
            'dense_score': self._dense_score, 
            'epslen': self._epslen, 
            'game_over': self._epslen == self.max_episode_steps
        }

        reward = np.split(reward, self.n_agents)
        if done and self._epslen == self.max_episode_steps:
            done = [np.zeros(1) for _ in range(self.n)]
        else:
            done = [np.ones(1) * done for _ in range(self.n)]
        return obs, reward, done, info

    def reset(self):
        obs, state, _ = self.env.reset()
        obs = get_obs(obs, state)

        self._score = np.zeros(self.n_agents)
        self._dense_score = np.zeros(self.n_agents)
        self._epslen = 0

        return obs

def get_obs(obs, state):
    agent_obs = []
    for o, s in zip(obs, state):
        o = np.expand_dims(o, 0)
        s = np.expand_dims(s, 0)
        agent_obs.append({'obs': o, 'global_state': s})
    return agent_obs