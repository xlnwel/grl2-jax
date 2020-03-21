"""
https://github.com/openai/baselines/blob/master/baselines/common/wrappers.py
"""
import numpy as np
import gym
from utility.utils import infer_dtype

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ActionRepetition(gym.Wrapper):
    def __init__(self, env, n_ar=1):
        print(f'Action repetition({n_ar})')
        super().__init__(env)
        self.n_ar = n_ar

    def step(self, action, n_ar=None, gamma=1):
        rewards = 0
        n_ar = n_ar or self.n_ar
        for i in range(1, n_ar+1):
            state, reward, done, info = self.env.step(action)

            rewards += gamma**(i-1) * reward
            if done:
                break
        info['n_ar'] = i
        
        return state, rewards, done, info

class EnvStats(gym.Wrapper):
    """ Provide Environment Statistics Recording """
    def __init__(self, env, precision=32):
        super().__init__(env)
        self.already_done = True
        self.precision = precision

    def reset(self):
        if getattr(self, 'was_real_done', True):
            self.score = 0
            self.epslen = 0
            self.already_done = False
            self.mask = 1

        return self.env.reset()

    def step(self, action):
        if self.already_done:
            self.mask = 0
            return np.zeros(self.obs_shape), 0, True, {}
        else:
            self.mask = 1 - self.already_done
            state, reward, done, info = self.env.step(action)
            self.score += 0 if self.already_done else reward
            self.epslen += 0 if self.already_done else 1
            self.already_done = done and getattr(self, 'was_real_done', True)
            # ignore done signal if the time limit is reached
            if self.epslen == self.env.spec.max_episode_steps:
                done = False

            return state, reward, done, info

    def get_mask(self):
        """ Get mask at the current step. Should only be called after self.step """
        return bool(self.mask)

    def get_score(self):
        return self.score

    def get_epslen(self):
        return self.epslen

    @property
    def is_action_discrete(self):
        return isinstance(self.env.action_space, gym.spaces.Discrete)

    @property
    def obs_shape(self):
        return self.observation_space.shape

    @property
    def obs_dtype(self):
        return infer_dtype(self.observation_space.dtype, self.precision)

    @property
    def action_shape(self):
        return self.action_space.shape

    @property
    def action_dtype(self):
        return infer_dtype(self.action_space.dtype, self.precision)

    @property
    def action_dim(self):
        return self.action_space.n if self.is_action_discrete else self.action_shape[0]


""" The following wrappers rely on already_done defined in EnvStats.
Therefore, they should only be called after EnvStats """
class LogEpisode(gym.Wrapper):
    def reset(self):
        obs = self.env.reset()
        transition = dict(
            obs=obs,
            action=np.zeros(self.env.action_space.shape),
            reward=0.,
            done=False
        )
        self._episode = [transition]
        return obs
    
    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action)
        transition = dict(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            **kwargs
        )
        self._episode.append(transition)
        if self.already_done:
            episode = {k: np.array([t[k] for t in self._episode])
                for k in self._episode[0]}
            info['episode'] = episode
        return obs, reward, done, info

class AutoReset(gym.Wrapper):
    def step(self, action, **kwargs):
        if self.already_done:
            state = self.env.reset()
            reward = 0.
            done = False
            info = {}
        else:
            state, reward, done, info = self.env.step(action, **kwargs)
        
        return state, reward, done, info

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper) and hasattr(currentenv, 'env'):
            currentenv = currentenv.env
        else:
            # don't raise error here, only return None
            return None
            