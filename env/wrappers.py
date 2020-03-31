"""
https://github.com/openai/baselines/blob/master/baselines/common/wrappers.py
"""
import numpy as np
import gym


from utility.utils import infer_dtype, convert_dtype


class EnvSpec:
    def __init__(self, max_episode_steps):
        self.max_episode_steps = max_episode_steps

class TimeLimit:
    def __init__(self, env, max_episode_steps=None):
        self.env = env
        self.spec = EnvSpec(max_episode_steps)
        self._elapsed_steps = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.spec.max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class NormalizeActions:
    """ Normalize infinite action dimension in range [-1, 1] """
    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)

class ClipActionsWrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ActionRepeat:
    def __init__(self, env, n_ar=1):
        self.env = env
        self.n_ar = n_ar

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, n_ar=None):
        total_reward = 0
        n_ar = n_ar or self.n_ar
        for i in range(1, n_ar+1):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        info['n_ar'] = i
        
        return obs, total_reward, done, info

class EnvStats:
    """ Provide Environment Statistics Recording """
    def __init__(self, env, precision=32):
        self.env = env
        self.already_done = True
        self.precision = precision

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        if getattr(self, 'was_real_done', True):
            self.score = 0
            self.epslen = 0
            self.already_done = False
            self.mask = 1

        return self.env.reset()

    def step(self, action):
        if self.already_done:
            self.mask = 0
            return np.zeros(self.obs_shape, dtype=np.float32), 0, True, {}
        else:
            self.mask = 1 - self.already_done
            obs, reward, done, info = self.env.step(action)
            self.score += 0 if self.already_done else reward
            self.epslen += 0 if self.already_done else info.get('n_ar', 1)
            self.already_done = done
            # ignore done signal if the time limit is reached
            if self.epslen == self.env.spec.max_episode_steps:
                done = False

            return obs, reward, done, info

    def get_mask(self):
        """ Get mask at the current step. Should only be called after self.step """
        return bool(self.mask)

    def get_score(self, **kwargs):
        return self.score

    def get_epslen(self, **kwargs):
        return self.epslen

    def get_already_done(self):
        return self.already_done

    @property
    def is_action_discrete(self):
        return isinstance(self.env.action_space, gym.spaces.Discrete)

    @property
    def obs_shape(self):
        return self.observation_space.shape

    @property
    def obs_dtype(self):
        """ this is not the observation's real dtype, but the desired dtype """
        return infer_dtype(self.observation_space.dtype, self.precision)

    @property
    def action_shape(self):
        return self.action_space.shape

    @property
    def action_dtype(self):
        """ this is not the action's real dtype, but the desired dtype """
        return infer_dtype(self.action_space.dtype, self.precision)

    @property
    def action_dim(self):
        return self.action_space.n if self.is_action_discrete else self.action_shape[0]


""" The following wrappers rely on members defined in EnvStats.
Therefore, they should only be invoked after EnvStats """
class LogEpisode:
    def __init__(self, env):
        self.env = env
        self.prev_episode = {}

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs = self.env.reset()
        transition = dict(
            obs=obs,
            action=np.zeros(self.env.action_space.shape, np.float32),
            reward=0.,
            discount=1
        )
        self._episode = [transition]
        return obs
    
    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action)
        transition = dict(
            obs=obs,
            action=action,
            reward=reward,
            discount=1-done,
            **kwargs
        )
        self._episode.append(transition)
        if self.already_done:
            episode = {k: convert_dtype([t[k] for t in self._episode], self.precision)
                for k in self._episode[0]}
            info['episode'] = self.prev_episode = episode
        return obs, convert_dtype(reward, self.precision), done, info

class AutoReset:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, **kwargs):
        if self.already_done:
            obs = self.env.reset()
        obs, reward, done, info = self.env.step(action, **kwargs)
        
        return obs, reward, done, info

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
            