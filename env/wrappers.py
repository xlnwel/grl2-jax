"""
https://github.com/openai/baselines/blob/master/baselines/common/wrappers.py
"""
import numpy as np
import gym

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

class EnvStats(gym.Wrapper):
    """ Provide Environment Statistics Recording """
    def __init__(self, env):
        super().__init__(env)
        self.already_done = True

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
            return np.zeros(self.state_shape), 0, True, {}
        else:
            self.mask = 1 - self.already_done
            state, reward, done, info = self.env.step(action)
            self.score += 0 if self.already_done else reward
            self.epslen += 0 if self.already_done else 1
            self.already_done = done and getattr(self, 'was_real_done', True)

            return state, reward, done, info

    def get_mask(self):
        """ Get mask at the current step. Should only be called after self.step """
        return self.mask

    def get_score(self):
        return self.score

    def get_epslen(self):
        return self.epslen

    @property
    def is_action_discrete(self):
        return isinstance(self.env.action_space, gym.spaces.Discrete)

    @property
    def state_shape(self):
        return self.observation_space.shape

    @property
    def state_dtype(self):
        return np.float32 if self.observation_space.dtype == np.float64 else self.observation_space.dtype

    @property
    def action_shape(self):
        return self.action_space.shape

    @property
    def action_dtype(self):
        return np.int32 if self.is_action_discrete else np.float32

    @property
    def action_dim(self):
        return self.action_space.n if self.is_action_discrete else self.action_shape[0]


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


class AutoReset(gym.Wrapper):
    def step(self, action, **kwargs):
        state, reward, done, info = self.env.step(action, **kwargs)
        if done:
            state = self.env.reset()
        
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
            