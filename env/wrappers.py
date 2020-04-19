"""
https://github.com/openai/baselines/blob/master/baselines/common/wrappers.py
"""
import numpy as np
import gym


from utility.utils import infer_dtype, convert_dtype


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

class FrameSkip:
    def __init__(self, env, frame_skip=1):
        self.env = env
        self.frame_skip = frame_skip

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, frame_skip=None):
        total_reward = 0
        frame_skip = frame_skip or self.frame_skip
        for i in range(1, frame_skip+1):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        info['frame_skip'] = i
        
        return obs, total_reward, done, info

class EnvStats:
    """ Provide Environment Statistics Recording """
    def __init__(self, env, max_episode_steps, precision=32, timeout_done=False):
        self.env = env
        self.max_episode_steps = max_episode_steps
        # already_done indicate whether an episode is finished, 
        # either due to timeout or due to environment done
        self._already_done = True
        self._precision = precision
        # if we take timeout as done
        self._timeout_done = timeout_done

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        if self.game_over():
            self._score = 0
            self._epslen = 0
        self._already_done = False
        self._mask = 1
        return self.env.reset(**kwargs)

    def step(self, action):
        self._mask = 1 - self._already_done
        obs, reward, done, info = self.env.step(action)
        self._score += 0 if self._already_done else reward
        self._epslen += 0 if self._already_done else info.get('frame_skip', 1)
        self._already_done = done
        if self._epslen >= self.max_episode_steps:
            self._already_done = True
            if self._timeout_done:
                done = True

        return obs, reward, done, info

    def mask(self):
        """ Get mask at the current step. Should only be called after self.step """
        return bool(self._mask)

    def score(self, **kwargs):
        return self._score

    def epslen(self, **kwargs):
        return self._epslen

    def already_done(self):
        return self._already_done

    def game_over(self):
        return getattr(self, 'was_real_done', self._already_done)

    @property
    def is_action_discrete(self):
        return isinstance(self.env.action_space, gym.spaces.Discrete)

    @property
    def obs_shape(self):
        return self.observation_space.shape

    @property
    def obs_dtype(self):
        """ this is not the observation's real dtype, but the desired dtype """
        return infer_dtype(self.observation_space.dtype, self._precision)

    @property
    def action_shape(self):
        return self.action_space.shape

    @property
    def action_dtype(self):
        """ this is not the action's real dtype, but the desired dtype """
        return infer_dtype(self.action_space.dtype, self._precision)

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
        reward = convert_dtype(reward, self._precision)
        transition = dict(
            obs=obs,
            action=action,
            reward=reward,
            discount=1-done,
            **kwargs
        )
        self._episode.append(transition)
        if self.game_over():
            episode = {k: convert_dtype([t[k] for t in self._episode], self._precision)
                for k in self._episode[0]}
            info['episode'] = self.prev_episode = episode
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
            