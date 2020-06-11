import numpy as np
import gym

from utility.utils import infer_dtype, convert_dtype
import cv2
# stop using GPU
cv2.ocl.setUseOpenCL(False)


class Dummy:
    """ Useful to break the inheritance of unexpected attributes """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.spec = env.spec

        self.reset = env.reset
        self.step = env.step
        self.render = env.render
        self.close = env.close


class Wrapper:
    def __getattr__(self, name):
        return getattr(self.env, name)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

class NormalizeActions(gym.Wrapper):
    """ Normalize infinite action dimension in range [-1, 1] """
    def __init__(self, env):
        super().__init__(env)
        self._act_mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._act_mask, env.action_space.low, -1)
        self._high = np.where(self._act_mask, env.action_space.high, 1)

    @property
    def action_space(self):
        low = np.where(self._act_mask, -np.ones_like(self._low), self._low)
        high = np.where(self._act_mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._act_mask, original, action)
        return self.env.step(original)

class GrayScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        original_space = self.observation_space
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(*original_space.shape[:2], 1),
            dtype=np.uint8,
        )
        assert original_space.dtype == np.uint8, original_space.dtype
        assert len(original_space.shape) == 3, original_space.shape
        self.observation_space = new_space
    
    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = np.expand_dims(obs, -1)

        return obs

class FrameSkip(gym.Wrapper):
    """ Unlike MaxAndSkipEnv defined in baselines
    this wrapper does not max pool observations.
    This is useful for RGB observations
    """
    def __init__(self, env, frame_skip=1):
        super().__init__(env)
        self.frame_skip = frame_skip

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


class AutoReset(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self._done = True

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    if self._done:
      obs, reward, done, info = self.env.reset(), 0.0, False, {}
    else:
      obs, reward, done, info = self.env.step(action)
    self._done = done
    return obs, reward, done, info

  def reset(self):
    self._done = False
    return self._env.reset()


class EnvStats(gym.Wrapper):
    """ Records environment statistics """
    def __init__(self, env, max_episode_steps=None, precision=32, timeout_done=False):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps or int(1e9)
        # already_done indicate whether an episode is finished, 
        # either due to timeout or due to environment done
        self._already_done = True
        self._precision = precision
        # if we take timeout as done
        self._timeout_done = timeout_done
        self._fake_obs = np.zeros(self.obs_shape, dtype=self.obs_dtype)
        self._score = 0
        self._epslen = 0
        self._info = {}
        if timeout_done:
            print('Timeout is treated as done')

    def reset(self, **kwargs):
        if self.already_done() == self.game_over():
            if not self.game_over():
                print("Warning: reset env before it's done")
            self._score = 0
            self._epslen = 0
        self._already_done = False
        self._mask = 1
        return self.env.reset(**kwargs)

    def step(self, action, **kwargs):
        if self.game_over():
            # as some environment, e.g. Ant-v3 implicitly reset env
            # when keeping stepping after game's over
            # here, we override this behavior
            self._mask = 0
            self._info['mask'] = self._mask
            return self._fake_obs, 0, True, self._info
        assert not np.any(np.isnan(action)), action
        obs, reward, done, info = self.env.step(action, **kwargs)
        self._score += reward
        self._epslen += info.get('frame_skip', 1)
        self._already_done = done
        if self._epslen >= self.max_episode_steps:
            self._already_done = True
            if hasattr(self.env, '_game_over'):
                self.env.set_game_over()    # define set_game_over for env
            done = self._timeout_done
            info['timeout'] = True
        if self._already_done:
            info['already_done'] = self._already_done
        if self.game_over():
            info['score'] = self._score
            info['epslen'] = self._epslen
            info['game_over'] = True
        self._info = info
        return np.array(obs, dtype=self.obs_dtype), \
                np.float32(reward), np.float32(done), info

    def mask(self):
        """ Get mask at the current step. """
        return self._mask

    def score(self, **kwargs):
        return self._score

    def epslen(self, **kwargs):
        return self._epslen

    def already_done(self):
        return self._already_done

    def info(self):
        return self._info
        
    def game_over(self):
        if hasattr(self.env, 'game_over'):
            return self.env.game_over()
        else:
            return self._already_done

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
class LogEpisode(gym.Wrapper):
    """ Log episodic information, useful when we need to record the reset state """
    def __init__(self, env):
        super().__init__(env)
        self.prev_episode = {}

    def reset(self, **kwargs):
        obs = self.env.reset()
        transition = dict(
            obs=obs,
            prev_action=np.zeros(self.env.action_space.shape, np.float32),
            reward=0.,
            discount=1, 
            **kwargs
        )
        self._episode = [transition]
        return obs
    
    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action)
        reward = convert_dtype(reward, self._precision)
        transition = dict(
            obs=obs,
            prev_action=action,
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


class RewardHack(gym.Wrapper):
    def __init__(self, env, reward_scale=1, reward_clip=None, **kwargs):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward *= self.reward_scale
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        return obs, reward, done, info


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname == currentenv.__class__.__name__:
            return currentenv
        elif hasattr(currentenv, 'env'):
            currentenv = currentenv.env
        else:
            # don't raise error here, only return None
            return None
            