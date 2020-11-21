import collections
import logging
import numpy as np
import gym

from utility.utils import infer_dtype, convert_dtype
import cv2
# stop using GPU
cv2.ocl.setUseOpenCL(False)
logger = logging.getLogger(__name__)

# for multi-processing efficiency, we do not return info at every step
EnvOutput = collections.namedtuple('EnvOutput', 'obs reward discount reset')
# Output format of gym
GymOutput = collections.namedtuple('GymOutput', 'obs reward discount')


def post_wrap(env, config):
    """ Does some post processing and bookkeeping. 
    Does not change anything that will affect the agent's performance 
    """
    env = DataProcess(env, config.get('precision', 32))
    env = EnvStats(
        env, config.get('max_episode_steps', None), 
        timeout_done=config.get('timeout_done', False),
        auto_reset=config.get('auto_reset', True),
        log_episode=config.get('log_episode', False))
    return env

class Dummy:
    """ Useful to break the inheritance of unexpected attributes """
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.spec = env.spec
        self.reward_range = env.reward_range
        self.metadata = env.metadata

        self.reset = env.reset
        self.step = env.step
        self.render = env.render
        self.close = env.close
        self.seed = env.seed


""" Wrappers from OpenAI's baselines. 
Some modifications are done to meet specific requirements """
class LazyFrames:
    def __init__(self, frames):
        """ Different from the official implementation from baselines
        we do not cache the results to save memory.
        """
        self._frames = frames

    def _force(self):
        return np.concatenate(self._frames, axis=-1)

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, frame_skip=4):
        """Return only every `frame_skip`-th frame"""
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self.frame_skip  = frame_skip

    def step(self, action, frame_skip=None, **kwargs):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        frame_skip = frame_skip or self.frame_skip
        for i in range(frame_skip):
            obs, reward, done, info = self.env.step(action, **kwargs)
            if i == frame_skip - 2: self._obs_buffer[0] = obs
            if i == frame_skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        info['frame_skip'] = i+1

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, np_obs):
        super().__init__(env)
        self.k = k
        self.np_obs = np_obs
        self.frames = collections.deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action, **kwargs):
        ob, reward, done, info = self.env.step(action, **kwargs)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=-1) \
            if self.np_obs else LazyFrames(list(self.frames))


""" Custom wrappers """
class NormalizeActions(gym.Wrapper):
    """ Normalize infinite action dimension in range [-1, 1] """
    def __init__(self, env):
        super().__init__(env)
        self._act_mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._act_mask, env.action_space.low, -1)
        self._high = np.where(self._act_mask, env.action_space.high, 1)

        low = np.where(self._act_mask, -np.ones_like(self._low), self._low)
        high = np.where(self._act_mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action, **kwargs):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._act_mask, original, action)
        return self.env.step(original, **kwargs)

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

    def step(self, action, frame_skip=None, **kwargs):
        total_reward = 0
        frame_skip = frame_skip or self.frame_skip
        for i in range(1, frame_skip+1):
            obs, reward, done, info = self.env.step(action, **kwargs)
            total_reward += reward
            if done:
                break
        info['frame_skip'] = i
        
        return obs, total_reward, done, info


class FrameDiff(gym.Wrapper):
    def __init__(self, env, n):
        super().__init__(env)

        original_space = self.observation_space
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(*original_space.shape[:2], original_space.shape[-1]*2),
            dtype=np.uint8,
        )
        self.observation_space = new_space

    def reset(self):
        obs = self.env.reset()
        new_obs = np.concatenate([np.zeros_like(obs, dtype=obs.dtype), obs], axis=-1)
        self.prev_obs = obs

        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        new_obs = np.concatenate([obs - self.prev_obs, obs], axis=-1)
        self.prev_obs = obs

        return new_obs, rew, done, info


class DataProcess(gym.Wrapper):
    def __init__(self, env, precision=32):
        super().__init__(env)
        self.precision = precision

        self.is_action_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.obs_shape = self.observation_space.shape
        self.action_shape = self.action_space.shape
        self.action_dim = self.action_space.n if self.is_action_discrete else self.action_shape[0]

        self.obs_dtype = infer_dtype(self.observation_space.dtype, precision)
        self.action_dtype = np.int32 if self.is_action_discrete \
            else infer_dtype(self.action_space.dtype, self.precision)
        self.float_dtype = infer_dtype(np.float32, self.precision)

    def observation(self, observation):
        if isinstance(observation, np.ndarray):
            return convert_dtype(observation, self.precision)
        return observation
    
    def action(self, action):
        if isinstance(action, np.ndarray):
            return convert_dtype(action, self.precision)
        return np.int32(action) # always keep int32 for integers as tf.one_hot does not support int16

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        return self.observation(obs), self.float_dtype(reward), self.float_dtype(done), info


""" 
Distinctions several signals:
    done: an episode is done, may due to life loss
    game over: a game is over, may due to timeout. Life loss is not game over
    reset: an new episode starts after done. In auto-reset mode, environment 
        resets when game's over. Life loss should be automatically handled by
        the environment/previous wrapper.
"""
class EnvStats(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None, timeout_done=False, 
        auto_reset=True, log_episode=False, initial_state={}):
        """ Records environment statistics """
        super().__init__(env)
        self.max_episode_steps = max_episode_steps or int(1e9)
        # if we take timeout as done
        self.timeout_done = timeout_done
        self.auto_reset = auto_reset
        self.log_episode = log_episode
        # game_over indicates whether an episode is finished, 
        # either due to timeout or due to environment done
        self._game_over = True
        self._score = 0
        self._epslen = 0
        self._info = {}
        self._init_state = initial_state
        self._output = tuple()
        self.float_dtype = getattr(self.env, 'float_dtype', np.float32)
        if timeout_done:
            logger.info('Timeout is treated as done')
        self._reset()

    def reset(self):
        if self.auto_reset:
            self.auto_reset = False
            logger.info('Explicitly resetting turns off auto-reset. Maker sure this is done intentionally at evaluation')
        if not self._output.reset:
            return self._reset()
        else:
            return self._output

    def _reset(self):
        obs = self.env.reset()
        self._score = 0
        self._epslen = 0
        self._game_over = False
        reward = self.float_dtype(0)
        discount = self.float_dtype(1)
        reset = self.float_dtype(True)
        self._output = EnvOutput(obs, reward, discount, reset)

        if self.log_episode:
            transition = dict(
                obs=obs,
                prev_action=np.zeros(self.action_shape, self.action_dtype),
                reward=reward,
                discount=discount, 
                **self._init_state
            )
            self._episode = [transition]
        return self._output

    def step(self, action, **kwargs):
        if self.game_over():
            assert self.auto_reset == False
            reward = self.float_dtype(0)
            discount = self.float_dtype(0)
            reset = self.float_dtype(0)
            self._output = EnvOutput(self._output.obs, reward, discount, reset)
            self._info['mask'] = False
            return self._output

        assert not np.any(np.isnan(action)), action
        obs, reward, done, info = self.env.step(action, **kwargs)
        self._score += reward
        self._epslen += info.get('frame_skip', 1)
        self._game_over = info.get('game_over', done)
        if self._epslen >= self.max_episode_steps:
            self._game_over = True
            done = self.timeout_done
            info['timeout'] = True
        reward = self.float_dtype(reward)
        discount = self.float_dtype(1-done)
        # we expect auto-reset environments, which artificially reset due to life loss,
        # return reset in info when resetting
        reset = self.float_dtype(info.get('reset', False))
        
        # log transition
        if self.log_episode:
            transition = dict(
                obs=obs,
                prev_action=action,
                reward=reward,
                discount=discount,
                **kwargs
            )
            self._episode.append(transition)
            if self._game_over:
                episode = {k: np.array([t[k] for t in self._episode])
                    for k in self._episode[0]}
                self._prev_episode = episode

        # reset env
        if self._game_over:
            info['game_over'] = self._game_over
            info['score'] = self._score
            info['epslen'] = self._epslen
            if self.auto_reset:
                info['prev_env_output'] = GymOutput(obs, reward, discount)
                # when resetting, we override the obs and reset but keep the others
                obs, _, _, reset = self._reset()
        self._info = info

        self._output = EnvOutput(obs, reward, discount, reset)
        return self._output

    def score(self, **kwargs):
        return self._info.get('score', self._score)

    def epslen(self, **kwargs):
        return self._info.get('epslen', self._epslen)

    def mask(self, **kwargs):
        return self._info.get('mask', True)

    def game_over(self):
        return self._game_over

    def prev_obs(self):
        return self._info['prev_env_output'].obs

    def prev_episode(self):
        eps = self._prev_episode
        self._prev_episode = None
        return eps

    def info(self):
        return self._info
        
    def output(self):
        return self._output


class RewardHack(gym.Wrapper):
    """ This wrapper should be invoked after EnvStats to avoid inaccurate bookkeeping """
    def __init__(self, env, reward_scale=1, reward_clip=None, **kwargs):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip

    def step(self, action, **kwargs):
        output = self.env.step(action, **kwargs)
        obs, reward, discount, reset = output
        reward *= self.reward_scale
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        reward = self.float_dtype(reward)
        return EnvOutput(obs, reward, discount, reset)


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
            
if __name__ == '__main__':
    from env.func import create_env
    env = create_env(dict(
        name='LunarLander-v2',
        seed=0
    ))
    n_act = env.action_dim
    for i in range(500):
        out = env.step(i % n_act)
        print(i, out)
        print(env.score(), env.epslen())
        if out.reset:
            info = env.info()
            print(info['score'], info['epslen'])
            break