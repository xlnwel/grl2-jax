import collections
import copy
import logging
import numpy as np
import gym
import cv2

from core.log import do_logging
from env.utils import compute_aid2uids
from utility.feature import one_hot
from utility.utils import dict2AttrDict, infer_dtype, convert_dtype
from utility.typing import AttrDict
from env.typing import EnvOutput, GymOutput

# stop using GPU
cv2.ocl.setUseOpenCL(False)
logger = logging.getLogger(__name__)


def post_wrap(env, config):
    """ Does some post processing and bookkeeping. 
    Does not change anything that will affect the agent's performance 
    """
    env = DataProcess(env, config.get('precision', 32))
    env = EnvStats(
        env, config.get('max_episode_steps', None), 
        timeout_done=config.get('timeout_done', False),
        auto_reset=config.get('auto_reset', True))
    return env


""" Wrappers from OpenAI's baselines. 
Some modifications are done to meet specific requirements """
class LazyFrames:
    def __init__(self, frames):
        """ Different from the official implementation from OpenAI's baselines,
        we do not cache the results to save memory. Also, notice we do not define
        functions like __getitem__ avoid unintended overhead introduced by
        not caching the results. This means we do not support something like the 
        following
        # error as __getitem is not defined
        np.array([LazyFrames(frames) for _ in range(4)])
        """
        self._frames = list(frames)
        self._concat = len(frames[0].shape) == 3
    
    def __array__(self):
        if self._concat:
            out = np.concatenate(self._frames, -1)
        else:
            out = np.stack(self._frames, -1)

        return out


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
    def __init__(self, env, gray_scale, distance=1):
        super().__init__(env)

        self._gray_scale = gray_scale
        self._distance = distance
        self._residual_channel = 1 if self._gray_scale else 3
        w, h, c = self.observation_space.shape
        assert c == 3, self.observation_space.shape
        assert self.observation_space.dtype == np.uint8, self.observation_space.dtype
        assert len(self.observation_space.shape) == 3, self.observation_space.shape
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(w, h, c+self._residual_channel),
            dtype=np.uint8,
        )
        self.observation_space = new_space
        self._buff = np.zeros((w, h, self._residual_channel*(self._distance+1)))
    
    def _append_residual(self, obs):
        res = (self._buff[..., -self._residual_channel:].astype(np.int16) 
            - self._buff[..., :self._residual_channel].astype(np.int16))
        res = (res + 255) // 2
        obs = np.concatenate([obs, res.astype(np.uint8)], axis=-1)
        assert obs.dtype == np.uint8
        return obs
    
    def _add_obs_to_buff(self, obs):
        self._buff = np.roll(self._buff, -self._residual_channel, axis=-1)

        if self._gray_scale:
            self._buff[..., -1] = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            self._buff[..., -self._residual_channel:] = obs

    def reset(self):
        obs = self.env.reset()
        
        buff_obs = np.expand_dims(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), -1) \
            if self._gray_scale else obs
        self._buff = np.tile(buff_obs, [1, 1, self._distance+1])
        obs = self._append_residual(obs)
        
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._add_obs_to_buff(obs)
        res_obs = self._append_residual(obs)
        # self._plot(obs, res_obs)

        return res_obs, rew, done, info

    def _plot(self, obs, res_obs):
        import matplotlib.pyplot as plt
        res_obs = np.squeeze(res_obs[..., -self._residual_channel:])
        fig, axs = plt.subplots(1, 6, figsize=(20, 6))
        fig.suptitle("FrameDifference Plot")
        axs[0].imshow(np.squeeze(self._buff[:, :, :self._residual_channel]))
        axs[0].set_title("oldest frame")
        axs[1].imshow(np.squeeze(self._buff[:, :, -self._residual_channel:]))
        axs[1].set_title("newest frame")
        axs[2].imshow(res_obs)
        axs[2].set_title("frame diff")
        axs[3].imshow(obs)
        axs[3].set_title("newest obs")
        axs[4].hist(res_obs.flatten())
        axs[4].set_title("Frame difference histogram")
        axs[5].hist(obs.flatten())
        axs[5].set_title("Observation histogram")
        print(obs.min())
        print(obs.max())
        print(res_obs.mean())
        print(res_obs.std())
        print()
        plt.show()


class CumulativeRewardObs(gym.Wrapper):
    """Append cumulative reward to observation
    """
    def __init__(self, env, obs_reward_scale):
        super().__init__(env)

        self._cumulative_reward = 0
        self._reward_scale = obs_reward_scale
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        reward_channel_low = np.zeros((*low.shape[:-1], 1), dtype=np.float32)
        reward_channel_high = np.ones((*high.shape[:-1], 1), dtype=np.float32) * np.inf
        low = np.concatenate([low, reward_channel_low], axis=-1)
        high = np.concatenate([high, reward_channel_high], axis=-1)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=low.dtype)

    def _get_ob(self, ob):
        reward_channel = np.ones((*ob.shape[:-1], 1), dtype=np.float32) \
            * self._reward_scale * self._cumulative_reward
        return np.concatenate([ob, reward_channel], axis=-1)

    def reset(self):
        ob = self.env.reset()
        self._cumulative_reward = 0
        return self._get_ob(ob)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self._cumulative_reward += reward
        return self._get_ob(ob), reward, done, info


class RewardHack(gym.Wrapper):
    def __init__(self, env, reward_scale=1, reward_min=None, reward_max=None, **kwargs):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_min = reward_min
        self.reward_max = reward_max

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        info['reward'] = reward
        reward = reward * self.reward_scale
        if self.reward_min is not None or self.reward_max is not None:
            reward = np.clip(reward, self.reward_min, self.reward_max)
        return obs, reward, done, info


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


class StateRecorder(gym.Wrapper):
    def __init__(self, env, rnn_type, state_size):
        super().__init__(env)
        from nn.typing import LSTMState, GRUState
        if rnn_type.endswith('lstm'):
            self.default_states = LSTMState(
                np.zeros((self.n_agents, state_size)), np.zeros((self.n_agents, state_size)))
        elif rnn_type.endswith('gru'):
            self.default_states = GRUState(
                np.zeros((self.n_agents, state_size)))
        self.states = [None for _ in range(self.n_agents)]

    def reset(self):
        obs = self.env.reset()
        self.states = self.default_states.copy()
        obs = self._add_states_to_obs(obs, self.states)

        return obs

    def step(self, action):
        action, self.states = action
        obs, reward, done, info = self.env.step(action)
        obs = self._add_states_to_obs(obs, self.states)

        return obs, reward, done, info

    def record_default_state(self, aid, state):
        self.default_states[aid] = state

    def _add_states_to_obs(self, obs, states):
        keys = list(states[0]._asdict())
        vals = [np.concatenate(s) for s in zip(*states)]
        for k, v in zip(keys, vals):
            obs[k] = v
        return obs


class ContinuousActionMapper(gym.ActionWrapper):
    to_print = True
    def __init__(
        self, 
        env, 
        bound_method='clip', # clip, tanh, rescale
        to_rescale=True, 
        # clip sampled actions, this embodied in training
        action_low=None,
        action_high=None, 
    ):
        self.is_multi_agent = getattr(env, 'is_multi_agent', False)
        if self.is_multi_agent:
            assert np.all([isinstance(a, gym.spaces.Box) for a in env.action_space]), env.action_space
            assert bound_method in ('clip', 'tanh', None), bound_method
        else:
            assert isinstance(env.action_space, gym.spaces.Box), env.action_space
            assert bound_method in ('clip', 'tanh', None), bound_method
        super().__init__(env)

        self.bound_method = bound_method
        self.to_rescale = to_rescale
        if bound_method == 'clip':
            action_low = -1
            action_high = 1
        if self.is_multi_agent:
            self.env_action_low = [a.low for a in self.action_space]
            self.env_action_high = [a.high for a in self.action_space]
        else:
            self.env_action_low = self.action_space.low
            self.env_action_high = self.action_space.high
        self.action_low = action_low
        self.action_high = action_high
        if ContinuousActionMapper.to_print:
            print('Continuous Action Wrapper', self.action_low, self.action_high)
            ContinuousActionMapper.to_print = False
        self._is_random_action = False

    def random_action(self):
        self._is_random_action = True
        return self.env.random_action()

    def action(self, action):
        if self.is_multi_agent:
            if self._is_random_action:
                self._is_random_action = False
                return action
            if self.bound_method == 'clip':
                action = [np.clip(a, -1, 1) for a in action]
            elif self.bound_method == 'tanh':
                action = [np.tanh(a) for a in action]
            if self.to_rescale:
                size = [ah - al for ah, al in zip(self.env_action_high, self.env_action_low)]
                action = [s * (a - self.action_low) / (self.action_high - self.action_low) + al
                    for a, al, s in zip(action, self.env_action_low, size)]
                assert np.all([al <= a for al, a in zip(self.env_action_low, action)]) \
                    and np.all([a <= ah for a, ah, in zip(action, self.env_action_high)]), action
        else:
            if self._is_random_action:
                self._is_random_action = False
                return action
            if self.bound_method == 'clip':
                action = np.clip(action, -1, 1)
            elif self.bound_method == 'tanh':
                action = np.tanh(action)
            if self.to_rescale:
                assert np.all(self.action_low <= action) and np.all(action <= self.action_high), (action, self.action_low, self.action_high)
                size = self.env_action_high - self.env_action_low
                action = size * (action - self.action_low) / (self.action_high - self.action_low) + self.env_action_low
                assert np.all(self.env_action_low <= action) \
                    and np.all(action <= self.env_action_high), action

        return action


class TurnBasedProcess(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.env = env
        self._current_player = -1

        self._prev_action = [np.zeros(ad, dtype=np.float32) 
            for ad in self.env.action_dim]
        self._prev_rewards = np.zeros(self.env.n_units, dtype=np.float32)
        self._dense_score = np.zeros(self.env.n_units, dtype=np.float32)
        self._epslen = np.zeros(self.env.n_units, dtype=np.int32)

    def reset(self):
        self._prev_action = [np.zeros(ad, dtype=np.float32) 
            for ad in self.env.action_dim]
        self._prev_rewards = np.zeros(self.env.n_units, dtype=np.float32)
        self._dense_score = np.zeros(self.env.n_units, dtype=np.float32)
        self._epslen = np.zeros(self.env.n_units, dtype=np.int32)

        obs = self.env.reset()
        self._current_player = obs['uid']
        obs.update({
            'prev_action': self._prev_action[self._current_player],
            'prev_reward': np.float32(0.),
        })

        return obs

    def step(self, action):
        assert self._current_player >= 0, self._current_player
        self._prev_action[self._current_player] = np.zeros(
            self.env.action_dim[self._current_player], dtype=np.float32)
        self._prev_action[self._current_player][np.squeeze(action)] = 1

        # obs is specific to the current player only, 
        # while others are for all players
        obs, rewards, discounts, info = self.env.step(action)
        assert len(rewards) == len(discounts) == self.env.n_units, (len(rewards), len(discounts), self.env.n_units)
        self._current_player = obs['uid']

        acc_rewards = self._get_reward(rewards, self._current_player)
        scores = self._get_scores(rewards)
        self._epslen[self._current_player] += 1

        info.update(scores)
        info['current_player'] = obs['uid']
        info['total_epslen'] = np.sum(self._epslen)
        info['epslen'] = self._epslen

        obs.update({
            'prev_action': self._prev_action[self._current_player],
            'prev_reward': acc_rewards[self._current_player],
        })

        return obs, acc_rewards, discounts, info

    def _get_reward(self, rewards, pid):
        self._prev_rewards += rewards
        acc_rewards = self._prev_rewards.copy()
        self._prev_rewards[pid] = 0

        return acc_rewards
    
    def _get_scores(self, rewards):
        self._dense_score += rewards
        win_score = self._dense_score > 0
        draw_score = self._dense_score == 0
        score = np.sign(self._dense_score)

        return dict(
            dense_score=self._dense_score, 
            win_score=win_score, 
            draw_score=draw_score, 
            score=score, 
        )


class Single2MultiAgent(gym.Wrapper):
    """ Add unit dimension """
    def __init__(self, env, obs_only=False):
        super().__init__(env)
        self._obs_only = obs_only
        self.is_multi_agent = True
        self.obs_shape = {'obs': env.obs_shape}
        self.obs_dtype = {'obs': env.obs_dtype}

    def random_action(self):
        action = self.env.random_action()
        action = np.expand_dims(action, 0)
        return [action]

    def reset(self):
        obs = self.env.reset()
        obs = self._get_obs(obs)

        return [obs]

    def step(self, action, **kwargs):
        action = np.squeeze(action)
        obs, reward, done, info = self.env.step(action, **kwargs)
        obs = self._get_obs(obs)
        if not self._obs_only:
            reward = np.expand_dims(reward, 0)
            done = np.expand_dims(done, 0)

        return [obs], [reward], [done], info

    def _get_obs(self, obs):
        if isinstance(obs, dict):
            for k, v in obs.items():
                obs[k] = np.expand_dims(v, 0)
        else:
            obs = {'obs': np.expand_dims(obs, 0)}
        return obs


class MultiAgentUnitsDivision(gym.Wrapper):
    def __init__(self, env, uid2aid):
        super().__init__(env)

        self.uid2aid = uid2aid
        self.aid2uids = compute_aid2uids(self.uid2aid)
        self.n_units = len(self.uid2aid)
        self.n_agents = len(self.aid2uids)

        if isinstance(self.env.action_space, list):
            self.action_space = self.env.action_space
        else:
            self.action_space = [self.env.action_space for _ in range(self.n_agents)]
        if isinstance(self.env.action_shape, list):
            self.action_shape = self.env.action_shape
        else:
            self.action_shape = [self.env.action_shape for _ in range(self.n_agents)]
        if isinstance(self.env.action_dim, list):
            self.action_dim = self.env.action_dim
        else:
            self.action_dim = [self.env.action_dim for _ in range(self.n_agents)]
        if isinstance(self.env.action_dtype, list):
            self.action_dtype = self.env.action_dtype
        else:
            self.action_dtype = [self.env.action_dtype for _ in range(self.n_agents)]
        if isinstance(self.env.is_action_discrete, list):
            self.is_action_discrete = self.env.is_action_discrete
        else:
            self.is_action_discrete = [self.env.is_action_discrete for _ in range(self.n_agents)]

        if isinstance(self.env.obs_shape, list):
            self.obs_shape = self.env.obs_shape
        else:
            self.obs_shape = [self.env.obs_shape for _ in range(self.n_agents)]
        if isinstance(self.env.obs_dtype, list):
            self.obs_dtype = self.env.obs_dtype
        else:
            self.obs_dtype = [self.env.obs_dtype for _ in range(self.n_agents)]
    
    def reset(self):
        obs = super().reset()
        obs = self._convert_obs(obs)
        return obs
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = self._convert_obs(obs)
        reward = [reward[uids] for uids in self.aid2uids]
        done = [done[uids] for uids in self.aid2uids]

        return obs, reward, done, info
    
    def _convert_obs(self, obs):
        return [{k: v[uids] for k, v in obs.items()} for uids in self.aid2uids]


class PopulationSelection(gym.Wrapper):
    def __init__(self, env, population_size=1):
        super().__init__(env)

        self.population_size = population_size
        self.sids = None

        self.obs_shape = self.env.obs_shape
        self.obs_dtype = self.env.obs_dtype
        if self.population_size > 1:
            if isinstance(self.obs_shape, list):
                for o in self.obs_shape:
                    o['sid'] = (self.population_size,)
            else:
                self.obs_shape['sid'] = (self.population_size,)
            if isinstance(self.obs_dtype, list):
                for o in self.obs_dtype:
                    o['sid'] = np.float32
            else:
                self.obs_dtype['sid'] = np.float32
    
    def reset(self):
        obs = self.env.reset()

        self.reset_sids()
        obs = self._add_population_idx(obs)

        # self._dense_score = np.zeros((self.population_size, self.env.n_units))
        # self._score = np.zeros((self.population_size, self.env.n_units))

        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        obs = self._add_population_idx(obs)

        return obs, reward, done, info

    def reset_sids(self):
        if self.population_size == 1:
            return
        if isinstance(self.obs_shape, (list, tuple)):
            self.sids = []
            for uids in self.env.aid2uids:
                sids = np.random.randint(0, self.population_size, len(uids))
                self.sids.append(np.array([
                    one_hot(sid, self.population_size) for sid in sids
                ], np.float32))
        else:
            sid = np.random.randint(0, self.population_size)
            self.sids = np.array(one_hot(sid, self.population_size), np.float32)

    def _add_population_idx(self, obs):
        if self.population_size == 1:
            return obs
        if isinstance(obs, (list, tuple)):
            for o, sid in zip(obs, self.sids):
                o['sid'] = sid
        else:
            obs['sid'] = sid
        return obs


class DataProcess(gym.Wrapper):
    """ Convert observation to np.float32 or np.float16 """
    def __init__(self, env, precision=32):
        super().__init__(env)
        self.precision = precision
        self.float_dtype = np.float32 if precision == 32 else np.float16

        if not hasattr(self.env, 'is_action_discrete'):
            self.is_action_discrete = isinstance(
                self.env.action_space, gym.spaces.Discrete)
        if not self.is_action_discrete and precision == 16:
            self.action_space = gym.spaces.Box(
                self.action_space.low, self.action_space.high, 
                self.action_space.shape, self.float_dtype)
        if not hasattr(self.env, 'obs_shape'):
            self.obs_shape = {'obs': self.observation_space.shape}
        if not hasattr(self.env, 'obs_dtype'):
            self.obs_dtype = {'obs': infer_dtype(self.observation_space.dtype, precision)}
        if not hasattr(self.env, 'action_shape'):
            self.action_shape = self.action_space.shape
        if not hasattr(self.env, 'action_dim'):
            self.action_dim = self.action_space.n if self.is_action_discrete else self.action_shape[0]
        if not hasattr(self.env, 'action_dtype'):
            self.action_dtype = np.int32 if self.is_action_discrete \
                else infer_dtype(self.action_space.dtype, self.precision)

    def observation(self, observation):
        if isinstance(observation, list):
            return [self.observation(o) for o in observation]
        elif isinstance(observation, dict):
            for k, v in observation.items():
                observation[k] = convert_dtype(v, self.precision)
        return observation
    
    # def action(self, action):
    #     if isinstance(action, np.ndarray):
    #         return convert_dtype(action, self.precision)
    #     return np.int32(action) # always keep int32 for integers as tf.one_hot does not support int16

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        return self.observation(obs), reward, done, info


""" Subclasses of EnvStatsBase change the gym API:
Both <reset> and <step> return EnvOutput of form
(obs, reward, discount, reset), where 
    - obs is a dict regardless of the original form of obs
    - reward is the reward from the original env 
    - discount=1-done is the discount factor
    - reset indicates if the environment has been reset. 

By default, EnvStats automatically
reset the environment when the environment is done.
Explicitly calling EnvStats.reset turns off auto-reset.

For some environments truncated by max episode steps,
we recommand to retrieve the last observation of an 
episode using method "prev_obs"

We distinguish several signals:
    done: an episode is done, may due to life loss(Atari)
    game over: a game is over, may due to timeout. Life 
        loss in Atari is not game over. Do store <game_over> 
        in <info> for multi-agent environments.
    reset: a new episode starts after done. In auto-reset 
        mode, environment resets when the game's over. 
        Life loss should be automatically handled by 
        the environment/previous wrapper.
"""
class EnvStatsBase(gym.Wrapper):
    def __init__(
        self, 
        env, 
        max_episode_steps=None, 
        timeout_done=False, 
        auto_reset=True
    ):
        """ Records environment statistics """
        super().__init__(env)
        if max_episode_steps is None:
            if hasattr(self.env, 'max_episode_steps'):
                max_episode_steps = self.env.max_episode_steps
            elif hasattr(self.env, 'spec'):
                max_episode_steps = self.env.spec.max_episode_steps
            else:
                max_episode_steps = int(1e9)
        self.max_episode_steps = max_episode_steps
        # if we take timeout as done
        self.timeout_done = timeout_done
        self.auto_reset = auto_reset
        self.n_envs = getattr(self.env, 'n_envs', 1)
        # game_over indicates whether an episode is finished, 
        # either due to timeout or due to environment done
        self._game_over = True
        self._score = 0         # the final metrics for performance evaluation (i.e., win rate)
        self._dense_score = 0   # the (shaped) reward for training
        self._epslen = 0
        self._info = {}
        self._output = None
        self.float_dtype = getattr(self.env, 'float_dtype', np.float32)
        if hasattr(self.env, 'stats'):
            self._stats = dict2AttrDict(self.env.stats)
        else:
            self.n_agents = getattr(self.env, 'n_agents', 1)
            self.n_units = getattr(self.env, 'n_units', 1)
            self.uid2aid = getattr(self.env, 'uid2aid', [0 for _ in range(self.n_units)])
            self.aid2uids = getattr(self.env, 'aid2uids', compute_aid2uids(self.uid2aid))
            self._stats = AttrDict(
                obs_shape=env.obs_shape,
                obs_dtype=env.obs_dtype,
                action_shape=env.action_shape,
                action_dim=env.action_dim,
                action_low=getattr(env, 'action_low', None), 
                action_high=getattr(env, 'action_high', None), 
                is_action_discrete=env.is_action_discrete,
                action_dtype=env.action_dtype,
                n_agents=self.n_agents,
                n_units=self.n_units,
                uid2aid=self.uid2aid,
                aid2uids=self.aid2uids,
                use_life_mask=getattr(env, 'use_life_mask', False),
                use_action_mask=getattr(env, 'use_action_mask', False),
                is_multi_agent=getattr(env, 'is_multi_agent', len(self.uid2aid) > 1),
                is_simultaneous_move=getattr(env, 'is_simultaneous_move', True),
            )
        if 'obs_keys' not in self._stats:
            if getattr(env, 'obs_keys', None):
                self._stats['obs_keys'] = env.obs_keys
            elif isinstance(env.obs_shape, list):
                self._stats['obs_keys'] = [list(o) for o in env.obs_shape]
            else:
                self._stats['obs_keys'] = list(env.obs_shape)
        if timeout_done:
            do_logging('Timeout is treated as done', logger=logger)
        self._reset()

    def stats(self):
        # return a copy to avoid being modified from outside
        return dict2AttrDict(self._stats, to_copy=True)

    def reset(self):
        raise NotImplementedError

    def _reset(self):
        obs = self.env.reset()
        self._score = 0
        self._epslen = 0
        self._game_over = False
        return self.observation(obs)

    def manual_reset(self):
        self.auto_reset = False

    def score(self, **kwargs):
        return self._info.get('score', self._score)

    def epslen(self, **kwargs):
        return self._info.get('epslen', self._epslen)

    def game_over(self):
        return self._game_over

    def prev_obs(self):
        return self._prev_env_output.obs

    def info(self):
        return self._info
        
    def output(self):
        return self._output


class EnvStats(EnvStatsBase):
    manual_reset_warning = True
    def reset(self):
        # if self.auto_reset:
        #     self.auto_reset = False
        #     if EnvStats.manual_reset_warning:
        #         do_logging('Explicitly resetting turns off auto-reset. Maker sure this is done intentionally at evaluation', logger=logger)
        #         EnvStats.manual_reset_warning = False
        if not self._output.reset:
            return self._reset()
        else:
            if EnvStats.manual_reset_warning:
                logger.debug('Repetitively calling reset results in no environment interaction')
            return self._output

    def _reset(self):
        obs = super()._reset()
        if self._stats['is_multi_agent']:
            reward = [np.zeros(1, self.float_dtype)]
            discount = [np.ones(1, self.float_dtype)]
            reset = [np.ones(1, self.float_dtype)]
        else:
            reward = self.float_dtype(0)
            discount = self.float_dtype(1)
            reset = self.float_dtype(True)
        self._output = EnvOutput(obs, reward, discount, reset)

        return self._output

    def step(self, action, **kwargs):
        if self._game_over:
            assert self.auto_reset == False, self.auto_reset
            # step after the game is over
            if self._stats.is_multi_agent:
                reward = [np.zeros(1, self.float_dtype)]
                discount = [np.zeros(1, self.float_dtype)]
                reset = [np.zeros(1, self.float_dtype)]
            else:
                reward = self.float_dtype(0)
                discount = self.float_dtype(0)
                reset = self.float_dtype(0)
            self._output = EnvOutput(self._output.obs, reward, discount, reset)
            return self._output

        # assert not np.any(np.isnan(action)), action
        obs, reward, done, info = self.env.step(action, **kwargs)
        if 'score' in info:
            self._score = info['score']
        else:
            self._score += info.get('reward', reward)
        self._dense_score = info['dense_score'] = info.get('dense_score', self._score)
        if 'epslen' in info:
            self._epslen = info['epslen']
        else:
            self._epslen += 1
        self._game_over = bool(info.get(
            'game_over', done[0] if self._stats.is_multi_agent else done))
        if self._epslen >= self.max_episode_steps:
            self._game_over = True
            if self._stats['is_multi_agent']:
                done = [np.ones(1, self.float_dtype) * self.timeout_done]
            else:
                done = self.timeout_done
        
        # we expect auto-reset environments, 
        # which artificially reset due to life loss,
        # return reset in info when resetting
        if self._stats.is_multi_agent:
            reward = [self.float_dtype(reward[0])]
            discount = [self.float_dtype(1-done[0])]
            reset = [np.array([info.get('reset', False)], dtype=self.float_dtype)]
        else:
            reward = self.float_dtype(reward)
            discount = self.float_dtype(1-done)
            reset = self.float_dtype(info.get('reset', False))
        # store previous env output for later retrieval
        self._prev_env_output = GymOutput(obs, reward, discount)

        assert isinstance(self._game_over, bool), self._game_over
        # reset env
        if self._game_over:
            info['score'] = self._score
            info['epslen'] = self._epslen
            if self.auto_reset:
                # when resetting, we override the obs and reset but keep the others
                obs, _, _, reset = self._reset()

        self._info = info
        
        self._output = EnvOutput(obs, reward, discount, reset)
        return self._output


class SqueezeObs(gym.Wrapper):
    """ Squeeze the unit dimension of keys in observation """
    def __init__(self, env, keys=[]):
        super().__init__(env)
        self.env = env
        self._keys = keys

    def reset(self):
        obs = self.env.reset()
        obs = self._squeeze_obs(obs)

        return obs
    
    def step(self, action, **kwargs):
        obs, reward, discount, info = self.env.step(action, **kwargs)
        obs = self._squeeze_obs(obs)

        return obs, reward, discount, info

    def _squeeze_obs(self, obs):
        if isinstance(obs, dict):
            for k in self._keys:
                obs[k] = np.squeeze(obs[k], 0)
        else:
            obs = np.squeeze(obs, 0)
        return obs

class MASimEnvStats(EnvStatsBase):
    """ Wrapper for multi-agent simutaneous environments
    <MASimEnvStats> expects agent-wise reward and done signal per step.
    Otherwise, go for <EnvStats>
    """
    manual_reset_warning = True
    def __init__(self, 
                 env, 
                 max_episode_steps=None, 
                 timeout_done=False, 
                 auto_reset=True):
        super().__init__(
            env, 
            max_episode_steps=max_episode_steps, 
            timeout_done=timeout_done, 
            auto_reset=auto_reset
        )
        self._stats.is_multi_agent = True

    def reset(self):
        # if self.auto_reset:
        #     self.auto_reset = False
        #     if EnvStats.manual_reset_warning:
        #         do_logging('Explicitly resetting turns off auto-reset. Maker sure this is done intentionally at evaluation', logger=logger)
        #         EnvStats.manual_reset_warning = False
        if not np.any(self._output.reset):
            return self._reset()
        else:
            logger.debug('Repetitively calling reset results in no environment interaction')
            return self._output

    def _reset(self):
        obs = super()._reset()
        reward = self._get_agent_wise_zeros()
        discount = self._get_agent_wise_ones()
        reset = self._get_agent_wise_ones()
        self._output = EnvOutput(obs, reward, discount, reset)

        return self._output

    def step(self, action, **kwargs):
        if self.game_over():
            assert self.auto_reset == False
            # step after the game is over
            reward = self._get_agent_wise_zeros()
            discount = self._get_agent_wise_zeros()
            reset = self._get_agent_wise_zeros()
            self._output = EnvOutput(self._output.obs, reward, discount, reset)
            return self._output

        assert not np.any(np.isnan(action)), action
        obs, reward, done, info = self.env.step(action, **kwargs)
        # expect score, epslen, and game_over in info as multi-agent environments may vary in metrics 
        self._score = info['score']
        self._dense_score = info['dense_score']
        self._epslen = info['epslen']
        self._game_over = info.pop('game_over')
        if self._epslen >= self.max_episode_steps:
            self._game_over = True
            if self.timeout_done:
                done = self._get_agent_wise_ones()
        discount = [np.array(1-d, self.float_dtype) for d in done]

        # store previous env output for later retrieval
        self._prev_env_output = GymOutput(obs, reward, discount)

        # reset env
        if self._game_over and self.auto_reset:
            # when resetting, we override the obs and reset but keep the others
            obs, _, _, reset = self._reset()
        else:
            reset = self._get_agent_wise_zeros()
        obs = self.observation(obs)
        self._info = info

        # we return reward and discount for all agents so that
        # 
        # while obs and reset is for the current agent only
        self._output = EnvOutput(obs, reward, discount, reset)
        # assert np.all(done) == info.get('game_over', False), (reset, info['game_over'])
        # assert np.all(reset) == info.get('game_over', False), (reset, info['game_over'])
        return self._output

    def observation(self, obs):
        if isinstance(obs, dict):
            obs = [obs]
        assert isinstance(obs, list) and len(obs) == self.n_agents, obs
        assert isinstance(obs[0], dict), obs[0]
        return obs

    def _get_agent_wise_zeros(self):
        return [np.zeros(len(uids), self.float_dtype) for uids in self.aid2uids]
    
    def _get_agent_wise_ones(self):
        return [np.ones(len(uids), self.float_dtype) for uids in self.aid2uids]


class MATurnBasedEnvStats(EnvStatsBase):
    """ Wrapper for multi-agent simutaneous environments
    <MASimEnvStats> expects agent-wise reward and done signal per step.
    Otherwise, go for <EnvStats>
    """
    manual_reset_warning = True
    def __init__(self, 
                 env, 
                 max_episode_steps=None, 
                 timeout_done=False, 
                 auto_reset=True):
        super().__init__(
            env, 
            max_episode_steps=max_episode_steps, 
            timeout_done=timeout_done, 
            auto_reset=auto_reset
        )
        self._stats.is_multi_agent = True
        self._stats.is_simultaneous_move = False

    def reset(self):
        if not np.any(self._output.reset):
            return self._reset()
        else:
            logger.debug('Repetitively calling reset results in no environment interaction')
            return self._output

    def _reset(self):
        obs = super()._reset()
        reward = self._get_zeros()
        discount = self._get_ones()
        self._resets = np.ones(self.env.n_units, dtype=np.float32)
        reset = np.expand_dims(self._resets[obs['uid']], 0)
        self._resets[obs['uid']] = 0
        self._output = EnvOutput(obs, reward, discount, reset)

        return self._output

    def step(self, action, **kwargs):
        assert not self._game_over, self._game_over
        assert not np.any(np.isnan(action)), action
        obs, reward, discount, info = self.env.step(action, **kwargs)
        # expect score, epslen, and game_over in info as multi-agent environments may vary in metrics 
        self._score = info['score']
        self._dense_score = info['dense_score']
        self._epslen = info['epslen']
        self._game_over = info.pop('game_over')
        if np.sum(self._epslen) >= self.max_episode_steps:
            self._game_over = True
        if np.any(discount == 0):
            assert self._game_over, self._game_over
        if self._game_over:
            assert np.all(discount == 0), discount

        # store previous env output for later retrieval
        self._prev_env_output = GymOutput(obs, reward, discount)

        # reset env
        if self._game_over and self.auto_reset:
            # when resetting, we override the obs and reset but keep the others
            obs, _, _, reset = self._reset()
            np.testing.assert_array_equal(discount, 0)
        else:
            reset = np.expand_dims(self._resets[obs['uid']], 0)
            self._resets[obs['uid']] = 0
        self._info = info

        self._output = EnvOutput(obs, reward, discount, reset)
        # assert np.all(done) == info.get('game_over', False), (reset, info['game_over'])
        # assert np.all(reset) == info.get('game_over', False), (reset, info['game_over'])
        return self._output

    def observation(self, obs):
        return obs

    def _get_zeros(self):
        return np.zeros(self.n_units, self.float_dtype)
    
    def _get_ones(self):
        return np.ones(self.n_units, self.float_dtype)


class UnityEnvStats(EnvStatsBase):
    def reset(self):
        if not self._output.reset:
            return self._reset()
        else:
            if EnvStats.manual_reset_warning:
                logger.debug('Repetitively calling reset results in no environment interaction')
            return self._output

    def _reset(self):
        obs = self.env.reset()
        self._score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._dense_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._epslen = np.zeros(self.n_envs, np.int32)

        self._game_over = False
        reward = [np.zeros((self.n_envs, len(uids))) for uids in self.aid2uids]
        discount = [np.ones((self.n_envs, len(uids))) for uids in self.aid2uids]
        reset = [np.ones((self.n_envs, len(uids)), dtype=bool) for uids in self.aid2uids]
        self._output = EnvOutput(obs, reward, discount, reset)

        return self._output

    def step(self, action, **kwargs):
        action = copy.deepcopy(action)
        kwargs = copy.deepcopy(kwargs)
        # assert not np.any(np.isnan(action)), action
        obs, reward, discount, reset = self.env.step(action, **kwargs)
        
        self._info = info = self.env.info()
        self._score = [i['score'] for i in info]
        self._dense_score = [i['dense_score'] for i in info]
        self._epslen = [i['epslen'] for i in info]
        self._game_over = [i.pop('game_over') for i in info]

        self._output = EnvOutput(obs, reward, discount, reset)
        return self._output

    def info(self, eids=None):
        info = self.env.info()
        if eids is None:
            return info
        else:
            return [info[i] for i in eids]

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
        name='smac_3s5z',
        seed=0
    ))

    for i in range(10000):
        a = env.random_action()
        out = env.step(a)
        print(out[2:])
        if np.all(out.reset):
            info = env.info()
            print(info['score'], info['epslen'])
