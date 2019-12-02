from abc import ABC, abstractmethod
import numpy as np

from utility.display import assert_colorize
from utility.run_avg import RunningMeanStd
from replay.utils import init_buffer, add_buffer, copy_buffer


def create_local_buffer(config, state_shape, state_dtype, 
                action_dim, action_dtype, gamma):
    buffer_type = config['type']
    if buffer_type == 'env':
        return EnvBuffer(config, state_shape, state_dtype, action_dim, action_dtype, gamma)
    elif buffer_type == 'envvec':
        return EnvVecBuffer(config, state_shape, state_dtype, action_dim, action_dtype, gamma)


class LocalBuffer(ABC):
    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def add_data(self, state, action, reward, done, next_state, mask):
        raise NotImplementedError


class EnvBuffer(LocalBuffer):
    """ Local buffer only stores one episode of transitions from n environments """
    def __init__(self, config, state_shape, state_dtype, 
                action_dim, action_dtype, gamma):
        self.epslen = epslen = config['epslen']
        self.n_steps = config['n_steps']
        self.gamma = gamma

        self.buffer = {}
        init_buffer(
            self.buffer, epslen+1, state_shape, state_dtype,
            action_dim, action_dtype, True, False
        )
        
        self.reward_scale = config.get('reward_scale', 1)
        self.reward_clip = config.get('reward_clip')
        self.normalize_reward = config['normalize_reward']
        if self.normalize_reward:
            self.running_reward_stats = RunningMeanStd()
        
        self.idx = 0

    def sample(self):
        state = self.buffer['state'][:self.idx]
        action = self.buffer['action'][:self.idx]
        reward = np.copy(self.buffer['reward'][:self.idx])
        done = self.buffer['done'][:self.idx]
        steps = self.buffer['steps'][:self.idx]
        next_state = self.buffer['state'][1:self.idx+1]

        # process rewards
        if self.normalize_reward:
            # since we only expect rewards to be used once
            # we update the running stats when we use them
            self.running_reward_stats.update(reward)
            reward = self.running_reward_stats.normalize(reward)
        reward *= np.where(done, 1, self.reward_scale)
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)

        return dict(
            state=state,
            action=action,
            reward=reward, 
            done=done,
            steps=steps,
            next_state=next_state,
        )

    def reset(self):
        self.idx = 0
        
    def add_data(self, state, action, reward, done, next_state=None, mask=None):
        """ Add experience to local buffer, 
        next_state and mask here are only for interface consistency 
        """
        idx = self.idx
        add_buffer(self.buffer, idx, state, action, reward, done, self.n_steps, self.gamma)
        self.idx = self.idx + 1


class EnvVecBuffer:
    """ Local buffer only stores one episode of transitions from n environments """
    def __init__(self, config, state_shape, state_dtype, 
                action_dim, action_dtype, gamma):
        self.n_envs = n_envs = config['n_envs']
        assert n_envs > 1, "currently don't support n_envs == 1"
        self.epslen = epslen = config['epslen']
        self.n_steps = config['n_steps']
        self.gamma = gamma

        self.buffer = dict(
            state=np.zeros((n_envs, epslen, *state_shape), dtype=state_dtype),
            action=np.zeros((n_envs, epslen, action_dim), dtype=action_dtype),
            reward=np.zeros((n_envs, epslen, 1), dtype=np.float32),
            done=np.zeros((n_envs, epslen, 1), dtype=np.bool),
            steps=np.zeros((n_envs, epslen, 1), dtype=np.uint8),
            next_state=np.zeros((n_envs, epslen, *state_shape), dtype=np.float32),
            mask=np.zeros((n_envs, epslen), dtype=np.bool)
        )
        
        self.reward_scale = config.get('reward_scale', 1)
        self.reward_clip = config.get('reward_clip')
        self.normalize_reward = config['normalize_reward']
        if self.normalize_reward:
            self.running_reward_stats = RunningMeanStd()
        
        self.idx = 0

    def sample(self, env_mask=None):
        state = self.buffer['state'][env_mask]
        action = self.buffer['action'][env_mask]
        reward = np.copy(self.buffer['reward'][env_mask])
        done = self.buffer['done'][env_mask]
        steps = self.buffer['steps'][env_mask]
        next_state = self.buffer['next_state'][env_mask]
        mask = self.buffer['mask'][env_mask]

        state = state[mask]
        action = action[mask]
        reward = reward[mask]
        done = done[mask]
        steps = steps[mask]
        next_state = next_state[mask]

        # process rewards
        if self.normalize_reward:
            # since we only expect rewards to be used once
            # we update the running stats when we use them
            self.running_reward_stats.update(reward)
            reward = self.running_reward_stats.normalize(reward)
        reward *= np.where(done, 1, self.reward_scale)
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)

        return dict(
            state=state,
            action=action,
            reward=reward, 
            done=done,
            steps=steps,
            next_state=next_state,
        )

    def reset(self):
        self.idx = 0
        self.buffer['mask'] = np.zeros_like(self.buffer['mask'], dtype=np.bool)
        
    def add_data(self, state, action, reward, done, next_state, mask):
        """ Add experience to local buffer """
        idx = self.idx
        self.buffer['state'][:, idx] = state
        self.buffer['action'][:, idx] = action
        self.buffer['reward'][:, idx] = reward
        self.buffer['done'][:, idx] = done
        self.buffer['steps'][:, idx] = 1
        self.buffer['mask'][:, idx] = mask
        self.buffer['next_state'][:, idx] = next_state
        # Update previous experience if multi-step is required
        for i in range(1, self.n_steps):
            k = idx - i
            k_done = self.buffer['done'][:, k]
            self.buffer['reward'][:, k] += np.where(k_done, 0, self.gamma**i * reward)
            self.buffer['done'][:, k] = np.where(k_done, k_done, done)
            self.buffer['steps'][:, k] += np.where(k_done, 0, 1).astype(np.uint8)
            self.buffer['next_state'][:, k] = np.where(k_done, self.buffer['next_state'][:, k], next_state)

        self.idx = self.idx + 1
