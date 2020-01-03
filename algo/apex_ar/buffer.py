from abc import ABC, abstractmethod
import numpy as np

from utility.display import assert_colorize
from utility.run_avg import RunningMeanStd
from algo.sacar.replay.utils import init_buffer, add_buffer, copy_buffer


def create_local_buffer(config, *keys):
    buffer_type = EnvBuffer if config.get('n_envs', 1) == 1 else EnvVecBuffer
    return buffer_type(config, *keys)


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
    """ Local memory only stores one episode of transitions from each of n environments """
    def __init__(self, config, *keys):
        self.seqlen = seqlen = config['seqlen']
        self.n_steps = config['n_steps']
        self.gamma = config['gamma']

        self.memory = {}
        init_buffer(self.memory, *keys, capacity=self.seqlen+1)
        
        self.reward_scale = config.get('reward_scale', 1)
        self.reward_clip = config.get('reward_clip')
        self.normalize_reward = config.get('normalize_reward', False)
        if self.normalize_reward:
            self.running_reward_stats = RunningMeanStd()
        
        self.idx = 0

    def is_full(self):
        return self.idx == self.seqlen

    def sample(self):
        state = self.memory['state']
        action = self.memory['action']
        n_ar = self.memory['n_ar']
        reward = np.copy(self.memory['reward'])
        done = self.memory['done']
        steps = self.memory['steps']
        next_state = self.memory['next_state']

        assert len(state.shape) == 2 or state.dtype == np.uint8
        if state.dtype == np.uint8:
            state = state / 255.
            next_state = next_state / 255.
            
        # process rewards
        reward *= np.where(done, 1, self.reward_scale)
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        if self.normalize_reward:
            # since we only expect rewards to be used once
            # we update the running stats when we use them
            self.running_reward_stats.update(reward)
            reward = self.running_reward_stats.normalize(reward)

        return dict(
            state=state.astype(np.float32),
            action=action,
            n_ar=n_ar,
            reward=np.expand_dims(reward, -1).astype(np.float32), 
            done=np.expand_dims(done, -1).astype(np.float32),
            steps=np.expand_dims(steps, -1).astype(np.float32),
            next_state=next_state.astype(np.float32),
        )

    def reset(self):
        self.idx = 0
        
    def add_data(self, state, action, n_ar, reward, done, next_state):
        """ Add experience to local memory, 
        next_state and mask here are only for interface consistency 
        """
        add_buffer(self.memory, self.idx, self.n_steps, self.gamma, 
                    state=state, action=action, n_ar=n_ar, reward=reward,
                    done=done, next_state=next_state)
        self.idx = self.idx + 1


class EnvVecBuffer:
    """ Local memory only stores one episode of transitions from n environments """
    def __init__(self, config, *keys):
        self.n_envs = n_envs = config['n_envs']
        assert n_envs > 1
        self.seqlen = seqlen = config['seqlen']
        self.n_steps = config['n_steps']
        self.gamma = config['gamma']

        self.memory = dict(
            state=np.ndarray((n_envs, seqlen), dtype=object),
            action=np.ndarray((n_envs, seqlen), dtype=object),
            n_ar=np.ndarray((n_envs, seqlen), dtype=np.uint8),
            reward=np.ndarray((n_envs, seqlen), dtype=np.float32),
            done=np.zeros((n_envs, seqlen), dtype=np.bool),
            steps=np.zeros((n_envs, seqlen), dtype=np.uint8),
            next_state=np.ndarray((n_envs, seqlen), dtype=object),
            mask=np.zeros((n_envs, seqlen), dtype=np.bool)
        )
        
        self.reward_scale = config.get('reward_scale', 1)
        self.reward_clip = config.get('reward_clip')
        self.normalize_reward = config.get('normalize_reward', False)
        if self.normalize_reward:
            self.running_reward_stats = RunningMeanStd()
        
        self.idx = 0

    def is_full(self):
        return self.idx == self.seqlen
        
    def sample(self):
        state = self.memory['state']
        action = self.memory['action']
        n_ar = self.memory['n_ar']
        reward = np.copy(self.memory['reward'])
        done = self.memory['done']
        steps = self.memory['steps']
        next_state = self.memory['next_state']
        mask = self.memory['mask']
            
        state = np.stack(state[mask])
        action = np.stack(action[mask])
        n_ar = n_ar[mask]
        reward = reward[mask]
        done = done[mask]
        steps = steps[mask]
        next_state = np.stack(next_state[mask])
        
        assert len(state.shape) == 2 or state.dtype == np.uint8
        if state.dtype == np.uint8:
            state = state / 255.
            next_state = next_state / 255.

        # process rewards
        reward *= np.where(done, 1, self.reward_scale)
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        if self.normalize_reward:
            # since we only expect rewards to be used once
            # we update the running stats when we use them
            self.running_reward_stats.update(reward)
            reward = self.running_reward_stats.normalize(reward)

        return mask, dict(
            state=state.astype(np.float32),
            action=action,
            n_ar=n_ar,
            reward=np.expand_dims(reward, -1).astype(np.float32), 
            done=np.expand_dims(done, -1).astype(np.float32),
            steps=np.expand_dims(steps, -1).astype(np.float32),
            next_state=next_state.astype(np.float32),
        )

    def reset(self):
        self.idx = 0
        self.memory['mask'] = np.zeros_like(self.memory['mask'], dtype=np.bool)
        
    def add_data(self, state, action, n_ar, reward, done, next_state, mask, env_ids=None):
        """ Add experience to local memory """
        env_ids = env_ids or range(self.n_envs)
        idx = self.idx
        for i, env_id in enumerate(env_ids):
            self.memory['state'][env_id, idx] = state[i]
            self.memory['action'][env_id, idx] = action[i]
            self.memory['n_ar'][env_id, idx] = n_ar[i]
            self.memory['reward'][env_id, idx] = reward[i]
            self.memory['done'][env_id, idx] = done[i]
            self.memory['steps'][env_id, idx] = 1
            self.memory['mask'][env_id, idx] = mask[i]
            self.memory['next_state'][env_id, idx] = next_state[i]
            # Update previous experience if multi-step is required
            for j in range(1, self.n_steps):
                k = idx - j
                k_done = self.memory['done'][i, k]
                if k_done:
                    break
                self.memory['reward'][i, k] += self.gamma**i * reward[i]
                self.memory['done'][i, k] = done[i]
                self.memory['steps'][i, k] += 1
                self.memory['next_state'][i, k] = next_state[i]

        self.idx = self.idx + 1
