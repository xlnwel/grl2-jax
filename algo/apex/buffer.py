from abc import ABC, abstractmethod
import numpy as np

from utility.display import assert_colorize
from utility.run_avg import RunningMeanStd
from replay.utils import init_buffer, add_buffer, copy_buffer


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
        self.type = config['type']
        self.seqlen = seqlen = config['seqlen']
        self.n_steps = config['n_steps']
        self.gamma = config['gamma']

        self.memory = {}
        # init_buffer(self.memory, *keys, capacity=self.seqlen+1)
        
        self.reward_scale = config.get('reward_scale', 1)
        self.reward_clip = config.get('reward_clip')
        self.normalize_reward = config.get('normalize_reward', False)
        if self.normalize_reward:
            self.running_reward_stats = RunningMeanStd()
        
        self.idx = 0

    def is_full(self):
        return self.idx == self.seqlen

    def sample(self):
        results = {}
        for k, v in self.memory.items():
            if 'state' in k or 'action' in k:
                results[k] = np.array(v[:self.idx], copy=False)
            else:
                results[k] = np.array(v[:self.idx], copy=False, dtype=np.float32)
        
        indexes = np.arange(self.idx)
        steps = results['steps'].astype(np.int32)
        next_indexes = indexes + steps
        results['next_state'] = np.array([self.memory['state'][i] for i in next_indexes], copy=False)

        # process rewards
        results['reward'] *= np.where(results['done'], 1, self.reward_scale)
        if self.reward_clip:
            results['reward'] = np.clip(results['reward'], -self.reward_clip, self.reward_clip)
        if self.normalize_reward:
            # since we only expect rewards to be used once
            # we update the running stats when we use them
            self.running_reward_stats.update(results['reward'])
            results['reward'] = self.running_reward_stats.normalize(results['reward'])

        return None, results

    def reset(self):
        self.idx = 0
        
    def add_data(self, **kwargs):
        """ Add experience to local memory """
        next_state = kwargs['next_state']
        if self.memory == {}:
            del kwargs['next_state']
            keys = list(kwargs)
            keys.append('steps')
            init_buffer(self.memory, *keys, capacity=self.seqlen+1)
            print(f'Local bufffer keys: {list(self.memory.keys())}')
            
        add_buffer(self.memory, self.idx, self.n_steps, self.gamma, **kwargs)
        self.idx = self.idx + 1
        self.memory['state'][self.idx] = next_state


class EnvVecBuffer:
    """ Local memory only stores one episode of transitions from n environments """
    def __init__(self, config, *keys):
        self.type = config['type']
        self.n_envs = n_envs = config['n_envs']
        assert n_envs > 1
        self.seqlen = seqlen = config['seqlen']
        self.n_steps = config['n_steps']
        self.gamma = config['gamma']

        self.memory = dict(
            state=np.ndarray((n_envs, seqlen), dtype=object),
            action=np.ndarray((n_envs, seqlen), dtype=object),
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
        results = {}
        mask = self.memory['mask']
        for k, v in self.memory.items():
            if 'state' in k or 'action' in k:
                results[k] = np.stack(v[mask])
            elif k == 'mask':
                continue
            else:
                results[k] = v[mask]

        # process rewards
        results['reward'] *= np.where(results['done'], 1, self.reward_scale)
        if self.reward_clip:
            results['reward'] = np.clip(results['reward'], -self.reward_clip, self.reward_clip)
        if self.normalize_reward:
            # since we only expect rewards to be used once
            # we update the running stats when we use them
            self.running_reward_stats.update(results['reward'])
            results['reward'] = self.running_reward_stats.normalize(results['reward'])

        return mask, results

    def reset(self):
        self.idx = 0
        self.memory['mask'] = np.zeros_like(self.memory['mask'], dtype=np.bool)
        
    def add_data(self, env_ids=None, **kwargs):
        """ Add experience to local memory """
        env_ids = env_ids or range(self.n_envs)
        idx = self.idx
        for i, env_id in enumerate(env_ids):
            for k, v in kwargs.items():
                self.memory[k][env_id, idx] = v[i]
            self.memory['steps'][env_id, idx] = 1

            # Update previous experience if multi-step is required
            for j in range(1, self.n_steps):
                k = idx - j
                k_done = self.memory['done'][i, k]
                if k_done:
                    break
                self.memory['reward'][i, k] += self.gamma**i * kwargs['reward'][i]
                self.memory['done'][i, k] = kwargs['done'][i]
                self.memory['steps'][i, k] += 1
                self.memory['next_state'][i, k] = kwargs['next_state'][i]

        self.idx = self.idx + 1
