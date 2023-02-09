import collections
import logging
import time
import numpy as np

from core.elements.buffer import Buffer
from core.elements.model import Model
from core.log import do_logging
from core.typing import AttrDict, dict2AttrDict
from tools.utils import standardize, stack_data_with_state

logger = logging.getLogger(__name__)


def concat_obs(obs, last_obs):
    obs = np.concatenate([obs, np.expand_dims(last_obs, 0)], 0)
    return obs


class LocalBuffer(Buffer):
    def __init__(
        self, 
        config: AttrDict,
        env_stats: AttrDict,  
        model: Model,
        aid: int, 
        runner_id: int,
    ):
        super().__init__(config, env_stats, model, aid)

        self.runner_id = runner_id

        self.n_steps = self.config.n_steps
        self.n_envs = self.config.n_envs

        self.reset()

    def size(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def is_full(self):
        return self._size >= self.n_steps

    def reset(self):
        self._size = 0
        self._buffer = collections.defaultdict(list)

    def collect(self, env, env_step, reset, obs, next_obs, **kwargs):
        for k, v in obs.items():
            if k not in kwargs:
                kwargs[k] = v
        for k, v in next_obs.items():
            kwargs[f'next_{k}'] = v
        self.add(**kwargs, reset=reset)

    def add(self, data: dict):
        self._replace_obs_with_next_obs(data)
        for k, v in data.items():
            self._buffer[k].append(v)
        self._size += 1

    def retrieve_all_data(self, latest_piece=None):
        # assert self._size == self.n_steps, (self._size, self.n_steps)
        if latest_piece is not None:
            for k, v in latest_piece.items():
                if k in self._buffer:
                    self._buffer[k].append(v)

        data = stack_data_with_state(self._buffer, self.sample_keys)
        self.reset()

        return self.runner_id, data, self.n_envs * self.n_steps

    def _replace_obs_with_next_obs(self, data):
        if f'next_{self.obs_keys[0]}' not in data:
            return
        if self.config.timeout_done:
            # for environments that treat timeout as done, 
            # we do not separately record obs and next obs
            # instead, we record n+1 obs 
            if self._size == 0:
                for k in self.obs_keys:
                    assert data[k].shape == data[f'next_{k}'].shape, (data[k].shape, data[f'next_{k}'].shape)
                    # add the first obs when buffer is empty
                    self._buffer[k].append(data[k])
                    data[k] = data.pop(f'next_{k}')
            else:
                for k in self.obs_keys:
                    data[k] = data.pop(f'next_{k}')


class ACBuffer(Buffer):
    def __init__(
        self, 
        config: AttrDict, 
        env_stats: AttrDict, 
        model: Model, 
        aid: int=0, 
    ):
        super().__init__(config, env_stats, model, aid)

        self.n_runners = self.config.n_runners
        self.n_envs = self.n_runners * self.config.n_envs
        self.n_steps = self.config.n_steps
        self.max_size = self.config.get('max_size', self.n_envs * self.n_steps)
        self.batch_size = self.max_size // self.n_steps if self.n_steps else self.max_size

        self.config.n_envs = self.n_envs
        self._buffer = LocalBuffer(
            self.config, 
            env_stats, 
            model, 
            0, 
            self.aid, 
        )

        self.reset()

        self._sleep_time = 0.025
        self._sample_wait_time = 0
        self.max_wait_time = self.config.get('max_wait_time', 5)

    def __contains__(self, k):
        return k in self._queue[0]
    
    def size(self):
        return self._buffer.size()

    def ready(self):
        return len(self._queue) > 0

    def reset(self):
        self._buffers = [collections.defaultdict(list) for _ in range(self.n_runners)]
        self._queue = []
        self._memory = []
        self._current_size = 0

    """ Filling Methods """
    def collect(self, env, env_step, reset, obs, next_obs, **kwargs):
        for k, v in obs.items():
            if k not in kwargs:
                kwargs[k] = v
        for k, v in next_obs.items():
            kwargs[f'next_{k}'] = v
        self.add(**kwargs, reset=reset)
    
    def add_last_value(self, value):
        self._buffer.add({'value': value})

    def add(self, **data):
        """ Add transitions """
        self._buffer.add(data)

    def _add_first_obs(self, data):
        assert len(self._buffer) == 0, self._buffer
        self._buffer['obs'] = data['obs']

    def merge_data(self, rid: int, data: dict, n: int):
        """ Merging Data from Other Buffers """
        for k, v in data.items():
            self._buffers[rid][k].append(v)
        self._current_size += n

        if self._current_size >= self.max_size:
            data = {k: np.concatenate(
                [np.concatenate(b[k]) for b in self._buffers if b]
                )[-self.batch_size:] for k in self.sample_keys
            }
            self._queue.append(data)
            self._buffers = [collections.defaultdict(list) for _ in range(self.n_runners)]
            self._current_size = 0

    def get_data(self, last_piece=None):
        _, data, _ = self._buffer.retrieve_all_data(last_piece)
        return data

    def move_to_queue(self, data):
        self._queue.append(data)

    """ Update Data """
    def update(self, key, value):
        self._queue[0][key] = value

    """ Sampling """
    def sample(self, sample_keys=None):
        ready = self._wait_to_sample()
        if not ready:
            return None
        sample = self._sample(sample_keys)
        sample = dict2AttrDict(sample, shallow=True)

        return sample

    """ Implementations """
    def _wait_to_sample(self):
        while len(self._queue) == 0 and (
                self._sample_wait_time < self.max_wait_time):
            time.sleep(self._sleep_time)
            self._sample_wait_time += self._sleep_time
        # print(f'PPOBuffer starts sampling: waiting time: {self._sample_wait_time}', self._ready)
        if self._sample_wait_time >= self.max_wait_time:
            do_logging(f'No data is received in time {self.max_wait_time}s.')
            return False
        self._sample_wait_time = 0
        return len(self._queue) != 0

    def _sample(self, sample_keys=None):
        sample_keys = sample_keys or self.sample_keys
        assert len(self._queue) == 1, self._queue
        sample = self._queue[0]
        self._queue = []

        return sample

    def clear(self):
        self.reset()
        self._queue = []


def create_buffer(config, model, env_stats, **kwargs):
    config = dict2AttrDict(config)
    env_stats = dict2AttrDict(env_stats)
    BufferCls = {
        'ac': ACBuffer, 
        'local': LocalBuffer
    }[config.type]
    return BufferCls(
        config=config, 
        env_stats=env_stats, 
        model=model, 
        **kwargs
    )
