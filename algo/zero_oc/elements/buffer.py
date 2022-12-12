import collections
import logging
import time
import numpy as np

from core.elements.buffer import Buffer
from core.elements.model import Model
from core.log import do_logging
from core.typing import AttrDict, dict2AttrDict
from tools.display import print_dict_info
from tools.utils import batch_dicts
from .utils import collect

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
        runner_id: int,
        aid: int, 
        n_units: int,
    ):
        self.config = config
        self.runner_id = runner_id
        self.aid = aid
        self.n_units = len(env_stats.aid2uids[aid])

        self._add_attributes(env_stats, model)

    def _add_attributes(self, env_stats, model):
        self._obs_keys = env_stats.obs_keys[self.aid]
        self.state_keys, self.state_type, \
            self.sample_keys, self.sample_size = \
                extract_sampling_keys(self.config, env_stats, model)

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
        # self._train_steps = [[None for _ in range(self.n_units)] 
        #     for _ in range(self.n_envs)]

    def collect(self, env, **kwargs):
        collect(self, env, 0, **kwargs)

    def add(self, data: dict):
        self._replace_obs_with_next_obs(data)
        for k, v in data.items():
            self._buffer[k].append(v)
        self._size += 1

    def retrieve_all_data(self, latest_obs=None):
        # assert self._size == self.n_steps, (self._size, self.n_steps)
        if latest_obs is not None:
            for k, v in latest_obs.items():
                self._buffer[k].append(v)

        data = {}
        for k in self.sample_keys:
            if k not in self._buffer:
                continue
            v = np.stack(self._buffer[k], 1)
            # if self.config.timeout_done and k in self._obs_keys:
            #     assert v.shape[:3] == (self.n_envs, self.n_steps+1, self.n_units), \
            #         (k, v.shape, (self.n_envs, self.n_steps+1, self.n_units))
            # else:
            #     assert v.shape[:3] == (self.n_envs, self.n_steps, self.n_units), \
            #         (k, v.shape, (self.n_envs, self.n_steps, self.n_units))
            data[k] = v

        self.reset()
        return self.runner_id, data, self.n_envs * self.n_steps

    def _replace_obs_with_next_obs(self, data):
        if f'next_{self._obs_keys[0]}' not in data:
            return
        if self.config.timeout_done:
            # for environments that treat timeout as done, 
            # we do not separately record obs and next obs
            # instead, we record n+1 obs 
            if self._size == 0:
                for k in self._obs_keys:
                    assert data[k].shape == data[f'next_{k}'].shape, (data[k].shape, data[f'next_{k}'].shape)
                    # add the first obs when buffer is empty
                    self._buffer[k].append(data[k])
                    data[k] = data.pop(f'next_{k}')
            else:
                for k in self._obs_keys:
                    data[k] = data.pop(f'next_{k}')


class ACBuffer(Buffer):
    def __init__(
        self, 
        config: AttrDict, 
        env_stats: AttrDict, 
        model: Model, 
        aid: int=0, 
    ):
        self.config = dict2AttrDict(config)
        self.aid = aid
        self._add_attributes(env_stats, model)

    def _add_attributes(self, env_stats, model):
        self.state_keys, self.state_type, \
            self.sample_keys, self.sample_size = \
                extract_sampling_keys(self.config, env_stats, model)

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

    def __getitem__(self, k):
        return self._queue[0][k]

    def __contains__(self, k):
        return k in self._queue[0]
    
    def size(self):
        return self._buffer.size()

    def ready(self):
        return len(self._queue) > 0

    def reset(self):
        self._buffers = [collections.defaultdict(list) for _ in range(self.n_runners)]
        self._queue = collections.deque(maxlen=self.config.get('queue_size', 2))
        self._memory = []
        self._current_size = 0

    """ Filling Methods """
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

    def move_to_queue(self):
        _, data, n = self._buffer.retrieve_all_data()
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
        assert len(self._queue) == 2, len(self._queue)
        samples = []
        samples.append(self._queue.popleft())
        samples.append(self._queue.popleft())
        sample = batch_dicts(samples, func=lambda x: np.concatenate(x, 0))
        assert len(self._queue) == 0, len(self._queue)
        assert set(sample) == set(self.sample_keys), (self.sample_keys, list(sample))

        return sample

    def clear(self):
        self.reset()
        self._queue.clear()


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


def extract_sampling_keys(
    config: AttrDict, 
    env_stats: AttrDict, 
    model: Model
):
    state_keys = model.state_keys
    state_type = model.state_type
    sample_keys, sample_size = _get_sample_keys_size(config)
    sample_keys = set(sample_keys)
    obs_keys = env_stats.obs_keys[model.config.aid] if 'aid' in model.config else env_stats.obs_keys
    for k in obs_keys:
        sample_keys.add(k)
    if not config.timeout_done:
        for k in obs_keys:
            sample_keys.add(f'next_{k}')
    # if env_stats.use_action_mask:
    #     self.sample_keys.append('action_mask')
    # elif 'action_mask' in self.sample_keys:
    #     self.sample_keys.remove('action_mask')
    # if env_stats.use_life_mask:
    #     self.sample_keys.append('life_mask')
    # elif 'life_mask' in self.sample_keys:
    #     self.sample_keys.remove('life_mask')
    return state_keys, state_type, sample_keys, sample_size

def _get_sample_keys_size(config: AttrDict):
    state_keys = ['h', 'c']
    if config.get('rnn_type'): 
        sample_keys = config.sample_keys
    else:
        sample_keys = _remote_state_keys(
            config.sample_keys, 
            state_keys, 
        )
        if 'mask' in sample_keys:
            sample_keys.remove('mask')
    sample_size = config.n_steps

    return sample_keys, sample_size

def _remote_state_keys(sample_keys, state_keys):
    for k in state_keys:
        if k in sample_keys:
            sample_keys.remove(k)

    return sample_keys
