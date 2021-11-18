import time
import collections
import numpy as np
import ray

from core.decorator import config
from core.mixin.actor import RMS
from distributed.remote.base import RayBase
from utility.utils import AttrDict2dict, batch_dicts, config_attr, to_array32
from utility import pkg
from replay.utils import *


class APGBuffer(RayBase):
    def __init__(self, config):
        rms_config = config.pop('rms')
        self.config = config_attr(self, config)
        create_buffer = pkg.import_module(
            'elements.buffer', algo=config['algorithm'], place=-1).create_buffer
        config['n_envs'] = config['n_trajs']
        self._buff = create_buffer(config)
        self._buff.reshape_to_store = lambda: None  # data is shaped in advance; no need of this function
        self.rms = RMS(rms_config)

        self._cache = []
        self._batch_idx = 0

        self._n_batch = self.config.n_trajs // self.config.n_envs   # #batch expected to received for training
        self._diag_stats = collections.defaultdict(list)
        self._n = 0

    @property
    def batch_size(self):
        return self._buff.batch_size

    def is_full(self):
        return self._batch_idx >= self._n_batch

    def merge(self, data):
        self._cache.append(data)
        self._batch_idx += 1

    def sample(self):
        if not self.is_full():
            return None

        self._prepare_for_sampling()
        sample = self._buff.sample()

        return sample

    def _prepare_for_sampling(self):        
        n, train_step = self._fill_buffer()
        self._record_async_stats(n, train_step)
        self._update_agent_rms()

        self._buff.compute_advantage_return_in_memory()
        self._buff.reshape_to_sample()

    def _fill_buffer(self):
        data = self._cache[-self._n_batch:]
        self._cache = []
        n = self._batch_idx
        self._batch_idx = 0
        data = batch_dicts(data, np.concatenate)
        train_step = data.pop('train_step')
        self._buff.replace_memory(data)

        return n, train_step

    def _record_async_stats(self, n, train_step):
        self._diag_stats['trajs_dropped'].append(
            n * self._n_envs - self._n_trajs)
        train_step_max = train_step.max()
        self._diag_stats['policy_version_max_diff'].append(
            train_step_max - train_step.min())
        self._diag_stats['policy_version_avg_diff'].append(
            train_step_max - train_step.mean())
        self._n += 1
        if self._n % 10 == 0:
            self.monitor.store_stats.remote(**self._diag_stats)
            self._n = 0
            self._diag_stats.clear()

    def _update_agent_rms(self):
        self.rms.update_obs_rms(np.concatenate(self._buff['obs']))
        self.rms.update_reward_rms(
            self._buff['reward'], self._buff['discount'])
        self._buff.update('reward', 
            self.rms.normalize_reward(self._buff['reward']), field='all')

    def get_aux_stats(self, name=None):
        rms = self.rms.get_rms_stats()
        if name is not None:
            return getattr(rms, name)
        return rms

    def get_async_stats(self):
        stats = self._diag_stats
        self._diag_stats = collections.defaultdict(list)
        return stats


class LocalBuffer:
    @config
    def __init__(self):
        self.reset()

    def size(self):
        return self._n_envs * self.N_STEPS

    def is_full(self):
        return self._idx == self.N_STEPS

    def reset(self):
        self._idx = 0
        self._memory = collections.defaultdict(list)

    def add(self, **data):
        for k, v in data.items():
            self._memory[k].append(v)
        self._idx += 1

    def sample(self):
        data = {}

        # make data batch-major
        for k, v in self._memory.items():
            v = to_array32(v)
            data[k] = np.swapaxes(v, 0, 1) if v.ndim > 1 else v

        return data

    def finish(self, last_value=None,
            last_obs=None, last_mask=None):
        """ Add last value to memory. 
        Leave advantage and return computation to the central buffer 
        """
        assert self._idx == self.N_STEPS, self._idx
        if last_value is not None:
            self._memory['last_value'] = last_value
        if last_obs is not None:
            assert last_mask is not None, 'last_mask is required'
            self._memory['obs'].append(last_obs)
            self._memory['mask'].append(last_mask)


def create_central_buffer(config):
    config = AttrDict2dict(config)
    buffer = APGBuffer.as_remote().remote(config=config)
    return buffer

def create_local_buffer(config):
    buffer = LocalBuffer(config=config)
    return buffer


if __name__ == '__main__':
    config = dict(
        algorithm='ppo',
        rms=dict(
            normalize_obs=False,
            normalize_reward=True,
            gamma=.99,
            root_dir='logs/ppo',
            model_name='baseline'
        ),
        use_dataset=True,
        adv_type='gae',
        gamma=.99,
        lam=95,
        n_trajs=19,
        n_envs=10,
        N_STEPS=256,
        N_EPOCHS=3,
        N_VALUE_EPOCHS=0,
        N_MBS=3,        # number of minibatches
        norm_adv='minibatch',
        sample_keys=[
            'obs', 'action', 'value', 'traj_ret', 'advantage', 'logpi'],
    )
    import ray
    ray.init(num_cpus=1)
    buffer = create_central_buffer(config)
    for _ in range(10):
        ray.get(buffer.merge.remote(dict(
            a=10,
            b=20
        )))
    print(ray.get(buffer.is_full.remote()))
    time.sleep(10)
    ray.shutdown()
