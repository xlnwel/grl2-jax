import collections
import psutil
import numpy as np

from core.mixin import IdentifierConstructor
from core.mixin.actor import RMS
from core.tf_config import *
from core.remote.base import RayBase
from env.func import create_env
from utility import pkg
from utility.utils import AttrDict2dict, dict2AttrDict


class RemoteWorker(RayBase):
    def __init__(self, config, name=None):
        self.config = dict2AttrDict(config)
        self._name = name
        self._idc = IdentifierConstructor()

        psutil.Process().nice(config.get('default_nice', 0)+10)
        self.rms = RMS(self.config.actor.rms)

        self._n_envs = self.config.env.n_envs
        self._n_vecenvs = None
        self._vecenvs = None
        self._env_step_counters = {}
        self._env_outputs = {}
        self._info = {}
        self._collect_funcs = {}

        self.central_buffers = {}
        self.local_buffers = {}

    def construct_worker_from_config(self, config):
        def create_envvecs(env_config):
            n_vecenvs = env_config.pop('n_vecenvs')
            env_config.pop('n_workers', None)
            envvecs = [create_env(env_config, force_envvec=True) 
                for _ in range(n_vecenvs)]
            return n_vecenvs, envvecs

        self.config = config = dict2AttrDict(config)
        self._n_vecenvs, self._vecenvs = create_envvecs(config.env)
        self._env_step_counters = {f'env_step_{i}': 0 for i in range(self._n_vecenvs)}
        self._env_outputs = [e.output() for e in enumerate(self._vecenvs)]
        self._info = collections.defaultdict(list)
        for aid in sorted(self.local_buffers):
            identifier, _ = self.actors[aid]
            algo = self.configs[identifier].algorithm
            collect = pkg.import_module(
                'elements.utils', algo=algo, place=-1).collect
            self._collect_funcs[aid] = collect

    def env_step(self, eid, action, terms):
        self._env_step_counters[f'env_step_{eid}'] += self._n_envs

        env_output = self._vecenvs[eid].step(action)
        kwargs = dict(
            **self._env_outputs[eid].obs, 
            action=action, 
            reward=env_output.reward,
            discount=env_output.discount, 
            reset=self._env_outputs[eid].reset,
            next_obs=None)
        kwargs.update(terms)
        self._env_outputs[eid] = env_output

        if self._buffs[eid].is_full():
            # Adds the last value/obs to buffer for gae computation. 
            if self._buffs[eid]._adv_type == 'vtrace':
                self._buffs[eid].finish(
                    last_obs=env_output.obs, 
                    last_mask=1-env_output.reset)
            else:
                self._buffs[eid].finish(last_value=terms['value'])
            self._send_data(self._replay, self._buffs[eid])

        self._collect(
            self._buffs[eid], self._vecenvs[eid], env_step=None, **kwargs)

        done_env_ids = [i for i, r in enumerate(env_output.reset) if np.all(r)]
        if done_env_ids:
            self._info['score'] += self._vecenvs[eid].score(done_env_ids)
            self._info['epslen'] += self._vecenvs[eid].epslen(done_env_ids)
            if len(self._info['score']) > 10:
                self.monitor.store_stats.remote(**self._info)

        return env_output

    def random_warmup(self, steps):
        rewards = []
        discounts = []

        for e in self._vecenvs:
            for _ in range(steps // e.n_envs):
                env_output = e.step(e.random_action())
                if e.is_multiagent:
                    env_output = tf.nest.map_structure(np.concatenate, env_output)
                    life_mask = env_output.obs.get('life_mask')
                else:
                    life_mask = None
                self.process_obs_with_rms(env_output.obs, mask=life_mask)
                rewards.append(env_output.reward)
                discounts.append(env_output.discount)

        rewards = np.swapaxes(rewards, 0, 1)
        discounts = np.swapaxes(discounts, 0, 1)
        self.update_reward_rms(rewards, discounts)

        return self.get_rms_stats()

    def _send_episodic_info(self, monitor):
        """ Sends episodic info to monitor for bookkeeping """
        if self._info:
            monitor.record_episodic_info.remote(
                self._id, **self._info, **self._env_step_counters)
            self._info.clear()

    def register_buffer(self, aid, central_buffer):
        algo = self.config.algorithm.split('-')[0]
        buffer_constructor = pkg.import_module(
            'buffer', pkg=f'distributed.{algo}').create_local_buffer
        self.local_buffers[aid] = {eid: buffer_constructor(self.config.buffer) 
            for eid in range(self._n_vecenvs)}
        self.central_buffers[aid] = central_buffer
