import time
import queue
import threading
import numpy as np
import ray

from utility.utils import config_attr
from utility.display import pwc
from core.tf_config import *
from core.mixin import RMS
from algo.seed.actor import \
    get_actor_class as get_actor_base_class, \
    get_learner_class as get_learner_base_class, \
    get_worker_class as get_worker_base_class, \
    get_evaluator_class
from .buffer import Buffer, LocalBuffer


def get_actor_class(AgentBase):
    ActorBase = get_actor_base_class(AgentBase)
    class Actor(ActorBase):
        def __init__(self, actor_id, model_fn, config, model_config, env_config):
            super().__init__(actor_id, model_fn, config, model_config, env_config)
            
            # self._event = threading.Event()
            # self._event.set()

        def _process_output(self, obs, kwargs, out, evaluation):
            out = super()._process_output(obs, kwargs, out, evaluation)
            out[1]['train_step'] = np.ones(obs.shape[0]) * self.train_step
            return out

        def start(self, workers, learner, monitor):
            super().start(workers, learner, monitor)
            self._workers = workers
        #     self.resume(self.train_step)

        # def stop(self):
        #     self._event.clear()

        # def resume(self, policy_version):
        #     assert policy_version == self.train_step, \
        #         (policy_version, self.train_step)
        #     pwc(f'{self.name} resumes')
        #     for w in self._workers:
        #         ray.get(w.clear_buffer.remote())
        #     self._event.set()

    return Actor


def get_learner_class(AgentBase):
    LearnerBase = get_learner_base_class(AgentBase)
    class Learner(LearnerBase):
        def _add_attributes(self, env, dataset):
            super()._add_attributes(env, dataset)

            if not hasattr(self, '_push_names'):
                self._push_names = [
                    k for k in self.model.keys() if 'target' not in k]

        def _create_dataset(self, replay, model, env, config, replay_config):
            self.replay = Buffer(replay_config)
            return self.replay

        def set_handler(self, **kwargs):
            config_attr(self, kwargs)

        def push_weights(self):
            for a in self._actors:
                a.set_weights.remote(
                    *self.get_weights(name=self._push_names))

        def _learning(self):
            while True:
                self.dataset.wait_to_sample(self.train_step)

                self.update_obs_rms(np.concatenate(self.dataset['obs']))
                self.update_reward_rms(
                    self.dataset['reward'], self.dataset['discount'])
                self.dataset.reshape_to_sample()

                self.learn_log()

                obs_rms, _ = self.get_rms_stats()
                for a in self._actors:
                    ray.get(a.set_weights.remote(
                        *self.get_weights(name=self._push_names)))
                    ray.get(a.set_rms_stats.remote(obs_rms))
                    # ray.get(a.resume.remote(self.train_step))

        def _store_buffer_stats(self):
            super()._store_buffer_stats()
            self.store(**self.dataset.get_async_stats())
            # reset dataset for the next training iteration
            self.dataset.reset()

    return Learner


def get_worker_class():
    """ A Worker is only responsible for resetting&stepping environment """
    WorkerBase = get_worker_base_class()
    class Worker(WorkerBase, RMS):
        def __init__(self, worker_id, config, env_config, buffer_config):
            super().__init__(worker_id, config, env_config, buffer_config)

            self._setup_rms_stats()
            self._n_sends = 0
            pwc(f'Initial #sends: {self._n_sends}')
            # self._event = threading.Event()
            # self._event.set()

        def env_step(self, eid, action, terms):
            # TODO: consider using a queue here
            env_output = self._envvecs[eid].step(action)
            kwargs = dict(
                obs=self._obs[eid], 
                action=action, 
                reward=env_output.reward,
                discount=env_output.discount, 
                next_obs=env_output.obs)
            kwargs.update(terms)
            self._obs[eid] = env_output.obs

            if self._buffs[eid].is_full():
                # Adds the last value to buffer for gae computation. 
                self._buffs[eid].finish(terms['value'])
                self._send_data(self._replay, self._buffs[eid])

            self._collect(
                self._buffs[eid], self._envvecs[eid], env_step=None,
                reset=env_output.reset, **kwargs)

            done_env_ids = [i for i, r in enumerate(env_output.reset) if r]
            if done_env_ids:
                self._info['score'] += self._envvecs[eid].score(done_env_ids)
                self._info['epslen'] += self._envvecs[eid].epslen(done_env_ids)
                if len(self._info['score']) > 10:
                    self._send_episodic_info(self._monitor)

            return env_output

        def random_warmup(self, steps):
            rewards = []
            discounts = []

            for e in self._envvecs:
                for _ in range(steps // e.n_envs):
                    o, r, d, _ = e.step(e.random_action())
                    self._process_obs(o)
                    rewards.append(r)
                    discounts.append(d)

            rewards = np.swapaxes(rewards, 0, 1)
            discounts = np.swapaxes(discounts, 0, 1)
            self.update_reward_rms(rewards, discounts)

            return self.get_rms_stats()

        def _create_buffer(self, buffer_config, n_envvecs):
            buffer_config['force_envvec'] = True
            return {eid: LocalBuffer(buffer_config) 
                for eid in range(n_envvecs)}

        def _send_data(self, replay, buffer):
            super()._send_data(replay, buffer)
            self._n_sends += 1
            # pwc(f'{self.name} #sends: {self._n_sends}')
            # if self._n_sends == self._n_trajs // self._n_workers // buffer._n_envs:
            #     pwc(f'{self.name} stops at {self._n_sends}')

        # def clear_buffer(self):
        #     pwc(f'{self.name} clears buffer')
        #     for b in self._buffs.values():
        #         if b._idx > 0:
        #             pwc(f'{self.name} before clear: {b._idx}')
        #             b.reset()
        #     self._n_sends = 0

    return Worker
