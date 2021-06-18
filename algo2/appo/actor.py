import numpy as np
import ray

from utility.utils import config_attr
from core.tf_config import *
from algo.seed.actor import \
    get_actor_class as get_actor_base_class, \
    get_learner_class as get_learner_base_class, \
    get_worker_class as get_worker_base_class, \
    get_evaluator_class
from .buffer import Buffer, LocalBuffer


def get_actor_class(AgentBase):
    ActorBase = get_actor_base_class(AgentBase)
    class Actor(ActorBase):
        def _process_output(self, obs, kwargs, out, evaluation):
            out = super()._process_output(obs, kwargs, out, evaluation)
            out[1]['train_step'] = np.ones(obs.shape[0]) * self.train_step
            return out

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
                    self.train_step,
                    self.get_weights(name=self._push_names))

        def _sample_data(self):
            data = self.dataset.sample(self.train_step)
            return data

        def _store_buffer_stats(self):
            super()._store_buffer_stats()
            self.store(**self.dataset.get_async_stats())
            # reset dataset for the next training iteration
            self.dataset.reset()

    return Learner


def get_worker_class():
    """ A Worker is only responsible for resetting&stepping environment """
    WorkerBase = get_worker_base_class()
    class Worker(WorkerBase):
        def env_step(self, eid, action, terms):
            # TODO: consider using a queue here
            env_output = self._envvecs[eid].step(action)
            kwargs = dict(
                obs=self._obs[eid], 
                action=action, 
                reward=env_output.reward,
                discount=env_output.discount, 
                next_obs=env_output.obs
            )
            kwargs.update(terms)
            self._obs[eid] = env_output.obs

            if self._buffs[eid].is_full():
                self._buffs[eid].finish(terms['value'])
                self._send_data(self._replay, self._buffs[eid])

            self._collect(
                self._buffs[eid], self._envvecs[eid], env_step=None,
                reset=env_output.reset, **kwargs)

            done_env_ids = [i for i, r in enumerate(env_output.reset) if r]
            if np.any(done_env_ids):
                self._info['score'] += self._envvecs[eid].score(done_env_ids)
                self._info['epslen'] += self._envvecs[eid].epslen(done_env_ids)
                if len(self._info['score']) > 10:
                    self._send_episodic_info(self._monitor)

            return env_output

        def _create_buffer(self, buffer_config, n_envvecs):
            buffer_config['force_envvec'] = True
            return {eid: LocalBuffer(buffer_config) 
                for eid in range(n_envvecs)}

    return Worker
