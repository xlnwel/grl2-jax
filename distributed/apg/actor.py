import functools
import numpy as np
import ray

from env.func import create_env
from utility import pkg
from utility.display import pwc
from distributed.remote.actor import RemoteActorBase
from distributed.apg.buffer import LocalBuffer


class RemoteActor(RemoteActorBase):
    def register_buffer(self, aid, central_buffer):
        if not hasattr(self, 'buffers'):
            self.buffers = {}
        self.buffers[aid] = LocalBuffer(self.config.buffer)
        self.central_buffers[aid] = central_buffer

    def _sync_sim_loop(self):
        pwc(f'{self._name} synchronous loop', color='green')

        buffer_collect = []
        for aid in sorted(self.actors):
            identifier, actor = self.actors[aid]
            if aid in self.buffers:
                algo = self.configs[identifier].algorithm
                collect_fn = pkg.import_module('elements.utils', algo=algo).collect
                collect = functools.partial(collect_fn, self.buffers[aid])
                buffer_collect.append(self.buffers[aid], collect)

        env = create_env(self.config.env)
        env_output = env.output()

        if len(self.actors) == 1:
            buffer, collect = buffer_collect[0]
            
            q_sizes = []
            while True:
                if self._stop_act_loop:
                    break
                q_sizes = self.fetch_weights(q_sizes)
                action, terms = self.actor(env_output)
                next_env_output = env.step(action)

                next_obs, reward, discount, reset = next_env_output

                kwargs = dict(
                    obs=env_output.obs, 
                    action=action, 
                    reward=reward, 
                    discount=discount, 
                    next_obs=next_obs)
                kwargs.update(terms)

                buffer.add(**kwargs)

                if buffer.is_full():
                    # Adds the last value/obs to buffer for gae computation. 
                    if buffer._adv_type == 'vtrace':
                        buffer.finish(
                            last_obs=env_output.obs, 
                            last_mask=1-env_output.reset)
                    else:
                        buffer.finish(last_value=terms['value'])

                env_output = next_env_output
                self._env_steps += env.n_envs
        else:
            raise NotImplementedError

    def _async_sim_loop(self):
        pwc(f'{self._name} asynchronous loop', color='green')
        
        # retrieve the last env_output
        objs = {self.workers[wid].env_output.remote(eid): (wid, eid)
            for wid in range(self._wpa) 
            for eid in range(self._n_vecenvs)}

        self.env_step = 0
        q_size = []
        while True:
            q_size, fw = self._fetch_weights(q_size)

            # retrieve ready objs
            with Timer(f'{self.name} wait') as wt:
                ready_objs, _ = ray.wait(
                    list(objs), num_returns=self._action_batch)
            assert self._action_batch == len(ready_objs), \
                (self._action_batch, len(ready_objs))

            # prepare data
            wids, eids = zip(*[objs.pop(i) for i in ready_objs])
            assert len(wids) == len(eids) == self._action_batch, \
                (len(wids), len(eids), self._action_batch)
            env_output = list(zip(*ray.get(ready_objs)))
            assert len(env_output) == 4, env_output
            if isinstance(env_output[0][0], dict):
                # if obs is a dict
                env_output = EnvOutput(*[
                    batch_dicts(x, np.concatenate)
                    if isinstance(x[0], dict) else np.concatenate(x, 0)
                    for x in env_output])
            else:
                env_output = EnvOutput(*[
                    np.concatenate(x, axis=0) 
                    for x in env_output])
            # do inference
            with Timer(f'{self.name} call') as ct:
                actions, terms = self(wids, eids, env_output)

            # distribute action and terms
            actions = np.split(actions, self._action_batch)
            terms = [list(itertools.product([k], np.split(v, self._action_batch))) 
                for k, v in terms.items()]
            terms = [dict(v) for v in zip(*terms)]

            # step environments
            objs.update({
                workers[wid].env_step.remote(eid, a, t): (wid, eid)
                for wid, eid, a, t in zip(wids, eids, actions, terms)})

            self.env_step += self._action_batch * self._n_envs

            if self._to_sync(self.env_step):
                monitor.record_run_stats.remote(
                    worker_name=self._id,
                    **{
                    'time/wait_env': wt.average(),
                    'time/agent_call': ct.average(),
                    'time/fetch_weights': fw.average(),
                    'n_ready': self._action_batch,
                    'param_queue_size': np.mean(q_size)
                })
                q_size = []


def create_remote_actor(config, env_stats, name):
    return RemoteActor.as_remote(**config.trainer_manager.ray)(
        config, env_stats, name=name)
