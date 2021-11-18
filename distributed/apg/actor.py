import time
import numpy as np
import ray

from distributed.remote.actor import RemoteActorBase
from env.func import create_env
from utility import pkg
from utility.display import pwc
from utility.timer import Timer
from utility.utils import AttrDict2dict


class RemoteActor(RemoteActorBase):
    def _sync_sim_loop(self):
        pwc(f'{self._name} synchronous loop', color='green')

        collect_funcs = {}
        for aid in sorted(self.local_buffers):
            identifier, _ = self.actors[aid]
            algo = self.configs[identifier].algorithm
            collect = pkg.import_module(
                'elements.utils', algo=algo, place=-1).collect
            collect_funcs[aid] = collect

        def single_agent_loop():
            env = create_env(self.config.env)
            env_output = env.output()

            aid = list(self.local_buffers.keys())[0]
            buffer, collect = self.local_buffers[aid], collect_funcs[aid]
            actor = self.actors[aid].actor

            q_sizes = []
            start_env_time = time.time()
            env_steps = buffer.size()
            scores = []
            epslens = []
            while True:
                if self._stop_act_loop:
                    break
                q_sizes = self.fetch_weights(aid, q_sizes)
                inp = env_output.obs
                action, terms, state = actor(inp, evaluation=False)
                next_env_output = env.step(action)

                _, reward, discount, reset = next_env_output

                kwargs = dict(
                    **env_output.obs, 
                    action=action, 
                    reward=reward, 
                    discount=discount, 
                    reset=env_output.reset, 
                    next_obs=None,
                    train_step=self._train_step)
                kwargs.update(terms)
                collect(buffer, env, self._env_step, **kwargs)

                if np.any(reset):
                    done_env_ids = [i for i, r in enumerate(reset)
                        if (np.all(r) if isinstance(r, np.ndarray) else r)]
                    scores += env.score(done_env_ids)
                    epslens += env.epslen(done_env_ids)
                if buffer.is_full():
                    # Adds the last value/obs to buffer for gae computation. 
                    if buffer._adv_type == 'vtrace':
                        buffer.finish(
                            last_obs=env_output.obs, 
                            last_mask=1-env_output.reset)
                    else:
                        buffer.finish(last_value=terms['value'])
                    self._send_data(aid)
                    buffer.reset()
                    env_time = time.time()
                    self.monitor.store_stats.remote(
                        env_steps=env_steps,
                        fps=env_steps / (env_time - start_env_time),
                        score=scores,
                        epslen=epslens
                    )

                env_output = next_env_output
                self._env_step += env.n_envs

        if len(self.actors) == 1:
            single_agent_loop()
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
            env_output = EnvOutput(*[
                batch_dicts(x, np.concatenate)
                if isinstance(x[0], dict) else np.concatenate(x, 0)
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
    ray_config = config.coordinator.actor_ray
    config = AttrDict2dict(config)
    env_stats = AttrDict2dict(env_stats)
    return RemoteActor.as_remote(**ray_config
        ).remote(config, env_stats, name=name)
