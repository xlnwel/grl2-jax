import collections
import functools
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.tf_config import configure_gpu, silence_tf_logs
from distributed.remote.base import RayBase
from env.cls import TwoPlayerSequentialVecEnv
from utility.run import Runner
from utility import pkg
from utility.utils import batch_dicts, config_attr, dict2AttrDict
from utility.typing import AttrDict


class TwoPlayerRunner(RayBase):
    def __init__(self, config, name='zero', store_data=True):
        silence_tf_logs()
        configure_gpu()
        self.config = dict2AttrDict(config)
        self.store_data = store_data
        self.config.buffer.type = 'local'
        self.config.env.n_workers = 1

        self.control_pids = self.config.runner.control_pids
        for pid in self.control_pids:
            assert pid not in self.config.env.skip_players, (self.control_pids, self.config.env.skip_players)
        other_pids = [i for i in range(4) if i not in self.control_pids]
        self.env = TwoPlayerSequentialVecEnv(self.config.env, other_pids, None)
        self.env_output = self.env.output()
        self.env_stats = self.env.stats()

        builder = ElementsBuilder(self.config, self.env_stats, name=name)
        elements = builder.build_actor_agent_from_scratch()
        self.agent = elements.agent
        self.step = self.agent.get_env_step()
        self.n_episodes = 0

        if self.store_data:
            self.buffer = builder.build_buffer(elements.model)
            collect_fn = pkg.import_module('elements.utils', algo=self.config.algorithm).collect
            self.collect = functools.partial(collect_fn, self.buffer)
        else:
            self.buffer = None
            self.collect = None
        
        # TODO: debug code
        self._reset_eid2pid = collections.defaultdict(list)
        self._prev_reset = self.env_output.reset

    def initialize_rms(self):
        for _ in range(10):
            self.runner.run(action_selector=self.env.random_action, step_fn=self.collect)
            self.agent.actor.update_obs_rms(np.concatenate(self.buffer['obs']))
            self.agent.actor.update_reward_rms(self.buffer['reward'], self.buffer['discount'])
            self.buffer.reset()
        self.buffer.clear()
        return self.agent.actor.get_rms_stats()

    def set_other_player(self, config):
        builder = ElementsBuilder(config, self.env_stats, name=config['algorithm'])
        elements = builder.build_actor_agent_from_scratch()
        self.env.set_other_player(elements.agent)

    def reset(self):
        self.env.reset(convert_batch=False)

    def run(self, weights):
        if weights is not None:
            self.agent.set_weights(weights)
        step = self._run_impl()
        if self.store_data:
            data = self.buffer.retrieve_data()
            return step, data, self.agent.get_raw_stats()
        else:
            return step, None, self.agent.get_raw_stats()

    def _run_impl(self):
        obs = self.env_output.obs
        for pid in obs['pid']:
            assert pid in self.control_pids, obs['pid']
        
        while not self.buffer.ready_to_retrieve():
            action, terms = self.agent(self.env_output, evaluation=False)
            obs = self._step_env(obs, action, terms)
            self._log_for_done(self.env_output.reset)
            #TODO: remove
            for m, eid, pid, r in zip(obs['mask'], obs['eid'], obs['pid'], self._prev_reset):
                if m == 0:
                    if eid not in self._reset_eid2pid:
                        assert r, (m, eid, pid, r)
                    self._reset_eid2pid[eid].append(pid)
                    if len(self._reset_eid2pid[eid]) == 4:
                        del self._reset_eid2pid[eid]
            self._prev_reset = self.env_output.reset

        return self.step

    def _step_env(self, obs, action, terms):
        self.env_output = self.env.step(action)
        self.step += self.env.n_envs
        
        next_obs, reward, discount, _ = self.env_output

        if self.store_data:
            data = {
                **obs,
                'action_type': action[0],
                'card_rank': action[1],
                'reward': reward,
                'discount': discount,
                **terms
            }
            self.buffer.add(data)

        return next_obs

    def _log_for_done(self, reset):
        # logging when any env is reset 
        done_env_ids = [i for i, r in enumerate(reset) if r]
        if done_env_ids:
            info = self.env.info(done_env_ids)
            score, epslen, won = [], [], []
            for i in info:
                score.append(i['score'])
                epslen.append(i['epslen'])
                won.append(i['won'])
            self.n_episodes += len(info)
            

class RunnerManager:
    def __init__(self, config, name='zero', store_data=True):
        self.config = config_attr(self, config['runner'])
        if isinstance(config, AttrDict):
            config = config.asdict()
        RemoteRunner = TwoPlayerRunner.as_remote(**config['runner']['ray'])
        self.runners = [RemoteRunner.remote(config, name, store_data=store_data) 
            for _ in range(config['env']['n_workers'])]

    def initialize_rms(self):
        obs_rms_list, rew_rms_list = list(
            zip(*ray.get([r.initialize_rms.remote() for r in self.runners])))
        return obs_rms_list, rew_rms_list

    def set_other_player(self, config, wait=False):
        oids = [r.set_other_player.remote(config) for r in self.runners]
        if wait:
            ray.get(oids)
        else:
            return oids

    def reset(self, wait=False):
        oids = [r.reset.remote() for r in self.runners]
        if wait:
            ray.get(oids)
        else:
            return oids

    def run(self, weights):
        wid = ray.put(weights)
        steps, data, stats = list(zip(*ray.get([r.run.remote(wid) for r in self.runners])))
        stats = batch_dicts(stats, lambda x: sum(x, []))
        return steps, data, stats
    
    def evaluate(self, total_episodes, weights=None):
        n_eps = 0
        stats_list = []
        while n_eps < total_episodes:
            _, _, stats = self.run(weights)
            n_eps += len(next(iter(stats.values())))
            stats_list.append(stats)
        stats = batch_dicts(stats_list, lambda x: sum(x, []))
        return stats, n_eps
