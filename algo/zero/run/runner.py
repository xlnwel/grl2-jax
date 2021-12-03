import functools
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.tf_config import configure_gpu, silence_tf_logs
from distributed.remote.base import RayBase
from env.func import create_env
from env.utils import batch_env_output
from run.utils import search_for_config
from utility import pkg
from utility.utils import batch_dicts, config_attr, dict2AttrDict
from utility.typing import AttrDict


class TwoPlayerRunner(RayBase):
    def __init__(self, config, name='zero', store_data=True):
        silence_tf_logs()
        configure_gpu()
        self.name = name
        self.store_data = store_data
        self.n_agents = 4
        self.other_agent = None
        self.build_from_config(config)

    def build_from_config(self, config):
        self.config = dict2AttrDict(config)
        self.config.buffer.type = 'local'
        self.config.env.n_workers = 1

        self.agent_pids = self.config.runner.agent_pids
        for pid in self.agent_pids:
            assert pid not in self.config.env.skip_players, (self.agent_pids, self.config.env.skip_players)
        self.other_pids = [i for i in range(self.n_agents) if i not in self.agent_pids]
        self.env = create_env(self.config.env)
        self.env_stats = self.env.stats()

        builder = ElementsBuilder(self.config, self.env_stats, name=self.name)
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

    def initialize_rms(self):
        for _ in range(10):
            self.runner.run(action_selector=self.env.random_action, step_fn=self.collect)
            self.agent.actor.update_obs_rms(np.concatenate(self.buffer['obs']))
            self.agent.actor.update_reward_rms(self.buffer['reward'], self.buffer['discount'])
            self.buffer.reset()
        self.buffer.clear()
        return self.agent.actor.get_rms_stats()

    def set_other_agent(self, config, name=None):
        name = name or config.algorithm
        builder = ElementsBuilder(config, self.env_stats, name=name)
        elements = builder.build_actor_agent_from_scratch()
        self.other_agent = elements.agent

    def set_other_agent_from_path(self, path, name=None):
        config = search_for_config(path)
        self.set_other_agent(config, name=name)

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
        def divide_outs(outs):
            agent_eids = []
            agent_outs = []
            other_eids = []
            other_outs = []
            for i, o in enumerate(outs):
                if o.obs['pid'] in self.agent_pids:
                    agent_eids.append(i)
                    agent_outs.append(o)
                else:
                    assert o.obs['pid'] in self.other_pids, (o.obs['pid'], self.other_pids)
                    other_eids.append(i)
                    other_outs.append(o)
            if agent_outs:
                agent_outs = batch_env_output(agent_outs)
            if other_outs:
                other_outs = batch_env_output(other_outs)
            assert len(agent_eids) + len(other_eids) == self.n_agents, (agent_eids, other_eids)
            return agent_eids, agent_outs, other_eids, other_outs
        
        def merge(agent_eids, agent_action, other_eids, other_action):
            agent_action = list(zip(*agent_action))
            other_action = list(zip(*other_action))
            assert len(agent_eids) == len(agent_action), (agent_eids, agent_action)
            assert len(other_eids) == len(other_action), (other_eids, other_action)
            i = 0
            j = 0
            action = []
            while i != len(agent_eids) and j != len(other_eids):
                if agent_eids[i] < other_eids[j]:
                    action.append(agent_action[i])
                    i += 1
                else:
                    action.append(other_action[j])
                    j += 1
            while i != len(agent_eids):
                action.append(agent_action[i])
                i += 1
            while j != len(other_eids):
                action.append(other_action[j])
                j += 1
            return tuple(map(np.stack, zip(*action)))

        def check_end(i):
            if self.store_data:
                return self.buffer.ready_to_retrieve()
            else:
                return i >= self.config.runner.N_STEPS

        # reset as we store full trajectories each time
        self.env_output = self.env.reset(convert_batch=False)
        
        i = 0
        while not check_end(i):
            agent_eids, agent_outs, other_eids, other_outs = \
                divide_outs(self.env_output)
            agent_action, agent_terms = self.agent(agent_outs, evaluation=False) \
                if agent_outs else ([], [])
            other_action, other_terms = self.other_agent(other_outs, evaluation=False) \
                if other_outs else ([], [])
            action = merge(
                agent_eids, agent_action, other_eids, other_action)
            self._step_env(action, agent_eids, agent_outs, agent_action, agent_terms)
            self._log_for_done(self.env_output)
            i += 1

        return self.step

    def _step_env(self, action, agent_eids, agent_outs, agent_action, agent_terms):
        if agent_outs:
            for pid in agent_outs.obs['pid']:
                assert pid in self.agent_pids, agent_outs.obs['pid']
        else:
            assert agent_action == [], agent_action
            assert agent_terms == [], agent_terms
        self.env_output = self.env.step(action, convert_batch=False)
        self.step += self.env.n_envs
        
        if self.store_data and agent_outs:
            print('discount', [o.discount for o in self.env_output])
            data = {
                **agent_outs.obs,
                'action_type': agent_action[0],
                'card_rank': agent_action[1],
                'reward': [self.env_output[i].reward for i in agent_eids],
                'discount': [self.env_output[i].discount for i in agent_eids],
                **agent_terms
            }
            self.buffer.add(data)

    def _log_for_done(self, outs):
        # logging when any env is reset 
        done_env_ids = [i for i, o in enumerate(outs) if o.reset]
        if done_env_ids:
            info = self.env.info(done_env_ids)
            score, epslen, won = [], [], []
            for i in info:
                score.append(i['score'])
                epslen.append(i['epslen'])
                won.append(i['won'])
            self.agent.store(score=score, epslen=epslen, win_rate=won)
            self.n_episodes += len(info)


class RunnerManager(RayBase):
    def __init__(self, config, name='zero', store_data=True, parameter_server=None):
        self.config = config_attr(self, config['runner'])
        if isinstance(config, AttrDict):
            config = config.asdict()
        RemoteRunner = TwoPlayerRunner.as_remote(**config['runner']['ray'])
        self.runners = [RemoteRunner.remote(config, name, store_data=store_data) 
            for _ in range(config['env']['n_workers'])]
        self.parameter_server = parameter_server

    def initialize_rms(self):
        obs_rms_list, rew_rms_list = list(
            zip(*ray.get([r.initialize_rms.remote() for r in self.runners])))
        return obs_rms_list, rew_rms_list

    def set_other_agent_from_server(self, parameter_server, name=None, wait=False):
        paths = parameter_server.sample_strategy.remote(len(self.runners))
        oids = [r.set_other_agent_from_path.remote(p, name=name) 
            for r, p in zip(self.runners, paths)]
        if wait:
            ray.get(oids)
        else:
            return oids

    def set_other_agent(self, path, name=None, wait=False):
        oids = [r.set_other_agent_from_path.remote(path, name=name) for r in self.runners]
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
        if self.parameter_server is not None:
            self.set_other_agent_from_server(self.parameter_server)
        wid = ray.put(weights)
        steps, data, stats = list(zip(*ray.get([r.run.remote(wid) for r in self.runners])))
        stats = batch_dicts(stats, lambda x: sum(x, []))
        return steps, data, stats
    
    def evaluate(self, total_episodes, weights=None, other_path=None, other_name=None):
        """ Evaluation is problematic if self.runner.run does not end in a pass """
        if other_path is not None:
            self.set_other_agent(other_path, name=other_name)
        n_eps = 0
        stats_list = []
        while n_eps < total_episodes:
            _, _, stats = self.run(weights)
            n_eps += len(next(iter(stats.values())))
            stats_list.append(stats)
        stats = batch_dicts(stats_list, lambda x: sum(x, []))
        return stats, n_eps
