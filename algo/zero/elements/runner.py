import functools
from time import time
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from distributed.remote.base import RayBase
from env.func import create_env
from env.utils import batch_env_output
from run.utils import search_for_config
from utility import pkg
from utility.timer import Timer
from utility.utils import batch_dicts, config_attr, dict2AttrDict
from utility.typing import AttrDict


class RunnerManager(RayBase):
    def __init__(self, config, name='zero', store_data=True, parameter_server=None):
        self.config = config_attr(self, config['runner'])
        self.self_play_frac = self.config.get('self_play_frac', 0)
        if isinstance(config, AttrDict):
            config = config.asdict()
        RemoteRunner = TwoAgentRunner.as_remote(**config['runner']['ray'])
        self.runners = [RemoteRunner.remote(
            config, name, 
            store_data=store_data, 
            parameter_server=parameter_server) 
                for _ in range(config['env']['n_workers'])]

    def initialize_rms(self):
        obs_rms_list, rew_rms_list = list(
            zip(*ray.get([r.initialize_rms.remote() for r in self.runners])))
        return obs_rms_list, rew_rms_list

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
        wid = ray.put(weights)
        steps, data, stats = list(zip(*ray.get([r.run.remote(wid) for r in self.runners])))
        stats = batch_dicts(stats, lambda x: sum(x, []))
        return sum(steps), data, stats
    
    def evaluate(self, total_episodes, weights=None, other_path=None, other_name=None):
        """ Evaluation is problematic if self.runner.run does not end in a pass """
        if other_path is not None:
            self.set_other_agent(other_path, name=other_name)
        n_eps = 0
        stats_list = []
        i = 0
        while n_eps < total_episodes:
            _, _, stats = self.run(weights)
            n_eps += len(next(iter(stats.values())))
            stats_list.append(stats)
            i += 1
        print('Total number of runs:', i)
        stats = batch_dicts(stats_list, lambda x: sum(x, []))
        return stats, n_eps


class TwoAgentRunner(RayBase):
    def __init__(self, config, name='zero', store_data=True, parameter_server=None):
        super().__init__()
        self.name = name
        self.store_data = store_data
        self.n_agents = 4
        self.other_agent = None
        self.parameter_server = parameter_server
        self.build_from_config(config)

    def build_from_config(self, config):
        self.config = dict2AttrDict(config)
        self.config.buffer.type = 'local'
        self.config.env.n_workers = 1

        self.agent_pids = self.config.runner.agent_pids
        for pid in self.agent_pids:
            assert pid not in self.config.env.skip_players, (self.agent_pids, self.config.env.skip_players)
        self.other_pids = self.compute_other_pids()
        self.env = create_env(self.config.env)
        self.env_stats = self.env.stats()
        self.env_output = self.env.output(convert_batch=False)

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

    def compute_other_pids(self):
        return [i for i in range(self.n_agents) if i not in self.agent_pids]

    def set_other_agent(self, config, self_play=False, name=None):
        if self_play:
            self.agent_pids = list(range(self.n_agents))
        else:
            self.agent_pids = self.config.runner.agent_pids
            name = name or config.algorithm
            builder = ElementsBuilder(config, self.env_stats, name=name)
            elements = builder.build_actor_agent_from_scratch()
            self.other_agent = elements.agent
        self.other_pids = self.compute_other_pids()

    def set_other_agent_from_path(self, path, self_play=False, name=None):
        config = search_for_config(path)
        self.set_other_agent(config, self_play=self_play, name=name)

    def reset(self):
        self.env_output = self.env.reset(convert_batch=False)
        self.buffer.reset()

    def run(self, weights):
        if weights is not None:
            self.agent.set_weights(weights)
        if self.parameter_server is not None:
            weights = ray.get(self.parameter_server.sample_strategy.remote(actor_weights=True))
            self.agent.set_weights(weights)
        step = self._run_impl()
        if self.store_data:
            data = self.buffer.retrieve_data()
            return step, data, self.agent.get_raw_stats()
        else:
            return step, None, self.agent.get_raw_stats()

    def _run_impl(self):
        def check_end(i):
            if self.store_data:
                return self.buffer.ready_to_retrieve()
            else:
                return i >= self.config.runner.N_STEPS

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

        def step_env(action, agent_eids, agent_outs, agent_action, agent_terms):
            if agent_outs:
                for pid in agent_outs.obs['pid']:
                    assert pid in self.agent_pids, agent_outs.obs['pid']
            else:
                assert agent_action == [], agent_action
                assert agent_terms == [], agent_terms
            self.env_output = self.env.step(action, convert_batch=False)
            done_eids, done_rewards = [], []
            assert len(self.env_output) == self.env.n_envs, (len(self.env_output), self.env.n_envs)
            for i, o in enumerate(self.env_output):
                if o.discount == 0:
                    done_eids.append(i)
                    done_rewards.append(o.reward)
            self.step += self.env.n_envs

            if self.store_data and agent_outs:
                data = {
                    **agent_outs.obs,
                    'action_type': agent_action[0],
                    'card_rank': agent_action[1],
                    'reward': [self.env_output[i].reward for i in agent_eids],
                    'discount': [self.env_output[i].discount for i in agent_eids],
                    **agent_terms
                }
                self.buffer.add(data)
            self.buffer.finish(done_eids, done_rewards)

        def log_for_done(outs):
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
            step_env(action, agent_eids, agent_outs, agent_action, agent_terms)
            log_for_done(self.env_output)
            i += 1
        return self.step
