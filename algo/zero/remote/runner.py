import collections
import functools
from typing import List, Set
import numpy as np
from ray.util.queue import Queue

from .agent import Agent
from .parameter_server import ParameterServer
from .typing import ModelStats, ModelWeights
from ..elements.utils import collect
from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.mixin.actor import RMS
from core.monitor import Monitor
from core.remote.base import RayBase
from core.typing import ModelPath
from env.func import create_env
from env.typing import EnvOutput
from utility import pkg
from utility.timer import Timer
from utility.typing import AttrDict
from utility.utils import dict2AttrDict


class MultiAgentSimRunner(RayBase):
    def __init__(self, 
                 configs: List[dict], 
                 store_data: bool, 
                 evaluation: bool, 
                 remote_agents: List[Agent]=None, 
                 param_queues: List[Queue]=None, 
                 parameter_server: ParameterServer=None,
                 monitor: Monitor=None
                 ):
        super().__init__()

        assert len(param_queues) == len(remote_agents), (len(param_queues), len(remote_agents))

        self.store_data = store_data
        self.evaluation = evaluation
        self.remote_agents = remote_agents
        self.param_queues = param_queues
        self.n_agents = len(remote_agents)
        self.parameter_server = parameter_server
        self.monitor = monitor
        
        self.env = create_env(configs[0]['env'], no_remote=True)
        self.env_stats = self.env.stats()
        self.n_envs = self.env_stats.n_envs
        self.n_players = self.env_stats.n_players
        self.pid2aid = self.env_stats['pid2aid']
        self.env_output = self.env.output()

        self.n_players_per_agent = [0 for _ in range(self.n_agents)]
        for aid in self.pid2aid:
            self.n_players_per_agent[aid] += 1
        for i in range(1, len(self.pid2aid)):
            assert self.pid2aid[i] == self.pid2aid[i-1] \
                or self.pid2aid[i] == self.pid2aid[i-1] + 1, \
                    (self.pid2aid[i], self.pid2aid[i-1])

        self.builder = ElementsBuilder(configs[0], self.env_stats)

        self._push_every_episode = configs[0]['runner']['push_every_episode']
        
        self.build_from_configs(configs)

    def build_from_configs(self, configs):
        assert len(configs) == self.n_agents, (len(configs), self.n_agents)
        configs = [dict2AttrDict(config) for config in configs]
        config = configs[0]
        self.config = config.runner

        self.agents = []
        self.is_agent_active = [True for _ in configs]
        self.active_models: Set[ModelPath] = set()
        self.current_models: List[ModelPath] = []
        self.buffers = []
        self.collect_funcs = []
        self.rms: List[RMS] = []

        for aid, config in enumerate(configs):
            config.buffer.type = 'local'
            elements = self.builder.build_acting_agent_from_scratch(
                config, build_monitor=True, to_build_for_eval=self.evaluation)
            self.agents.append(elements.agent)
            
            model_path = ModelPath(config.root_dir, config.model_name)
            self.active_models.add(model_path)
            self.current_models.append(model_path)

            rms = AttrDict(config.actor.rms)
            rms.obs_normalized_axis = (0, 1)
            rms.reward_normalized_axis = (0, 1)
            self.rms.append(RMS(rms))

            if self.store_data:
                buffer = self.builder.build_buffer(
                    elements.model, 
                    config=config, 
                    n_players=self.n_players_per_agent[aid])
                self.buffers.append(buffer)
                self.collect_funcs.append(functools.partial(collect, buffer))
        assert len(self.agents) == len(self.active_models) \
            == len(self.rms) == len(self.buffers) == self.n_agents, \
            (len(self.agents), len(self.active_models), 
            len(self.rms), len(self.buffers), self.n_agents)

    """ Running Routines """
    def random_run(self):
        """ Random run the environment to collect running stats """
        step = 0
        agent_env_outs = self._divide_outs(self.env_output)
        self._update_rms(agent_env_outs)
        while step < self.config.N_STEPS:
            self.env_output = self.env.step(self.env.random_action())
            agent_env_outs = self._divide_outs(self.env_output)
            self._update_rms(agent_env_outs)
            self._log_for_done(self.env_output)
            step += 1

        for aid in range(self.n_agents):
            self.agents[aid].actor.update_rms_from_stats(
                self.rms[aid].get_rms_stats())
            self._send_aux_stats(aid)

        # stats = [a.get_stats() for a in self.agents]
        # print(f'Random running stats:')
        # for i, s in enumerate(stats):
        #     print(f'Random Agent {i}')
        #     print_dict(s)

    def run(self):
        def set_weights():
            for aid, pq in enumerate(self.param_queues):
                w = pq.get()
                self.is_agent_active[aid] = w.model in self.active_models
                self.current_models[aid] = w.model
                self.agents[aid].set_weights(w.weights)

        set_weights()
        self._reset_local_buffers()
        with Timer('runner_run') as rt:
            step, n_episodes = self._run_impl()

        for aid, is_active in enumerate(self.is_agent_active):
            if is_active:
                self.agents[aid].store(**{
                    'time/run': rt.total(),
                    'time/run_mean': rt.average(),
                    'time/fps': step / rt.last()
                })
                self._send_aux_stats(aid)
                self._send_run_stats(
                    aid, self.agents[aid].get_train_step(), step, n_episodes)

        return step

    """ Running Setups """
    def set_active_model_paths(self, model_paths):
        self.active_models = set(model_paths)

    """ Implementations """
    def _reset_local_buffers(self):
        if self.store_data:
            [b.reset() for b in self.buffers]

    def _reset(self):
        self.env_output = self.env.reset()
        self._reset_local_buffers()
        return self.env_output

    def _run_impl(self):
        def check_end(step):
            return step >= self.config.N_STEPS

        def agents_infer(env_outputs, agents):
            assert len(env_outputs)  == len(agents), (len(env_outputs), len(agents))
            action, terms = zip(*[
                a(o, evaluation=self.evaluation) 
                for a, o in zip(agents, env_outputs)])
            return action, terms

        def step_env(agent_actions):
            action = np.concatenate(agent_actions, 1)
            assert action.shape == (self.n_envs, self.n_agents), (action.shape, (self.n_envs, self.n_agents))
            self.env_output = self.env.step(action)
            agent_env_outs = self._divide_outs(self.env_output)
            return agent_env_outs
        
        def store_data(agent_env_outs, agent_actions, agent_terms, next_agent_env_outs):
            assert len(agent_env_outs) == len(agent_actions) \
                == len(agent_terms) == len(next_agent_env_outs) \
                == len(self.buffers), (
                    len(agent_env_outs), len(agent_actions), 
                    len(agent_terms), len(next_agent_env_outs),
                    len(self.buffers)
                )
            if self.store_data:
                for aid, (buffer, collect_fn) in enumerate(
                        zip(self.buffers, self.collect_funcs)):
                    if self.is_agent_active[aid]:
                        if self._push_every_episode:
                            for eid, reset in enumerate(agent_env_outs[aid].reset):
                                if reset:
                                    eps, epslen = buffer.retrieve_episode(eid)
                                    self.remote_agents[aid].merge_episode.remote(eps, epslen)
                        stats = {
                            **agent_env_outs[aid].obs,
                            'action': agent_actions[aid],
                            'reward': next_agent_env_outs[aid].reward,
                            'discount': next_agent_env_outs[aid].discount,
                        }
                        stats.update(agent_terms[aid])
                        collect_fn(stats)

        step = 0
        n_episodes = 0
        if self._push_every_episode:
            self._reset()
        agent_env_outs = self._divide_outs(self.env_output)
        self._update_rms(agent_env_outs)
        while not check_end(step):
            action, terms = agents_infer(agent_env_outs, self.agents)
            next_agent_env_outs = step_env(action)
            self._update_rms(next_agent_env_outs)
            store_data(agent_env_outs, action, terms, next_agent_env_outs)
            agent_env_outs = next_agent_env_outs
            n_episodes += self._log_for_done(self.env_output)
            step += 1
        _, terms = agents_infer(agent_env_outs, self.agents)

        if not self._push_every_episode:
            for aid, (term, buffer) in enumerate(zip(terms, self.buffers)):
                data, n = buffer.retrieve_all_data(term['value'])
                self.remote_agents[aid].merge_data.remote(data, n)

        return step * self.n_envs, n_episodes

    def _divide_outs(self, out):
        agent_obs = [collections.defaultdict(list) for _ in range(self.n_agents)]
        agent_reward = [[] for _ in range(self.n_agents)]
        agent_discount = [[] for _ in range(self.n_agents)]
        agent_reset = [[] for _ in range(self.n_agents)]
        for pid, aid in enumerate(self.pid2aid):
            for k, v in out.obs.items():
                agent_obs[aid][k].append(v[:, pid])
            agent_reward[aid].append(out.reward[:, pid])
            agent_discount[aid].append(out.discount[:, pid])
            agent_reset[aid].append(out.reset[:, pid])
        assert len(agent_obs) == len(agent_reward) == len(agent_discount) == len(agent_reset), \
            (len(agent_obs), len(agent_reward), len(agent_discount), len(agent_reset))
        outs = []
        for o, r, d, re in zip(agent_obs, agent_reward, agent_discount, agent_reset):
            for k, v in o.items():
                o[k] = np.stack(v, 1)
            r = np.stack(r, 1)
            d = np.stack(d, 1)
            re = np.stack(re, 1)
            outs.append(EnvOutput(o, r, d, re))
        # assert len(outs) == self.n_agents, (len(outs), self.n_agents)
        # # TODO: remove this test code
        # for i in range(self.n_agents):
        #     for k, v in outs[i].obs.items():
        #         assert v.shape[:2] == (self.n_envs, self.n_players_per_agent[i]), \
        #             (v.shape, (self.n_envs, self.n_players_per_agent))
        #     assert outs[i].reward.shape == (self.n_envs, self.n_players_per_agent[i]), (outs[i].reward.shape, (self.n_envs, self.n_players_per_agent[i]))
        #     assert outs[i].discount.shape == (self.n_envs, self.n_players_per_agent[i])
        #     assert outs[i].reset.shape == (self.n_envs, self.n_players_per_agent[i])

        # outs = tf.nest.map_structure(lambda x: np.reshape(x, (-1, *x.shape[2:])), outs)

        return outs

    def _update_rms(self, agent_env_outs):
        for rms, out in zip(self.rms, agent_env_outs):
            rms.update_obs_rms(out.obs)
            rms.update_reward_rms(out.reward, out.discount)

    def _log_for_done(self, output: EnvOutput):
        # logging when any env is reset 
        done_env_ids = [i for i, r in enumerate(output.reset) if np.all(r)]
        if done_env_ids:
            info = self.env.info(done_env_ids)
            score, epslen, dense_score = [], [], []
            for i in info:
                score.append(i['score'])
                dense_score.append(i['dense_score'])
                epslen.append(i['epslen'])
            score = np.stack(score, 1)
            dense_score = np.stack(dense_score, 1)
            epslen = np.array(epslen)
            if np.any(score):
                print(score, dense_score)
            for pid, aid in enumerate(self.pid2aid):
                if self.is_agent_active[aid]:
                    self.agents[aid].store(**{
                        f'score': score[pid], 
                        f'dense_score': dense_score[pid], 
                        f'epslen': epslen
                    })
                    
        return len(done_env_ids)

    def _send_aux_stats(self, aid):
        aux_stats = self.rms[aid].get_rms_stats()
        self.rms[aid].reset_rms_stats()
        model_weights = ModelWeights(
            self.current_models[aid], {'aux': aux_stats})
        self.parameter_server.update_strategy_aux_stats.remote(aid, model_weights)

    def _send_run_stats(self, aid, train_step, env_step, n_episodes):
        stats = self.agents[aid].get_raw_stats()
        stats['train_step'] = train_step
        stats['env_step'] = env_step
        stats['n_episodes'] = n_episodes
        model_stats = ModelStats(
            self.current_models[aid], stats)
        self.monitor.store_run_stats.remote(model_stats)
