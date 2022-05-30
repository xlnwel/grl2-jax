from typing import List, Set, Tuple, Union
import cloudpickle
import collections
import numpy as np
import ray

from .parameter_server import ParameterServer
from ..common.typing import ModelStats, ModelWeights
from core.elements.agent import Agent
from core.elements.builder import ElementsBuilder
from core.mixin.actor import RMS
from core.monitor import Monitor
from core.remote.base import RayBase
from core.typing import ModelPath
from env.func import create_env
from env.typing import EnvOutput
from utility.timer import Timer
from utility.utils import dict2AttrDict


class MultiAgentSimRunner(RayBase):
    def __init__(
        self, 
        runner_id,
        configs: Union[List[dict], dict], 
        store_data: bool, 
        evaluation: bool, 
        parameter_server: ParameterServer=None, 
        remote_buffers: List[RayBase]=None, 
        active_models: List[ModelPath]=None, 
        monitor: Monitor=None,
    ):
        super().__init__(runner_id, seed=configs[0].get('seed'))

        self.id = runner_id
        self.store_data = store_data
        self.evaluation = evaluation

        if isinstance(configs, list):
            configs = [dict2AttrDict(c) for c in configs]
            config = configs[0]
        else:
            config = dict2AttrDict(configs)
        env_config = self._setup_env_config(config.env)
        self.env = create_env(env_config, no_remote=True)
        self.env_stats = self.env.stats()
        self.n_envs = self.env_stats.n_envs
        self.n_agents = self.env_stats.n_agents
        self.n_units = self.env_stats.n_units
        self.uid2aid = self.env_stats.uid2aid
        self.aid2uids = self.env_stats.aid2uids
        self.n_units_per_agent = [len(uids) for uids in self.aid2uids]
        self.is_multi_agent = self.env_stats.is_multi_agent

        self.parameter_server = parameter_server
        self.remote_buffers: List[RayBase] = remote_buffers
        if self.remote_buffers is not None:
            assert len(self.remote_buffers) == self.n_agents, \
                (len(self.remote_buffers), self.n_agents)
        self.active_models: Set[ModelPath] = set(active_models) if active_models else set()
        self.current_models: List[ModelPath] = [None] * self.n_agents
        self.is_agent_active: List[bool] = [True] * self.n_agents
        self.monitor: Monitor = monitor

        self.n_steps = config.runner.n_steps
        self._steps = 0

        self.env_output = self.env.output()
        self.scores = [[] for _ in range(self.n_agents)]

        self.builder = ElementsBuilder(config, self.env_stats)

        self.build_from_configs(configs)

    def build_from_configs(self, configs: Union[List[dict], dict]):
        if isinstance(configs, list):
            assert len(configs) == self.n_agents, (len(configs), self.n_agents)
            configs = [dict2AttrDict(c) for c in configs]
        else:
            configs = [dict2AttrDict(configs) for _ in range(self.n_agents)]
        config = configs[0]
        self.config = config.runner

        self.agents: List[Agent] = []
        self.buffers = []
        self.rms: List[RMS] = []

        for aid, config in enumerate(configs):
            config.buffer.type = 'local'
            elements = self.builder.build_acting_agent_from_scratch(
                config, 
                env_stats=self.env_stats,
                build_monitor=True, 
                to_build_for_eval=self.evaluation, 
                to_restore=False
            )
            self.agents.append(elements.agent)

            # TODO: handle the case in which some algorithms do not require normalization with RMS
            self.rms.append(RMS(config.actor.rms))

            if self.store_data:
                # update n_steps to be consistent
                config.buffer.n_steps = self.n_steps
                buffer = self.builder.build_buffer(
                    elements.model, 
                    config=config, 
                    env_stats=self.env_stats, 
                    runner_id=self.id, 
                    n_units=self.n_units_per_agent[aid], 
                )
                self.buffers.append(buffer)
        assert len(self.agents) == len(self.rms) == self.n_agents, (
            len(self.agents), len(self.rms), self.n_agents)

    """ Running Routines """
    def random_run(self, aids=None):
        """ Random run the environment to collect running stats """
        step = 0
        agent_env_outs = self._divide_outs(self.env_output)
        self._update_rms(agent_env_outs)
        while step < self.n_steps:
            self.env_output = self.env.step(self.env.random_action())
            agent_env_outs = self._divide_outs(self.env_output)
            self._update_rms(agent_env_outs)
            step += 1

        if aids is None:
            aids = range(self.n_agents)
        for aid in aids:
            self._send_aux_stats(aid)

    def run_with_model_weights(self, mids: List[ModelWeights]):
        def set_weights(mids):
            for aid, mid in enumerate(mids):
                model_weights = ray.get(mid)
                self.is_agent_active[aid] = model_weights.model in self.active_models
                self.current_models[aid] = model_weights.model
                assert set(model_weights.weights) == set(['model', 'aux', 'train_step']), set(model_weights.weights)
                self.agents[aid].set_weights(model_weights.weights)

            assert any(self.is_agent_active), (self.active_models, self.current_models)

        def send_stats(steps, n_episodes, wt: Timer, rt: Timer):
            for aid, is_active in enumerate(self.is_agent_active):
                self.agents[aid].store(**{
                    **{f'time/{t.name}_total': t.total() for t in [wt, rt]}, 
                    **{f'time/{t.name}': t.average() for t in [wt, rt]}, 
                })
                if n_episodes > 0:
                    self._send_run_stats(aid, steps, n_episodes)
                if is_active:
                    self._send_aux_stats(aid)
            self._update_payoffs()

        with Timer('runner_set_weights') as wt:
            set_weights(mids)
        for b in self.buffers:
            assert b.is_empty(), b.size()
        with Timer('runner_run') as rt:
            steps, n_episodes = self.run()
        self._steps += steps

        send_stats(self._steps, n_episodes, wt, rt)
        if n_episodes > 0:
            self._steps = 0

        return steps

    def run(self, video: list=None, reward: list=None):
        run_func = self._run_impl_ma if self.is_multi_agent else self._run_impl_sa
        steps, n_episodes = run_func(video, reward)
        return steps, n_episodes

    def evaluate(self, total_episodes=None):
        if total_episodes is None:
            steps, n_episodes = self.run()
        else:
            n_episodes = 0
            steps = 0
            while n_episodes < total_episodes:
                step, n_eps = self.run()
                steps += step
                n_episodes += n_eps
        self._update_payoffs()

        return steps, n_episodes

    def evaluate_and_return_stats(self, total_episodes=None):
        video, rewards = [], []
        if total_episodes is None:
            steps, n_episodes = self.run(video, rewards)
        else:
            n_episodes = 0
            steps = 0
            while n_episodes < total_episodes:
                step, n_eps = self.run(video, rewards)
                steps += step
                n_episodes += n_eps

        if n_episodes > 0:
            stats = {
                'score': np.stack([
                    a.get_raw_item('score') for a in self.agents], 1),
                'dense_score': np.stack([
                    a.get_raw_item('dense_score') for a in self.agents], 1),
            }
        else:
            stats = {}

        return step, n_episodes, video, rewards, stats

    """ Running Setups """
    def set_active_models(self, model_paths: List[ModelPath]):
        assert len(model_paths) == len(self.active_models), (model_paths, self.active_models)
        self.active_models = set(model_paths)

    def set_current_models(self, model_paths: List[ModelPath]):
        assert len(model_paths) == len(self.current_models), (model_paths, self.current_models)
        self.current_models = model_paths

    def set_weights_from_configs(self, configs: List[dict], filename='params.pkl'):
        assert len(configs) == len(self.current_models) == self.n_agents, (configs, self.current_models)
        for aid, (config, agent) in enumerate(zip(configs, self.agents)):
            model = ModelPath(config['root_dir'], config['model_name'])
            self.set_weights_for_agent(agent, model, filename=filename)
            self.current_models[aid] = model

    def set_weights_from_model_paths(self, models: List[ModelPath], filename='params.pkl'):
        assert len(models) == len(self.current_models) == self.n_agents, (models, self.current_models)
        for aid, (model, agent) in enumerate(zip(models, self.agents)):
            self.set_weights_for_agent(agent, model, filename=filename)
            self.current_models[aid] = model

    def set_weights_for_agent(self, agent: Agent, model: ModelPath, filename='params.pkl'):
        path = '/'.join([model.root_dir, model.model_name, filename])
        with open(path, 'rb') as f:
            weights = cloudpickle.load(f)
            agent.set_weights(weights)

    def set_running_steps(self, n_steps):
        self.n_steps = n_steps

    """ Implementations """
    def _reset_local_buffers(self):
        if self.store_data:
            [b.reset() for b in self.buffers]

    def _reset(self):
        self.env_output = self.env.reset()
        
        self._reset_local_buffers()
        return self.env_output

    def _run_impl_ma(self, video: list=None, rewards: list=None):
        def check_end(step):
            return step >= self.n_steps

        def try_sending_data(step, terms: List[dict]):
            if self.store_data:
                sent = False
                # NOTE: currently we send all data at once
                for aid, (term, buffer) in enumerate(zip(terms, self.buffers)):
                    if self.is_agent_active[aid] and buffer.is_full():
                        rid, data, n = buffer.retrieve_all_data(last_value=term['value'])
                        self.remote_buffers[aid].merge_data.remote(rid, data, n)
                        sent = True
            else:
                sent = True
            return sent

        def agents_infer(agents: List[Agent], agent_env_outs: List[EnvOutput]):
            assert len(agent_env_outs)  == len(agents), (len(agent_env_outs), len(agents))
            # action, terms = zip(*[
            #     a(o, evaluation=self.evaluation) 
            #     for a, o in zip(agents, agent_env_outs)])
            action, terms = [], []
            for aid, (agent, o) in enumerate(zip(agents, agent_env_outs)):
                with Timer(f'{aid}/infer') as it:
                    a, t = agent(o, evaluation=self.evaluation)
                action.append(a)
                terms.append(t)
                agent.store(**{
                    'time/ind_infer_total': it.total(),
                    'time/ind_infer': it.average(),
                })
            return action, terms

        def step_env(actions: List):
            self.env_output = self.env.step(actions)
            # legacy code for visualizing overcooked.
            # if video is not None:
            #     video.append(self.env.get_screen(convert_batch=False)[0])
            if rewards is not None:
                rewards.append(self.env_output.reward[0])
            agent_env_outs = self._divide_outs(self.env_output)
            return agent_env_outs
        
        def store_data(
            agent_env_outs: List[EnvOutput], 
            agent_actions: List, 
            agent_terms: List[dict], 
            next_agent_env_outs: List[EnvOutput]
        ):
            if self.store_data:
                assert len(agent_env_outs) == len(agent_actions) \
                    == len(agent_terms) == len(next_agent_env_outs) \
                    == len(self.buffers), (
                        len(agent_env_outs), len(agent_actions), 
                        len(agent_terms), len(next_agent_env_outs),
                        len(self.buffers)
                    )
                # if self.config.get('partner_action'):
                #     agent_logits = [t.pop('logits') for t in agent_terms]
                #     agent_pactions = []
                #     agent_plogits = []
                #     for aid in range(self.n_agents):
                #         shape = (self.n_envs, sum([n for i, n in enumerate(self.n_units_per_agent) if i != aid]))
                #         plogits = [a for i, a in enumerate(agent_logits) if i != aid]
                #         plogits = np.concatenate(plogits, 1)
                #         assert plogits.shape == (*shape, self.env_stats.action_dim), \
                #             (plogits.shape, (*shape, self.env_stats.action_dim))
                #         agent_plogits.append(plogits)
                #         pactions = [a for i, a in enumerate(agent_actions) if i != aid]
                #         pactions = np.concatenate(pactions, 1)
                #         assert pactions.shape == shape, (pactions.shape, shape)
                #         agent_pactions.append(pactions)
                for aid, (agent, env_out, next_env_out, buffer) in enumerate(
                        zip(self.agents, agent_env_outs, next_agent_env_outs, self.buffers)):
                    if self.is_agent_active[aid]:
                        stats = {
                            **env_out.obs,
                            'action': agent_actions[aid],
                            'reward': agent.actor.normalize_reward(next_env_out.reward),
                            'discount': next_env_out.discount,
                        }
                        # if self.config.get('partner_action'):
                        #     stats['plogits'] = agent_plogits[aid]
                        #     stats['paction'] = agent_pactions[aid]
                        assert np.all(agent_terms[aid]['obs'] <= 5) and np.all(agent_terms[aid]['obs'] >= -5), f"{env_out.obs['life_mask']}\n{agent_terms[aid]['obs']}"
                        assert np.all(agent_terms[aid]['global_state'] <= 5) and np.all(agent_terms[aid]['global_state'] >= -5), agent_terms[aid]['global_state']
                        stats.update(agent_terms[aid])
                        buffer.add(stats)

        step = 0
        n_episodes = 0
        agent_env_outs = self._divide_outs(self.env_output)
        while True:
            with Timer('infer') as it:
                action, terms = agents_infer(self.agents, agent_env_outs)
            sent = try_sending_data(step, terms)
            if sent and check_end(step):
                break
            with Timer('step_env') as et:
                next_agent_env_outs = step_env(action)
            with Timer('update_rms') as st:
                self._update_rms(next_agent_env_outs)
            with Timer('store_data') as dt:
                store_data(agent_env_outs, action, terms, next_agent_env_outs)
            agent_env_outs = next_agent_env_outs
            with Timer('log') as lt:
                n_episodes += self._log_for_done(agent_env_outs[0].reset)
            step += 1

        for aid in range(self.n_agents):
            self.agents[aid].store(
                **{f'time/{t.name}_total': t.total() for t in [it, et, st, dt, lt]}, 
                **{f'time/{t.name}': t.average() for t in [it, et, st, dt, lt]}
            )

        return step * self.n_envs, n_episodes

    def _run_impl_sa(self, video: list=None, rewards: list=None):
        def check_end(step):
            return step >= self.n_steps

        def try_sending_data(terms):
            if self.store_data:
                sent = False
                if self.buffers[0].is_full():
                    rid, data, n = self.buffers[0].retrieve_all_data(terms['value'])
                    self.remote_buffers[0].merge_data.remote(rid, data, n)
                    sent = True
            else:
                sent = True
            return sent

        def step_env(action):
            assert action.shape == (self.n_envs,), (action.shape, (self.n_envs,))
            self.env_output = self.env.step(action)
            if video is not None:
                video.append(self.env.get_screen(convert_batch=False)[0])
            if rewards is not None:
                rewards.append(self.env_output.reward[0])
            return self.env_output
        
        def store_data(env_output, action, terms, next_env_output):
            if self.store_data:
                stats = {
                    **env_output.obs,
                    'action': action,
                    'reward': self.agents[0].actor.normalize_reward(next_env_output.reward),
                    'discount': next_env_output.discount,
                }
                stats.update(terms)
                self.buffers[0].add(stats)

        step = 0
        n_episodes = 0
        env_output = self.env_output
        while True:
            with Timer('infer') as it:
                action, terms = self.agents[0](env_output, evaluation=self.evaluation)
            sent = try_sending_data(terms)
            if sent and check_end(step):
                break
            with Timer('step_env') as et:
                self.env_output = step_env(action)
            with Timer('update_rms') as st:
                self._update_rms(env_output)
            with Timer('store_data') as dt:
                store_data(env_output, action, terms, self.env_output)
            env_output = self.env_output
            with Timer('log') as lt:
                n_episodes += self._log_for_done(self.env_output.reset)
            step += 1
        _, terms = self.agents[0](self.env_output, evaluation=self.evaluation)

        for aid in range(self.n_agents):
            self.agents[aid].store(
                **{f'time/{t.name}_total': t.total() for t in [it, et, st, dt, lt]}, 
                **{f'time/{t.name}': t.average() for t in [it, et, st, dt, lt]}
            )
        if self.store_data:
            for aid, buffer in enumerate(self.buffers):
                rid, data, n = buffer.retrieve_all_data(terms['value'])
                self.remote_buffers[aid].merge_data.remote(rid, data, n)

        return step * self.n_envs, n_episodes

    def _divide_outs(self, out: Tuple[List]):
        agent_outs = [EnvOutput(*o) for o in zip(*out)]
        assert len(agent_outs) == self.n_agents, (len(agent_outs), self.n_agents)
        # test code
        # for i in range(self.n_agents):
        #     for k, v in outs[i].obs.items():
        #         assert v.shape[:2] == (self.n_envs, self.n_units_per_agent[i]), \
        #             (k, v.shape, (self.n_envs, self.n_units_per_agent))
        #     assert outs[i].reward.shape == (self.n_envs, self.n_units_per_agent[i]), (outs[i].reward.shape, (self.n_envs, self.n_units_per_agent[i]))
        #     assert outs[i].discount.shape == (self.n_envs, self.n_units_per_agent[i])
        #     assert outs[i].reset.shape == (self.n_envs, self.n_units_per_agent[i])

        return agent_outs

    def _update_rms(self, agent_env_outs: Union[EnvOutput, List[EnvOutput]]):
        if isinstance(agent_env_outs, EnvOutput):
            self.rms[0].update_obs_rms(
                agent_env_outs.obs, 'obs', mask=out.obs.get('life_mask'))
            self.rms[0].update_obs_rms(
                agent_env_outs.obs, 'global_state', mask=out.obs.get('life_mask'))
            self.rms[0].update_reward_rms(agent_env_outs.reward, agent_env_outs.discount)
        else:
            assert len(self.rms) == len(agent_env_outs), (len(self.rms), len(agent_env_outs))
            for rms, out in zip(self.rms, agent_env_outs):
                rms.update_obs_rms(
                    out.obs, 'obs', mask=out.obs.get('life_mask'))
                rms.update_obs_rms(
                    out.obs, 'global_state')
                rms.update_reward_rms(out.reward, out.discount)

    def _log_for_done(self, reset):
        # logging when any env is reset 
        done_env_ids = [i for i, r in enumerate(reset) if np.all(r)]
        if done_env_ids:
            info = self.env.info(done_env_ids)
            stats = collections.defaultdict(list)
            for i in info:
                for k, v in i.items():
                    stats[k].append(v)
            for aid, uids in enumerate(self.aid2uids):
                self.scores[aid] += [v[uids].mean() for v in stats['score']]
                self.agents[aid].store(
                    **{
                        k: [vv[uids] for vv in v]
                        if isinstance(v[0], np.ndarray) else v
                        for k, v in stats.items()
                    }
                )

        return len(done_env_ids)

    def _send_aux_stats(self, aid):
        aux_stats = self.rms[aid].get_rms_stats()
        self.rms[aid].reset_rms_stats()
        model = self.current_models[aid]
        assert model in self.active_models, (model, self.active_models)
        model_weights = ModelWeights(model, {'aux': aux_stats})
        self.parameter_server.update_strategy_aux_stats.remote(aid, model_weights)

    def _send_run_stats(self, aid, env_steps, n_episodes):
        stats = self.agents[aid].get_raw_stats()
        stats['env_steps'] = env_steps
        stats['n_episodes'] = n_episodes
        model = self.current_models[aid]
        model_stats = ModelStats(model, stats)
        self.monitor.store_run_stats.remote(model_stats)

    def _setup_env_config(self, config: dict):
        config = dict2AttrDict(config)
        if config.get('seed') is not None:
            config.seed += self.id * 1000
        if config.env_name.startswith('unity'):
            config.unity_config.worker_id += config.n_envs * self.id + 1
        if config.env_name.startswith('grf'):
            if self.id == 0 and self.evaluation:
                config.write_video = True
                config.dump_frequency = 1
                config.write_full_episode_dumps = True
                config.render = True
            else:
                config.write_video = False
                config.write_full_episode_dumps = False
                config.render = False
        return config

    def _update_payoffs(self):
        if sum([len(s) for s in self.scores]) > 0:
            self.parameter_server.update_payoffs.remote(self.current_models, self.scores)
            self.scores = [[] for _ in range(self.n_agents)]