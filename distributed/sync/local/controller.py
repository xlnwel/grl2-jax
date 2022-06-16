from math import ceil
import logging
import cloudpickle
import time
from typing import List
import numpy as np
import ray

from .agent_manager import AgentManager
from .runner_manager import RunnerManager
from ..remote.monitor import Monitor
from ..remote.parameter_server import ParameterServer
from core.ckpt.base import YAMLCheckpointBase
from core.log import do_logging
from core.typing import ModelPath
from core.utils import save_code
from env.func import get_env_stats
from gt.alpharank import AlphaRank
from run.utils import search_for_all_configs
from utility.process import run_ray_process
from utility.timer import Every, Timer
from utility.typing import AttrDict
from utility.utils import dict2AttrDict, eval_config, modify_config
from utility import yaml_op


logger = logging.getLogger(__name__)


def _compute_max_buffer_size(config):
    return config.n_runners * config.n_envs * config.n_steps


def _check_configs_consistency(
    configs: List[AttrDict], 
    keys: List[str], 
):
    for key in keys:
        for i, c in enumerate(configs):
            for k in c[key].keys():
                if k != 'root_dir':
                    assert configs[0][key][k] == c[key][k], (
                        key, i, k, c[key][k], configs[0][key][k])


def _setup_configs(
    configs: List[AttrDict], 
    env_stats: List[AttrDict]
):
    configs = [dict2AttrDict(c, to_copy=True) for c in configs]
    max_size = _compute_max_buffer_size(configs[0].buffer)
    for aid, config in enumerate(configs):
        config.aid = aid
        config.buffer.n_envs = env_stats.n_envs
        config.buffer.max_size = max_size
        root_dir = config.root_dir
        model_name = '/'.join([config.model_name, f'a{aid}'])
        modify_config(
            config, 
            root_dir=root_dir, 
            model_name=model_name, 
            aid=aid
        )

    return configs


def _compute_running_steps(config):
    n_agents = config.n_agents
    n_runners = config.runner.n_runners
    n_steps = config.buffer.n_steps
    online_frac = config.parameter_server.get('online_frac', .2)  # the fraction of runners used for self-play
    assert n_agents <= n_runners, (n_agents, n_runners)

    worker_steps = n_runners * n_steps
    n_sp_workers = int(n_runners * online_frac)
    n_individual_workers = (n_runners - n_sp_workers) // n_agents
    assert n_sp_workers + n_individual_workers * n_agents == n_runners, \
        (n_agents, n_sp_workers, n_individual_workers, n_runners)
    n_agent_runners = n_sp_workers + n_individual_workers
    n_pbt_steps = ceil(worker_steps / n_agent_runners)
    # do_logging(worker_steps, n_sp_workers, n_agent_runners, n_pbt_steps)
    # assert False
    assert n_agent_runners * n_pbt_steps >= worker_steps, (n_agent_runners, n_pbt_steps)

    return n_agent_runners, n_pbt_steps


class Controller(YAMLCheckpointBase):
    def __init__(
        self, 
        config: AttrDict,
        name='controller',
        to_restore=True,
    ):
        self.config = eval_config(config.controller)
        model_path = ModelPath(self.config.root_dir, self.config.model_name.split('/')[0])
        self._dir = '/'.join(model_path)
        self._path = f'{self._dir}/{name}.yaml'
        save_code(model_path)

        self._iteration = 0
        self._steps = 0
        self._pids = []

        self._to_store = Every(
            self.config.store_period, time.time())

        if to_restore:
            self.restore()

    """ Manager Building """
    def build_managers_for_evaluation(self, config: AttrDict):
        self.parameter_server: ParameterServer = \
            ParameterServer.as_remote().remote(
                config=config.asdict(),
                to_restore_params=False, 
            )

        self.runner_manager: RunnerManager = RunnerManager(
            ray_config=config.ray_config.runner,
            parameter_server=self.parameter_server,
            monitor=None
        )

        self.runner_manager.build_runners(
            config,
            store_data=False,
        )

    def build_managers(
        self, 
        configs: List[AttrDict]
    ):
        _check_configs_consistency(configs, [
            'controller', 
            'parameter_server', 
            'ray_config', 
            'monitor', 
            'runner', 
            'env', 
        ])

        do_logging('Retrieving Environment Stats...', logger=logger)
        env_stats = get_env_stats(configs[0].env)
        self.suite, _ = configs[0].env.env_name.split('-', 1)
        self.configs = _setup_configs(configs, env_stats)

        do_logging('Computing Future Running Steps...', logger=logger)
        config = configs[0]
        self.n_runners = config.runner.n_runners
        if self.config.max_version_iterations > 1:
            self.n_agent_runners, self.n_pbt_steps = _compute_running_steps(config)
        else:
            self.n_agent_runners, self.n_pbt_steps = None, None

        do_logging('Building Parameter Server...', logger=logger)
        self.parameter_server: ParameterServer = \
            ParameterServer.as_remote().remote(
                config=config.asdict(),
                to_restore_params=True, 
            )
        ray.get(self.parameter_server.build.remote(
            configs=[c.asdict() for c in self.configs],
            env_stats=env_stats.asdict(),
        ))

        do_logging('Buiding Monitor...', logger=logger)
        self.monitor: Monitor = Monitor.as_remote().remote(
            config.asdict(), 
            self.parameter_server
        )

        do_logging('Building Agent Manager...', logger=logger)
        self.agent_manager: AgentManager = AgentManager(
            ray_config=config.ray_config.agent,
            env_stats=env_stats, 
            parameter_server=self.parameter_server,
            monitor=self.monitor
        )

        do_logging('Building Runner Manager...', logger=logger)
        self.runner_manager: RunnerManager = RunnerManager(
            ray_config=config.ray_config.runner,
            parameter_server=self.parameter_server,
            monitor=self.monitor
        )

    """ Training """
    def pbt_train(self):
        if isinstance(self.config.max_steps_per_iteration, (list, tuple)):
            assert len(self.config.max_steps_per_iteration) == 2, \
                self.config.max_steps_per_iteration
            
            max_steps_per_iteration = np.linspace(
                *self.config.max_steps_per_iteration, 
                num=self.config.max_version_iterations
            )
        else:
            max_steps_per_iteration = [
                self.config.max_steps_per_iteration 
                for _ in range(self.config.max_version_iterations)
            ]

        with Timer('iteration') as it:
            while self._iteration < self.config.max_version_iterations: 
                do_logging(f'Iteration {self._iteration} starts', logger=logger)
                self.initialize_actors()

                self.train(
                    self.agent_manager, 
                    self.runner_manager, 
                    max_steps_per_iteration[self._iteration]
                )

                self.cleanup()
                self._iteration += 1
                self.save()
        
        do_logging(f'Training Finished. Total Iterations: {self._iteration}', 
            logger=logger)
        self.monitor.store_stats.remote(it.to_stats())
        self._log_remote_stats(self._pids)

    def initialize_actors(self):
        if self._iteration > 0:
            for c in self.configs:
                c.runner.n_steps = self.n_pbt_steps

        model_weights, is_raw_strategy = ray.get(
            self.parameter_server.sample_training_strategies.remote())
        self.current_models = [m.model for m in model_weights]
        do_logging(f'Training Strategies at Iteration {self._iteration}: {self.current_models}', 
            logger=logger)
        self.agent_manager.build_agents(self.configs)
        self.agent_manager.set_model_weights(model_weights, wait=True)
        do_logging(f'Finish Setting Model Weights', 
            logger=logger)
        self.agent_manager.publish_weights(wait=True)
        do_logging(f'Finish Publishing Model Weights', 
            logger=logger)
        self.active_models = [model for model, _ in model_weights]
        self.runner_manager.build_runners(
            self.configs, 
            remote_buffers=self.agent_manager.get_agents(),
            active_models=self.active_models, 
        )
        do_logging(f'Finish Building Runners', 
            logger=logger)
        self._initialize_rms(self.active_models, is_raw_strategy)

    def train(
        self, 
        agent_manager: AgentManager, 
        runner_manager: RunnerManager, 
        max_steps_per_iteration
    ):
        agent_manager.start_training()
        to_restart_runners = self.config.get('restart_runners_priod', None) \
            and Every(self.config.restart_runners_priod, time.time())

        while self._steps < max_steps_per_iteration:
            model_weights = ray.get(self.parameter_server.get_strategies.remote())
            while model_weights is None:
                time.sleep(.025)
                model_weights = ray.get(self.parameter_server.get_strategies.remote())
            assert len(model_weights) == runner_manager.n_runners, (len(model_weights), runner_manager.n_runners)

            steps = sum(runner_manager.run_with_model_weights(model_weights))
            if self._iteration > 0:
                # we count the total number of steps taken by each training agent
                steps = steps // self.n_runners * self.n_agent_runners
            self._steps += steps

            curr_time = time.time()
            if self._to_store(curr_time):
                self.monitor.save_all.remote()
                self.save()
            if to_restart_runners is not None and to_restart_runners(curr_time):
                do_logging('Restarting Runners', logger=logger)
                self.runner_manager.build_runners(
                    self.configs, 
                    remote_buffers=agent_manager.get_agents(),
                    active_models=self.active_models, 
                )
            if self._pids:
                ready_pids, self._pids = ray.wait(self._pids, timeout=1e-5)
                if ready_pids:
                    self._log_remote_stats(ready_pids)

        self._post_processing()
        ray.get([
            self.monitor.save_all.remote(),
            self.monitor.save_payoff_table.remote(self._iteration+1)
        ])
        self.save()
        agent_manager.stop_training(wait=True)
        self._steps = 0

    def cleanup(self):
        do_logging(f'Cleaning up for The {self._iteration+1}-th Training Iteration...', logger=logger)
        oids = [
            self.parameter_server.archive_training_strategies.remote(),
            self.monitor.clear_iteration_stats.remote()
        ]
        self.runner_manager.destroy_runners()
        self.agent_manager.destroy_agents()
        ray.get(oids)

    def _initialize_rms(
        self, 
        models: List[ModelPath], 
        is_raw_strategy: List[bool]
    ):
        if self.config.initialize_rms and any(is_raw_strategy):
            aids = [i for i, is_raw in enumerate(is_raw_strategy) if is_raw]
            do_logging(f'Initializing RMS for Agents: {aids}', logger=logger)
            self.runner_manager.set_current_models(models)
            self.runner_manager.random_run(aids)

    def _post_processing(self):
        # return
        if self.suite == 'spiel':
            do_logging('Computing Nash Convergence...', logger=logger)
            self.compute_nash_conv()

    def _log_remote_stats(self, pids):
        stats_list = ray.get(pids)
        for stats in stats_list:
            step = stats.pop('step')
            self.monitor.store_stats.remote(
                stats, 
                step=step, 
                record=True
            )

    """ Evaluation """
    def evaluate_all(self, total_episodes, filename):
        ray.get(self.parameter_server.reset_payoffs.remote(from_scratch=False))
        self.runner_manager.evaluate_all(total_episodes)
        payoffs = self.parameter_server.get_payoffs.remote()
        counts = self.parameter_server.get_counts.remote()
        payoffs = ray.get(payoffs)
        counts = ray.get(counts)
        do_logging('Final Payoffs', logger=logger, level='pwt')
        do_logging(payoffs, logger=logger, level='pwt')
        alpha_rank = AlphaRank(1000, 5, 0)
        ranks, mass = alpha_rank.compute_rank(payoffs, return_mass=True)
        print('Alpha Rank Results:\n', ranks)
        print('Mass at Stationary Point:\n', mass)
        path = f'{self._dir}/{filename}.pkl'
        with open(path, 'wb') as f:
            cloudpickle.dump((payoffs, counts), f)
        do_logging(f'Payoffs have been saved at {path}', logger=logger)

    def compute_nash_conv(self):
        configs = search_for_all_configs(self._dir, to_attrdict=False)
        from run.spiel_eval import main
        filename = '/'.join([self._dir, 'nash_conv.txt'])
        self._pids.append(
            run_ray_process(main, configs, step=self._iteration+1, filename=filename)
        )

    """ Checkpoints """
    def save(self):
        yaml_op.dump(self._path, iteration=self._iteration, steps=self._steps)
