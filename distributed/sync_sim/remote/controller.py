from typing import List
from ray.util.queue import Queue

from .agent_manager import AgentManager
from .parameter_server import ParameterServer
from .runner_manager import RunnerManager
from .monitor import Monitor
from env.func import get_env_stats
from utility.typing import AttrDict
from utility.utils import modify_config


class Controller:
    def __init__(self, configs: List[AttrDict]):
        self._check_configs_consistency(configs, [
            'controller', 
            'env', 
            'monitor', 
            'runner', 
            'ray_config'
        ])

        self.config = configs[0].controller

        self._build_managers(configs)

    def _build_managers(self, configs: List[AttrDict]):
        queues = self._build_queues(configs)
        self.parameter_server: ParameterServer = \
            ParameterServer.as_remote().remote(
                configs=[c.asdict() for c in configs],
                param_queues=queues.param)
        monitor: Monitor = Monitor.as_remote().remote(
            configs[0].monitor.asdict(), 
            self.parameter_server)

        env_stats = get_env_stats(configs[0].env)
        configs = self._setup_configs(configs, env_stats)

        self.agent_manager: AgentManager = AgentManager(
            ray_config=configs[0].ray_config.agent,
            env_stats=env_stats, 
            parameter_server=self.parameter_server,
            monitor=monitor)
        self.agent_manager.build_agents(configs)

        self.runner_manager: RunnerManager = RunnerManager(
            configs[0].runner,
            ray_config=configs[0].ray_config.runner,
            remote_agents=self.agent_manager.get_agents(), 
            param_queues=queues.param,
            parameter_server=self.parameter_server,
            monitor=monitor)
        self.runner_manager.build_runners(self.agent_manager.get_config())

    def pbt_train(self):
        iteration = 0
        while iteration < self.config.MAX_VERSION_ITERATIONS:
            model_paths = self.agent_manager.get_model_paths()
            self.agent_manager.publish_strategies(wait=True)
            self.runner_manager.set_active_model_paths(model_paths, wait=True)
            if self.config.initialize_rms:
                self.runner_manager.random_run()
            self.train(
                self.agent_manager, 
                self.runner_manager, 
            )
            self.agent_manager.increase_version()

    def train(
            self, 
            agent_manager: AgentManager, 
            runner_manager: RunnerManager, 
        ):
        iteration = 0
        agent_manager.start_training()

        while iteration < self.config.MAX_TRAINING_ITERATION:
            runner_manager.run()
            iteration += 1

    def _setup_configs(
            self, 
            configs: List[AttrDict], 
            env_stats: List[AttrDict]
        ):
        for aid, config in enumerate(configs):
            config.aid = aid
            config.buffer.n_envs = env_stats.n_workers * env_stats.n_envs
            root_dir = config.root_dir
            model_name = '/'.join([config.model_name, f'aid-{aid}'])
            modify_config(
                config, 
                root_dir=root_dir, 
                model_name=model_name, 
                aid=aid)

        return configs

    def _build_queues(
            self, 
            configs: List[AttrDict]
        ):
        config = configs[0]
        queues = AttrDict()
        queues.param = [
            [Queue() for _ in range(config.runner.n_workers)]
            for _ in configs
        ]
        return queues

    def _check_configs_consistency(
            self, 
            configs: List[AttrDict], 
            keys: List
        ):
        for key in keys:
            for i, c in enumerate(configs):
                assert configs[0][key] == c[key], (key, i, c[key], configs[0][key])
