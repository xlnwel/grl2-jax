from typing import List
import ray
from ray.util.queue import Queue

from .agent_manager import AgentManager
from .parameter_server import ParameterServer
from .runner_manager import RunnerManager
from .monitor import Monitor
from core.typing import ModelPath
from core.utils import save_code
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
        model_path = ModelPath(self.config.root_dir, self.config.model_name)
        save_code(model_path)

    def _build_managers(self, configs: List[AttrDict]):
        queues = self._build_queues(configs)
        env_stats = get_env_stats(configs[0].env)
        self.configs = configs = self._setup_configs(configs, env_stats)

        self.parameter_server: ParameterServer = \
            ParameterServer.as_remote().remote(
                configs=[c.asdict() for c in self.configs],
                param_queues=queues.param
            )
        monitor: Monitor = Monitor.as_remote().remote(
            self.configs[0].monitor.asdict(), 
            self.parameter_server
        )

        self.agent_manager: AgentManager = AgentManager(
            ray_config=self.configs[0].ray_config.agent,
            env_stats=env_stats, 
            parameter_server=self.parameter_server,
            monitor=monitor
        )

        self.runner_manager: RunnerManager = RunnerManager(
            self.configs[0].runner,
            ray_config=configs[0].ray_config.runner,
            param_queues=queues.param,
            parameter_server=self.parameter_server,
            monitor=monitor
        )
        self.runner_manager.build_runners(configs)

    def pbt_train(self):
        iteration = 0
        while iteration < self.config.MAX_VERSION_ITERATIONS:
            self.initialize_actors()

            self.train(
                self.agent_manager, 
                self.runner_manager, 
            )

            self.cleanup()

    def initialize_actors(self):
        self.agent_manager.build_agents(self.configs)
        self.runner_manager.register_handler(
            remote_agents=self.agent_manager.get_agents())
        model_weights = ray.get(
            self.parameter_server.sample_training_strategies.remote())
        self.agent_manager.set_model_weights(model_weights)
        model_paths = [path for path, _ in model_weights]
        self.runner_manager.set_active_model_paths(model_paths, wait=True)
        if self.config.initialize_rms:
            self.runner_manager.set_current_model_paths(model_paths)
            self.runner_manager.random_run()

    def train(
        self, 
        agent_manager: AgentManager, 
        runner_manager: RunnerManager, 
    ):
        agent_manager.publish_weights()
        agent_manager.start_training()

        steps = 0
        while steps < self.config.MAX_STEPS_PER_ITERATION:
            steps += sum(runner_manager.run())

    def cleanup(self):
        self.agent_manager.destroy_agents()
        self.parameter_server.archive_training_strategies.remote()

    def _setup_configs(
        self, 
        configs: List[AttrDict], 
        env_stats: List[AttrDict]
    ):
        for aid, config in enumerate(configs):
            config.aid = aid
            config.buffer.n_envs = env_stats.n_workers * env_stats.n_envs
            root_dir = config.root_dir
            model_name = '/'.join([config.model_name, f'a{aid}'])
            modify_config(
                config, 
                root_dir=root_dir, 
                model_name=model_name, 
                aid=aid
            )

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

    def _setup_models(self):
            model_weights = ray.get(
                self.parameter_server.sample_training_strategies.remote())
            self.agent_manager.set_model_weights(model_weights)
            model_paths = [path for path, _ in model_weights]
            self.runner_manager.set_active_model_paths(model_paths, wait=True)
