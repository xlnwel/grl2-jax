import ray

from .parameter_server import ParameterServer
from .ppo import train as ppo_train
from .runner import RunnerManager
from core.elements.builder import ElementsBuilder
from env.func import get_env_stats
from run.utils import search_for_config
from utility.ray_setup import sigint_shutdown_ray


@ray.remote
def train(
        config, 
        env_stats,
        name,
        parameter_server: ParameterServer):
    builder = ElementsBuilder(
        config, 
        env_stats, 
        name=name,
        incremental_version=True)
    runner_manager = RunnerManager(
        config, 
        name=name, 
        parameter_server=parameter_server)

    while True:
        elements = builder.build_agent_from_scratch()

        other_path = parameter_server.sample()
        other_config = search_for_config(other_path)
        runner_manager.set_other_player(other_config)

        ppo_train(elements.agent, elements.buffer, runner_manager)

        parameter_server.add_strategy(builder.get_model_path())
        builder.increase_version()


def main(config):
    ray.init()
    sigint_shutdown_ray()

    env_stats = get_env_stats(config.env)
    config.buffer.n_envs = env_stats.n_workers * env_stats.n_envs

    parameter_server = ParameterServer.as_remote()(config.parameter_server)

    train(config, parameter_server)

    ray.shutdown()
