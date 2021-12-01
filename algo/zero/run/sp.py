import ray

from .runner import RunnerManager
from .parameter_server import ParameterServer
from core.elements.builder import ElementsBuilder
from core.utils import save_config
from env.func import get_env_stats
from utility.ray_setup import sigint_shutdown_ray


def train(agent, buffer, runner_manager, parameter_server):
    pass

def sp_train(config):
    ray.init()
    sigint_shutdown_ray()

    root_dir = config.agent.root_dir
    model_name = config.agent.model_name

    env_stats = get_env_stats(config.env)
    config.buffer.n_envs = env_stats.n_workers * env_stats.n_envs

    name = f'{config.algorithm}_{config.id}' if 'id' in config else config.algorithm
    builder = ElementsBuilder(
        config, 
        env_stats, 
        name=name)
    elements = builder.build_agent_from_scratch()
    agent = elements.agent
    runner_manager = RunnerManager(config, name=agent.name)
    parameter_server = ParameterServer(config.parameter_server)

    save_config(root_dir, model_name, builder.get_config())

    train(agent, elements.buffer, runner_manager, parameter_server)

    ray.shutdown()