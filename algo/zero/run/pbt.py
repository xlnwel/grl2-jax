import numpy as np
import ray

from algo.zero.elements.parameter_server import ParameterServer
from algo.zero.elements.runner import RunnerManager
from core.elements.builder import ElementsBuilder
from core.tf_config import configure_gpu, silence_tf_logs
from env.func import get_env_stats
from run.utils import search_for_config
from utility.ray_setup import sigint_shutdown_ray
from utility.timer import Every, Timer


def main(config):
    ray.init()
    sigint_shutdown_ray()

    env_stats = get_env_stats(config.env)
    config.buffer.n_envs = env_stats.n_workers * env_stats.n_envs

    parameter_server = ParameterServer.as_remote().remote(
        config=config.parameter_server.asdict(),
        env_stats=env_stats.asdict())
    parameter_server.add_strategy_from_path.remote(
        '/'.join([config.root_dir, config.model_name, 'v0']))
    parameter_server.add_strategy_from_path.remote(
        '/'.join([config.root_dir, config.model_name, 'v1']))
    # parameter_server = ParameterServer(
    #     config=config.parameter_server.asdict(),
    #     env_stats=env_stats.asdict())
    # parameter_server.add_strategy_from_path(
    #     '/'.join([config.root_dir, config.model_name, 'v0']))
    ray.get([pbt_train.remote(
        config=config.asdict(), 
        env_stats=env_stats.asdict(),
        parameter_server=parameter_server) 
        for _ in range(1)])

    ray.shutdown()


@ray.remote
def pbt_train(
        config, 
        env_stats,
        parameter_server: ParameterServer):
    silence_tf_logs()
    configure_gpu()
    builder = ElementsBuilder(
        config, 
        env_stats, 
        incremental_version=True,
        start_version=1)
    runner_manager = RunnerManager(
        config, 
        to_initialize_other=True,
        parameter_server=parameter_server)
    elements = builder.build_agent_from_scratch()
    agent = elements.agent

    i = 1
    while True:
        model_path = builder.get_model_path()
        if not ray.get(parameter_server.is_empty.remote()):
            weights = ray.get(
                parameter_server.sample_strategy.remote(
                    opt_weights=True, actor_weights=True))
            agent.set_weights(weights)
            agent.reset(*model_path)
        # train(elements.agent, elements.buffer, runner_manager, parameter_server)
        print('/'.join(model_path))
        parameter_server.add_strategy_from_path.remote(
            '/'.join(model_path))
        builder.increase_version()
        i += 1


def train(agent, buffer, runner_manager, parameter_server):
    # assert agent.get_env_step() == 0, (agent.get_env_step(), 'Comment out this line when you want to restore from a trained model')
    if agent.get_env_step() == 0 and agent.actor.is_obs_normalized:
        obs_rms_list, rew_rms_list = runner_manager.initialize_rms()
        agent.update_rms_from_stats_list(obs_rms_list, rew_rms_list)

    step = agent.get_env_step()
    to_record = Every(agent.LOG_PERIOD, agent.LOG_PERIOD)
    rt = Timer('run', 1)
    tt = Timer('train', 1)
    lt = Timer('log', 1)

    def record_stats(step):
        with lt:
            agent.store(**{
                'misc/train_step': agent.get_train_step(),
                'time/run': rt.total(), 
                'time/train': tt.total(),
                'time/log': lt.total(),
                'time/run_mean': rt.average(), 
                'time/train_mean': tt.average(),
                'time/log_mean': lt.average(),
            })
            agent.record(step=step)
            agent.save()

    MAX_STEPS = runner_manager.max_steps()
    print('Training starts...')
    while step < MAX_STEPS:
        start_env_step = agent.get_env_step()
        with rt:
            weights = agent.get_weights(opt_weights=False)
            parameter_server.add_strategy.remote(
                '/'.join(agent.get_model_path()), weights)
            step, data, stats = runner_manager.run(weights)

        for d in data:
            buffer.append_data(d)
        buffer.finish()

        start_train_step = agent.get_train_step()
        with tt:
            agent.train_record()
        train_step = agent.get_train_step()

        agent.store(
            **stats,
            fps=(step-start_env_step)/rt.last(),
            tps=(train_step-start_train_step)/tt.last())
        agent.set_env_step(step)
        buffer.reset()
        runner_manager.reset()

        if to_record(train_step) and agent.contains_stats('score'):
            record_stats(step)

        print('win_rate', stats['win_rate'])
        if len(stats['win_rate']) > 20 and np.all(stats['win_rate'] > .6):
            break
