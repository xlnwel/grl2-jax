import ray

from algo.zero.elements.parameter_server import ParameterServer
from algo.zero.elements.runner import RunnerManager
from core.elements.builder import ElementsBuilder
from core.tf_config import configure_gpu, silence_tf_logs
from env.func import get_env_stats
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

    ray.get([pbt_train.remote(
        config=config.asdict(), 
        env_stats=env_stats.asdict(),
        name=config.algorithm,
        parameter_server=parameter_server) 
        for _ in range(1)])

    ray.shutdown()


@ray.remote
def pbt_train(
        config, 
        env_stats,
        name,
        parameter_server: ParameterServer):
    silence_tf_logs()
    configure_gpu()
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
        train(elements.agent, elements.buffer, runner_manager)

        parameter_server.add_strategy(builder.get_model_path())
        builder.increase_version()


def train(agent, buffer, runner_manager, parameter_server=None):
    # assert agent.get_env_step() == 0, (agent.get_env_step(), 'Comment out this line when you want to restore from a trained model')
    if agent.get_env_step() == 0 and agent.actor.is_obs_normalized:
        obs_rms_list, rew_rms_list = runner_manager.initialize_rms()
        agent.update_rms_from_stats_list(obs_rms_list, rew_rms_list)

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

    step = agent.get_env_step()
    print('Training starts...')
    while step < runner_manager.MAX_STEPS:
        start_env_step = agent.get_env_step()
        with rt:
            weights = agent.get_weights(opt_weights=False)
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
