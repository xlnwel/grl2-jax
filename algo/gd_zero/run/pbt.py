import numpy as np
import ray

from algo.gd_zero.remote.parameter_server import ParameterServer
from algo.gd_zero.remote.runner import RunnerManager
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
        config=config.asdict(),
        env_stats=env_stats.asdict())

    ray.get(parameter_server.search_for_strategies.remote(config.root_dir.split('/')[0]))

    pbt_train(
        config=config, 
        env_stats=env_stats,
        parameter_server=parameter_server) 


def pbt_train(
        config, 
        env_stats,
        parameter_server: ParameterServer):
    silence_tf_logs()
    configure_gpu()
    builder = ElementsBuilder(
        config, 
        env_stats, 
        incremental_version=True)
    runner_manager = RunnerManager(
        config, 
        parameter_server=parameter_server)
    elements = builder.build_agent_from_scratch()
    agent = elements.agent

    i = builder.get_version()
    verbose_path = f'{config["root_dir"]}/{config["model_name"]}/verbose.txt'
    model_path = builder.get_model_path()
    is_ps_empty = ray.get(parameter_server.is_empty.remote(model_path))
    while True:
        if not is_ps_empty:
            wid = parameter_server.retrieve_strategy_weights_from_checkpoint.remote()
            weights = ray.get(wid)
            if weights:
                agent.set_weights(weights.weights)
        if is_ps_empty:
            runner_manager.force_self_play()
            is_ps_empty = False
        else:
            runner_manager.play_with_pool()

        # agent.strategy.model.action_type.randomize_last_layer()
        # agent.strategy.model.card_rank.randomize_last_layer()
        model_path = builder.get_model_path()
        agent.reset_model_path(model_path)
        parameter_server.add_strategy_from_path.remote(model_path)
        runner_manager.set_model_path(model_path)
        end_with_cond = train(
            elements.agent, 
            elements.buffer, 
            runner_manager, 
            parameter_server, 
            version=builder.get_version())
        with open(verbose_path, 'a') as f:
            f.write(f'{end_with_cond}\n')
        builder.increase_version()
        # exit()
        i += 1


def train(
        agent, 
        buffer, 
        runner_manager: RunnerManager, 
        parameter_server: ParameterServer,
        version: int):
    rt = Timer('run')
    tt = Timer('train')
    at = Timer('aux')
    lt = Timer('log')

    def record_stats(step):
        with lt:
            agent.store(**{
                'misc/train_step': agent.get_train_step(),
                'time/run': rt.total(), 
                'time/train_record': tt.total(),
                'time/log': lt.total(),
                'time/run_mean': rt.average(), 
                'time/train_record_mean': tt.average(),
                'time/log_mean': lt.average(),
            })
            agent.record(step=step)
            agent.save()

    train_step = agent.get_train_step()
    to_record = Every(agent.LOG_PERIOD, train_step + agent.LOG_PERIOD)
    to_push_weights = Every(int(1e8), int(1e8))
    step = agent.get_env_step()
    assert step == 0, step
    assert train_step == 0, train_step
    MAX_STEPS = runner_manager.max_steps()
    WIN_RATE_THRESHOLD = .65
    count = 0
    model_path = agent.get_model_path()
    print('Training starts...')
    while step < MAX_STEPS:
        start_env_step = step
        swid = parameter_server.get_scores_and_weights.remote(model_path)

        weights = agent.get_weights(opt_weights=False)
        wid = ray.put(weights)

        with rt:
            step, data, run_stats = runner_manager.run(wid)

        for d in data:
            buffer.merge(d)
        buffer.finish()

        start_train_step = train_step
        with tt:
            agent.train_record()
        train_step = agent.get_train_step()

        if buffer.type() == 'ppg' and buffer.ready_for_aux_training():
            buffer.compute_aux_data_with_func(agent.compute_logits_values)
            with at:
                stats = agent.aux_train_record()
            agent.store(**stats)

        agent.set_env_step(step)
        runner_manager.reset()

        scores, weights, real_time_scores, real_time_weights = ray.get(swid)
        agent.store(
            **run_stats,
            **scores,
            **weights,
            **real_time_scores, 
            **real_time_weights,
            **{
                'time/fps': (step-start_env_step)/rt.last(),
                'time/outer_tps': (train_step-start_train_step)/tt.last()})

        if to_record(train_step):
            record_stats(step)

        if to_push_weights(step):
            parameter_server.update_strategy.remote(model_path, wid)
            parameter_server.search_for_strategies.remote(model_path.root_dir)

        if len(run_stats['win_rate']) > runner_manager.n_eval_envs:
            win_rate = np.mean(run_stats['win_rate'])
            if win_rate > WIN_RATE_THRESHOLD:
                count += 1
                if count >= 5:
                    return f'Version{version} ends because win rate({win_rate}) exceeds the threshold({WIN_RATE_THRESHOLD})'
            else:
                count = max(0, count - 1)
    record_stats(step)
    # push weights after finishing training
    parameter_server.update_strategy.remote(model_path, wid)
    return f'Version{version} ends because the maximum number of steps({step} > {MAX_STEPS}) are met'
