import numpy as np
import ray

from algo.zero.elements.parameter_server import ParameterServer
from algo.zero.elements.runner import RunnerManager
from core.elements.builder import ElementsBuilder
from core.tf_config import configure_gpu, silence_tf_logs
from core.typing import ModelPath
from env.func import get_env_stats
from utility.ray_setup import sigint_shutdown_ray
from utility.timer import Every, Timer


def main(config):
    ray.init()
    sigint_shutdown_ray()

    env_stats = get_env_stats(config.env)
    config.buffer.n_envs = env_stats.n_workers * env_stats.n_envs

    config = config.asdict()
    env_stats = env_stats.asdict()
    parameter_server = ParameterServer.as_remote().remote(
        config=config,
        env_stats=env_stats)
    parameter_server.add_strategy_from_path.remote(
        ModelPath(config['root_dir'], 'sp'))
    parameter_server.add_strategy_from_path.remote(
        ModelPath(config['root_dir'], 'self-play'))
    parameter_server.add_strategy_from_path.remote(
        ModelPath(config['root_dir'], 'sp_exploiter_ws'))
    parameter_server.add_strategy_from_path.remote(
        ModelPath(config['root_dir'], 'sp_exploiter_ws_ft'))
    
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
        incremental_version=True,
        start_version=0)
    runner_manager = RunnerManager(
        config, 
        parameter_server=parameter_server)
    elements = builder.build_agent_from_scratch()
    agent = elements.agent

    i = 0
    path = f'{config["root_dir"]}/{config["model_name"]}/verbose.txt'
    while True:
        if not ray.get(parameter_server.is_empty.remote()):
            wid = parameter_server.retrieve_latest_strategy_weights.remote()
            weights = ray.get(wid)
            agent.set_weights(weights)
        # agent.strategy.model.action_type.randomize_last_layer()
        # agent.strategy.model.card_rank.randomize_last_layer()
        model_path = builder.get_model_path()
        agent.reset_model_path(model_path)
        runner_manager.set_model_path(model_path)
        end_with_cond = train(
            elements.agent, 
            elements.buffer, 
            runner_manager, 
            parameter_server, 
            version=builder.get_version(),
            SCORE_UPDATE_PERIOD=config['parameter_server']['SCORE_UPDATE_PERIOD'])
        with open(path, 'a') as f:
            f.write(f'{end_with_cond}\n')
        parameter_server.add_strategy_from_path.remote(model_path)
        builder.increase_version()
        i += 1


def train(
        agent, 
        buffer, 
        runner_manager: RunnerManager, 
        parameter_server: ParameterServer,
        version: int,
        SCORE_UPDATE_PERIOD: int):
    # assert agent.get_env_step() == 0, (agent.get_env_step(), 'Comment out this line when you want to restore from a trained model')
    if agent.get_env_step() == 0 and agent.actor.is_obs_normalized:
        obs_rms_list, rew_rms_list = runner_manager.initialize_rms()
        agent.update_rms_from_stats_list(obs_rms_list, rew_rms_list)

    rt = Timer('run')
    tt = Timer('train')
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
    to_update_score = Every(SCORE_UPDATE_PERIOD, SCORE_UPDATE_PERIOD)
    step = agent.get_env_step()
    MAX_STEPS = runner_manager.max_steps()
    WIN_RATE_THRESHOLD = .65
    count = 0
    print('Training starts...')
    while step < MAX_STEPS:
        start_env_step = step
        model_path = agent.get_model_path()
        with rt:
            weights = agent.get_weights(opt_weights=False)
            wid = ray.put(weights)
            parameter_server.add_strategy.remote(
                model_path, wid)
            step, data, stats = runner_manager.run(wid)
        
        swid = parameter_server.get_scores_and_weights.remote(model_path)
        for d in data:
            buffer.merge(d)
        buffer.finish()

        start_train_step = train_step
        with tt:
            agent.train_record()
        train_step = agent.get_train_step()
        scores, weights = ray.get(swid)
        agent.store(
            **stats,
            **scores,
            **weights,
            **{
                'time/fps': (step-start_env_step)/rt.last(),
                'time/outer_tps': (train_step-start_train_step)/tt.last()})

        agent.set_env_step(step)
        buffer.reset()
        runner_manager.reset()

        if to_update_score(train_step):
            parameter_server.compute_scores.remote(model_path)

        if to_record(train_step) and agent.contains_stats('score'):
            record_stats(step)

        if len(stats['win_rate']) > runner_manager.n_eval_envs \
            and np.mean(stats['win_rate']) > WIN_RATE_THRESHOLD:
            count += 1
            if count >= 5:
                return f'Version{version} ends because win rate exceeds the threshold({WIN_RATE_THRESHOLD})'
        else:
            count = max(0, count - 1)
    return f'Version{version} ends because the maximum number of steps are met'
