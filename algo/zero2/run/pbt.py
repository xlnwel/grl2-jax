import numpy as np
import ray

from algo.zero2.remote.agent import RemoteAgent
from algo.zero2.remote.parameter_server import ParameterServer
from algo.zero2.remote.runner import RunnerManager
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
        config['root_dir'], 'sp')
    parameter_server.add_strategy_from_path.remote(
        config['root_dir'], 'self-play')
    parameter_server.add_strategy_from_path.remote(
        config['root_dir'], 'sp_exploiter_ws')
    parameter_server.add_strategy_from_path.remote(
        config['root_dir'], 'sp_exploiter_ws_ft')
    
    pbt_train(
        config=config, 
        env_stats=env_stats,
        parameter_server=parameter_server)


def pbt_train(
        config, 
        env_stats,
        parameter_server: ParameterServer):
    agent = RemoteAgent.as_remote(num_gpus=1).remote(config, env_stats)
    runner_manager = RunnerManager(
        config, 
        remote_agent=agent,
        parameter_server=parameter_server)

    model_path = ray.get(agent.get_model_path.remote())
    while True:
        if not ray.get(parameter_server.is_empty.remote()):
            wid = parameter_server.retrieve_latest_strategy_weights.remote()
            agent.set_weights.remote(wid)
        runner_manager.set_model_path(model_path)
        train(
            agent, 
            runner_manager, 
            parameter_server, 
            model_path, 
            config['agent']['LOG_PERIOD'],
        )
        parameter_server.add_strategy_from_path.remote(*model_path)
        model_path = agent.increase_version()


def train(
        agent: RemoteAgent, 
        runner_manager: RunnerManager, 
        parameter_server: ParameterServer, 
        model_path: ModelPath,
        LOG_PERIOD: int,
        SCORE_UPDATE_PERIOD: int):
    # # assert agent.get_env_step() == 0, (agent.get_env_step(), 'Comment out this line when you want to restore from a trained model')
    # if agent.get_env_step() == 0 and agent.actor.is_obs_normalized:
    #     obs_rms_list, rew_rms_list = runner_manager.initialize_rms()
    #     agent.update_rms_from_stats_list(obs_rms_list, rew_rms_list)

    rt = Timer('run')
    tt = Timer('train')

    train_step = ray.get(agent.get_train_step.remote())
    to_record = Every(LOG_PERIOD, train_step + LOG_PERIOD)
    to_update_score = Every(SCORE_UPDATE_PERIOD, SCORE_UPDATE_PERIOD)
    step = ray.get(agent.get_env_step.remote())
    version = ray.get(agent.get_version.remote())
    MAX_STEPS = runner_manager.max_steps()
    WIN_RATE_THRESHOLD = .65
    print('Training starts...')
    while step < MAX_STEPS:
        start_env_step = step
        with rt:
            wid = agent.get_weights.remote(opt_weights=False)
            parameter_server.add_strategy.remote(*model_path, wid)
            step, stats = runner_manager.run(wid)

        swid = parameter_server.get_scores_and_weights.remote(*model_path)

        start_train_step = train_step
        with tt:
            train_step = ray.get(agent.wait_for_train_step_update.remote())
        agent.store.remote({
            **stats,
            **{
                'time/run': rt.total(),
                'time/run_mean': rt.average(),
                'time/train': tt.total(),
                'time/train_mean': tt.average(),
                'time/fps': (step-start_env_step)/rt.last(),
                'time/outer_tps': (train_step-start_train_step)/tt.last()}
        })

        if to_update_score(train_step):
            parameter_server.compute_scores.remote(model_path)

        if to_record(train_step):
            scores, weights = ray.get(swid)
            agent.store.remote(**scores, **weights)
            agent.record.remote(step)

        if len(stats['win_rate']) > runner_manager.n_eval_envs \
            and np.mean(stats['win_rate']) > WIN_RATE_THRESHOLD:
            agent.record.remote(step)
            return f'Version{version} ends because win rate exceeds the threshold({WIN_RATE_THRESHOLD})'
    agent.record.remote(step)
    return f'Version{version} ends because the maximum number of steps are met'
