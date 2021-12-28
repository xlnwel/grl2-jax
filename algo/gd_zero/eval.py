import time
import numpy as np
import ray

from algo.gd_zero.remote.runner import RunnerManager
from core.typing import ModelPath
from utility.utils import set_path
from utility.ray_setup import sigint_shutdown_ray


def main(config, n, other_path, **kwargs):
    # from core.utils import save_config
    # config.name = 'gd_zero'
    # model_name = 'sp_exploiter_ws_ft'
    # set_path(config, ModelPath(config.root_dir, model_name))
    # config.runner.initialize_other = False
    # config.runner.self_play_frac = 0
    # save_config(config.root_dir, config.model_name, config)
    # exit()
    ray.init()
    sigint_shutdown_ray()

    config.buffer.agent_pids = config.runner.agent_pids = [0, 2]
    if other_path is None:
        config.env.skip_players = [1, 3]
    # config.env.name = 'card_gd'
    config.runner.initialize_other = False
    runner_manager = RunnerManager(config, store_data=False, evaluation=True)
    start = time.time()
    stats, n = runner_manager.evaluate(
        n, 
        other_path=other_path,
    )
    n_workers = config.env.n_workers

    n = n - n % n_workers
    for k, v in stats.items():
        v = np.array(v[:n])
        pstd = np.std(np.mean(v.reshape(n_workers, -1), axis=-1)) * np.sqrt(n // n_workers)
        print(f'{k} averaged over {n} episodes: mean({np.mean(v):3g}), std({np.std(v):3g}), pstd({pstd:3g})')
    
    duration = time.time() - start
    print(f'Evaluation time: total({duration:3g}), average({duration / n:3g})')

    ray.shutdown()

# def evaluate_against_reyn(agent, n):
#     from env.guandan.game import Game
#     from algo.gd_zero.ruleAI.sample import run_episode
#     env = Game(test=False)

#     scores = []
#     scores2 = []
#     epslens = []
#     wins = []
#     run_time = []
#     for i in range(n):
#         mptl, step, rt, eps_scores = run_episode(env, agent)
#         score = eps_scores[0]
#         score2 = eps_scores[1]
#         scores.append(score)
#         scores2.append(score2)
#         epslens.append(step)
#         wins.append(score > 0)
#         run_time.append(rt)

#     return scores, scores2, epslens, wins, run_time

# from core.elements.builder import ElementsBuilder
# from core.tf_config import *
# from env.func import get_env_stats
# from utility.utils import AttrDict2dict, dict2AttrDict

# @ray.remote
# def evaluate_config(config, n):
#     config = dict2AttrDict(config)
#     silence_tf_logs()
#     configure_gpu()
#     configure_precision(config.precision)
#     env_stats = get_env_stats(config.env)
#     builder = ElementsBuilder(config, env_stats)
#     elements = builder.build_acting_agent_from_scratch()
#     agent = elements.agent

#     return evaluate_against_reyn(agent, n)


# def main(config, n, **kwargs):
#     ray.init()

#     start = time.time()
#     n_workers = config.env.n_workers
#     config.env.n_workers = 1

#     results = ray.get([
#         evaluate_config.remote(AttrDict2dict(config), n//n_workers)
#         for _ in range(n_workers)
#     ])
#     # results = evaluate_config(AttrDict2dict(config), n)
#     scores, scores2, epslens, wins, run_time = list(zip(*results))

#     # scores, scores2, epslens, wins, run_time = evaluate_against_reyn(config, n)
#     # population std = std(sample_mean) * sqrt(sample_size)
#     win_pstd = np.std(np.mean(wins, axis=-1)) * np.sqrt(n // n_workers)
#     score_pstd = np.std(np.mean(scores, axis=-1)) * np.sqrt(n // n_workers)
#     score2_pstd = np.std(np.mean(scores2, axis=-1)) * np.sqrt(n // n_workers)
#     print(f'Win rate averaged over {n} episodes: {np.mean(wins)}({np.std(wins)}, {win_pstd})')
#     print(f'Average score: {np.mean(scores)}({np.std(scores)}, {score_pstd})')
#     print(f'Average score(Enemy): {np.mean(scores2)}({np.std(scores2)}, {score2_pstd})')
#     print(f'Average epslens: {np.mean(epslens)}')
#     print(f'Episodic run time: {np.mean(run_time)}')
#     print(f'Total evaluation time: {time.time() - start}')

#     ray.shutdown()
