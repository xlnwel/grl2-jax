import time
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.tf_config import *
from env.func import create_env
from env.guandan.game import Game
from utility.utils import AttrDict2dict, dict2AttrDict
from algo.gd.ruleAI.sample import run_episode


def evaluate_against_reyn(agent, n):
    env = Game(skip_players13=True, agent13='reyn')

    scores = []
    scores2 = []
    epslens = []
    wins = []
    run_time = []
    for i in range(n):
        mptl, step, rt, eps_scores = run_episode(env, agent)
        score = eps_scores[0]
        score2 = eps_scores[1]
        scores.append(score)
        scores2.append(score2)
        epslens.append(step)
        wins.append(score > 0)
        run_time.append(rt)

    return scores, scores2, epslens, wins, run_time


@ray.remote
def evaluate_config(config, n):
    config = dict2AttrDict(config)
    silence_tf_logs()
    configure_gpu()
    configure_precision(config.precision)
    env = create_env(config.env)

    env_stats = env.stats()

    builder = ElementsBuilder(config, env_stats, config.algorithm)
    elements = builder.build_actor_agent_from_scratch()
    agent = elements.agent

    return evaluate_against_reyn(agent, n)


def main(config, n, **kwargs):
    ray.init()

    start = time.time()
    n_workers = config.env.n_workers
    config.env.n_workers = 1

    results = ray.get([
        evaluate_config.remote(AttrDict2dict(config), n//n_workers)
        for _ in range(n_workers)
    ])
    scores, scores2, epslens, wins, run_time = list(zip(*results))

    # scores, scores2, epslens, wins, run_time = evaluate_against_reyn(config, n)
    # population std = std(sample_mean) * sqrt(sample_size)
    win_pstd = np.std(np.mean(wins, axis=-1)) * np.sqrt(n // n_workers)
    score_pstd = np.std(np.mean(scores, axis=-1)) * np.sqrt(n // n_workers)
    score2_pstd = np.std(np.mean(scores2, axis=-1)) * np.sqrt(n // n_workers)
    print(f'Win rate averaged over {n} episodes: {np.mean(wins)}({np.std(wins)}, {win_pstd})')
    print(f'Average score: {np.mean(scores)}({np.std(scores)}, {score_pstd})')
    print(f'Average score(Enemy): {np.mean(scores2)}({np.std(scores2)}, {score2_pstd})')
    print(f'Average epslens: {np.mean(epslens)}')
    print(f'Episodic run time: {np.mean(run_time)}')
    print(f'Total evaluation time: {time.time() - start}')

    ray.shutdown()