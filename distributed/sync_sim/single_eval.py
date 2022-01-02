import time
import numpy as np

from algo.zero.remote.runner import RunnerManager
from algo.zero.elements.model import create_model
from algo.zero.elements.actor import create_actor
from core.elements.strategy import create_strategy
from core.tf_config import *
from env.func import get_env_stats
from run.args import parse_eval_args
from run.utils import search_for_config


def main(config, n, **kwargs):
    args = parse_eval_args()
    config = search_for_config(args.directory)

    silence_tf_logs()
    configure_gpu()
    configure_precision(config.precision)


    config.buffer.agent_pids = config.runner.agent_pids = [0, 2]
    config.env.skip_players = []
    
    env_stats = get_env_stats(config.env)
    model = create_model(
        config.model, env_stats, name=config.name,
        to_build_for_eval=True)
    actor = create_actor(config.actor, model, name=config.name)
    strategy = create_strategy()

    runner_manager = RunnerManager(config, store_data=False)
    start = time.time()
    
    stats, n = runner_manager.evaluate(
        n, 
        other_root_dir='logs/card_gd/zero',
        other_model_name='baseline',
    )
    n_workers = config.env.n_workers

    n = n - n % n_workers
    for k, v in stats.items():
        v = np.array(v[:n])
        pstd = np.std(np.mean(v.reshape(n_workers, -1), axis=-1)) * np.sqrt(n // n_workers)
        print(f'{k} averaged over {n} episodes: mean({np.mean(v):3g}), std({np.std(v):3g}), pstd({pstd:3g})')
    
    duration = time.time() - start
    print(f'Evaluation time: total({duration:3g}), average({duration / n:3g})')
