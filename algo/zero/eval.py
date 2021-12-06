import time
import numpy as np
import ray

from algo.zero.elements.runner import RunnerManager
from utility.ray_setup import sigint_shutdown_ray


def main(config, n, **kwargs):
    # from core.utils import save_config
    # config.name = 'zero'
    # config.runner.initialize_other = False
    # config.runner.self_play_frac = 0
    # save_config(config.root_dir, config.model_name, config)
    # exit()
    ray.init()
    sigint_shutdown_ray()

    config.buffer.agent_pids = config.runner.agent_pids = [0, 2]
    config.env.skip_players = []
    runner_manager = RunnerManager(config, store_data=False)
    start = time.time()
    
    stats, n = runner_manager.evaluate(
        n, 
        other_root_dir='logs/card_gd/zero',
        other_model_name='sp_exploiter_ws',
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
