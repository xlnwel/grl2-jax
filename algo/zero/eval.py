import time
import numpy as np
import ray

from .run.runner import RunnerManager


def main(config, n, **kwargs):
    ray.init()

    config.runner.control_pids = [0, 2]
    config.env.skip_players = []
    name = 'zero'
    runner_manager = RunnerManager(config, name=name, store_data=False)
    start = time.time()
    
    stats, n = runner_manager.evaluate(
        n, 
        other_path='logs/card_gd/zero/baseline/',
        other_name='zero_0'
    )
    n_workers = config.env.n_workers

    n = n - n % n_workers
    for k, v in stats.items():
        v = np.array(v[:n])
        pstd = np.std(np.mean(v.reshape(n_workers, -1), axis=-1)) * np.sqrt(n // n_workers)
        print(f'{k} averaged over {n} episodes: {np.mean(v):3g}({np.std(v):3g}, {pstd:3g})')
    
    duration = time.time() - start
    print(f'Evaluation time: total({duration:3g}), average({duration / n:3g})')

    ray.shutdown()
