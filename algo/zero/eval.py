import time
import numpy as np
import ray

from .train import RunnerManager


def main(config, n, **kwargs):
    ray.init()
    if 'runner' not in config:
        config.runner = {}

    runner_manager = RunnerManager(config, name='zero', store_data=False)
    start = time.time()
    
    stats, n = runner_manager.evaluate(n)
    n_workers = config.env.n_workers

    for k, v in stats.items():
        pstd = np.std(np.mean(v, axis=-1)) * np.sqrt(n // n_workers)
        print(f'{k} averaged over {n} episodes: {np.mean(v)}({np.std(v)}, {pstd})')
    
    print(f'Total evaluation time: {time.time() - start}')

    ray.shutdown()
