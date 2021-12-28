import time
import numpy as np
import ray

from algo.zero.remote.runner import RunnerManager
from utility.ray_setup import sigint_shutdown_ray


def main(config, n, other_path, **kwargs):
    ray.init()
    sigint_shutdown_ray()

    config.buffer.agent_pids = config.runner.agent_pids = [0, 2]
    if other_path is None:
        config.env.skip_players = [1, 3]

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
