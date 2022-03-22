from distributed.sync_sim.eval import *
import time
import numpy as np

from .remote.runner import MultiAgentSimRunner


def main(configs, n, **kwargs):
    for c in configs:
        c['env']['unity_config']['file_name'] = None
        c['env']['unity_config']['worker_id'] = -1
        c['env']['n_envs'] = 1

    runner = MultiAgentSimRunner(
        0,
        configs,
        store_data=False,
        evaluation=True,
        param_queues=None,
        parameter_server=None,
    )
    runner.set_weights_from_configs(configs)
    start = time.time()
    steps, n_episodes, _, _, stats = runner.evaluate(n)
    duration = time.time() - start

    config = configs[0]
    n_agents = config.n_agents
    n_workers = config.runner.n_runners
    n = n_episodes - n_episodes % n_workers
    for k, v in stats.items():
        print(k, v[0])
        for aid in range(n_agents):
            v = np.array(v[:n])
            pstd = np.std(np.mean(v.reshape(n_workers, -1), axis=-1)) * np.sqrt(n // n_workers)
            print(
                f'Agent{aid}: {k} averaged over {n_episodes} episodes: mean({np.mean(v):3g}), std({np.std(v):3g}), pstd({pstd:3g})')

    print(f'Evaluation time: total({duration:3g}),',
        f'episode per second({duration / n_episodes:3g}),',
        f'steps per second({duration / steps})')
