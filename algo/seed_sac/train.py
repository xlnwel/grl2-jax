import time
import numpy as np
import ray

from utility.signal import sigint_shutdown_ray
from algo.seed_sac.learner import create_learner
from algo.seed_sac.worker import create_worker


def main(env_config, model_config, agent_config, buffer_config, restore=False, render=False):
    ray.init()
    sigint_shutdown_ray()

    learner = create_learner(
        name='Learner', config=agent_config, 
        model_config=model_config, env_config=env_config, 
        buffer_config=buffer_config)
    
    if restore:
        ray.get(learner.restore.remote())

    workers = []
    for worker_id in range(agent_config['n_workers']):
        worker = create_worker(
            worker_id=worker_id, 
            env_config=env_config)
        worker.start_env.remote(learner)
        worker.start_step_loop.remote(learner)
        workers.append(worker)

    learner.start_action_loop.remote(workers)
    
    while True:
        # sleep forever...
        time.sleep(6000)
