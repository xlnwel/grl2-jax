import time
import importlib
import numpy as np
import ray

from utility.ray_setup import sigint_shutdown_ray
from utility import pkg


def main(env_config, model_config, agent_config, replay_config, restore=False, render=False):
    ray.init()

    sigint_shutdown_ray()

    model_fn, Agent = pkg.import_agent(config=agent_config)
    am = pkg.import_module('actor', config=agent_config)
    fm = pkg.import_module('func', config=agent_config)
    
    name = agent_config['algorithm'].rsplit('-', 1)[-1]
    Learner = am.get_learner_class(Agent)
    learner = fm.create_learner(
        Learner=Learner, 
        name=name, 
        model_fn=model_fn,
        config=agent_config, 
        model_config=model_config, 
        env_config=env_config, 
        replay_config=replay_config)
    
    if restore:
        ray.get(learner.restore.remote())

    Worker = am.get_worker_class()
    workers = []
    for wid in range(agent_config['n_workers']):
        worker = fm.create_worker(Worker, name, wid, env_config)
        workers.append(worker)

    learner.start_learning.remote()
    learner.start.remote(workers)

    while True:
        time.sleep(10000)