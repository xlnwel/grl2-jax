import time
import importlib
import numpy as np
import ray

from utility.ray_setup import sigint_shutdown_ray
from run import pkg


def main(env_config, model_config, agent_config, replay_config, restore=False, render=False):
    ray.init(num_cpus=12, num_gpus=1)

    sigint_shutdown_ray()

    model_fn, Agent = pkg.import_agent(agent_config)
    am = pkg.import_module('actor', config=agent_config)
    fm = pkg.import_module('func', config=agent_config)

    name = agent_config['algorithm'].rsplit('-', 1)[-1]

    Learner = am.get_learner_class(Agent)
    learner = fm.create_learner(
        Learner, 
        name=name, 
        model_fn=model_fn,
        config=agent_config, 
        model_config=model_config, 
        env_config=env_config,
        replay_config=replay_config)
    
    Actor = am.get_actor_class(Agent)
    actor = fm.create_actor(
        Actor, name, model_fn, agent_config, model_config, env_config)

    if restore:
        ray.get(learner.restore.remote())

    Worker = am.get_worker_class()
    workers = []
    for wid in range(agent_config['n_workers']):
        worker = fm.create_worker(Worker, name, wid, env_config)
        workers.append(worker)
        
    actor.start.remote(workers, learner)
    learner.start.remote(actor)

    while True:
        time.sleep(10000)