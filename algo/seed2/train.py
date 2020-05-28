import time
import importlib
import numpy as np
import ray

from utility.signal import sigint_shutdown_ray
from run import pkg
from replay.func import create_replay_center



def main(env_config, model_config, agent_config, replay_config, restore=False, render=False):
    ray.init()

    sigint_shutdown_ray()

    replay_config['dir'] = agent_config['root_dir'].replace('logs', 'data')
    replay = create_replay_center(replay_config)
    ray.get(replay.load_data.remote())

    model_fn, Agent = pkg.import_agent(agent_config)
    am = pkg.import_module('actor', config=agent_config)
    fm = pkg.import_module('func', config=agent_config)

    name = agent_config['algorithm'].rsplit('-', 1)[-1]
    Actor = am.get_actor_class(Agent)
    actor = fm.create_actor(
        Actor, name, model_fn, agent_config, model_config, env_config)

    Learner = am.get_learner_class(Agent)
    learner = fm.create_learner(
        Learner, 
        name=name, 
        model_fn=model_fn,
        replay=replay,
        config=agent_config, 
        model_config=model_config, 
        env_config=env_config)
    
    if restore:
        ray.get(learner.restore.remote())

    Worker = am.get_worker_class()
    workers = []
    for wid in range(agent_config['n_workers']):
        worker = fm.create_worker(Worker, name, wid, env_config)
        workers.append(worker)

    actor.start.remote(workers, replay)
    ray.get(learner.start.remote(actor))
    