import time
import ray

from utility.ray_setup import sigint_shutdown_ray
from utility.yaml_op import load_config
from utility import pkg
from env.func import create_env
from replay.func import create_replay_center


default_agent_config = {    
    'MAX_STEPS': 1e8,
    'LOG_PERIOD': 1000,
    'N_UPDATES': 1000,
    'SYNC_PERIOD': 1000,
    'RECORD_PERIOD': 100,
    'N_EVALUATION': 10,

    # distributed algo params
    'n_learner_cpus': 3,
    'n_learner_gpus': 1,
    'n_workers': 8,
    'n_worker_cpus': 1,
    'n_worker_gpus': 0,
    'deterministic_evaluation': True,
}

def main(env_config, model_config, agent_config, replay_config):
    ray.init(num_cpus=12, num_gpus=1)
    sigint_shutdown_ray()
    default_agent_config.update(agent_config)
    agent_config = default_agent_config

    replay = create_replay_center(replay_config)

    model_fn, Agent = pkg.import_agent(config=agent_config)
    am = pkg.import_module('actor', config=agent_config)
    fm = pkg.import_module('func', config=agent_config)

    name = agent_config['algorithm']
    Learner = am.get_learner_class(Agent)
    learner = fm.create_learner(
        Learner=Learner, 
        model_fn=model_fn, 
        replay=replay, 
        config=agent_config, 
        model_config=model_config, 
        env_config=env_config,
        replay_config=replay_config)
   
    Worker = am.get_worker_class()
    workers = []
    pids = []
    for wid in range(agent_config['n_workers']):
        worker = fm.create_worker(
            Worker=Worker, 
            worker_id=wid, 
            model_fn=model_fn,
            config=agent_config, 
            model_config=model_config, 
            env_config=env_config, 
            buffer_config=replay_config)
        pids.append(worker.run.remote(learner, replay))
        workers.append(worker)

    Evaluator = am.get_evaluator_class()
    evaluator = fm.create_evaluator(
        Evaluator=Evaluator,
        model_fn=model_fn,
        config=agent_config,
        model_config=model_config,
        env_config=env_config)
    evaluator.run.remote(learner)

    learner.start_learning.remote()

    while ray.get(learner.is_learning.remote()):
        time.sleep(60)

    ray.get(learner.save.remote())
    
    ray.shutdown()
