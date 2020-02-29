import time
import ray

from utility.signal import sigint_shutdown_ray
from utility.yaml_op import load_config
from env.gym_env import create_gym_env
from replay.func import create_replay_center
from algo.apex.actor import create_learner, create_worker


def import_agent(config):
    algo = config['algorithm']
    if algo.endswith('-sac-il'):
        from algo.sac_il.nn import create_model
        from algo.sac_il.agent import Agent
    elif algo.endswith('sac'):
        from algo.sac.nn import create_model
        from algo.sac.agent import Agent
    else:
        raise ValueError(f'Unknown algorithm: {algo}')

    return create_model, Agent

def import_worker_class(agent_config):
    if agent_config['algorithm'].startswith('asap2'):
        from algo.asap2.worker import Worker
    elif agent_config['algorithm'].startswith('asap'):
        if agent_config['algorithm'].endswith('il'):
            from algo.asap_il.worker import ILWorker as Worker
        else:
            from algo.asap.worker import Worker
    elif agent_config['algorithm'].startswith('apex'):
        from algo.apex.worker import Worker
    else:
        raise NotImplementedError

    return Worker

def import_learner_class(agent_config, BaseAgent):
    if agent_config['algorithm'].startswith('asap2'):
        from algo.asap2.learner import get_learner_class
    elif agent_config['algorithm'].startswith('asap'):
        from algo.apex.learner import get_learner_class
    elif agent_config['algorithm'].startswith('apex'):
        from algo.apex.learner import get_learner_class
    else:
        raise NotImplementedError

    return get_learner_class(BaseAgent)

def main(env_config, model_config, agent_config, replay_config, restore=False, render=False):
    if env_config.get('is_deepmind_env'):
        ray.init()
    else:
        ray.init(memory=8*1024**3, object_store_memory=7*1024**3)
    
    sigint_shutdown_ray()

    replay = create_replay_center(replay_config)

    model_fn, Agent = import_agent(agent_config)
    Learner = import_learner_class(agent_config, Agent)
    learner = create_learner(
        Learner=Learner, name='Learner', model_fn=model_fn, 
        replay=replay, config=agent_config, 
        model_config=model_config, env_config=env_config,
        replay_config=replay_config)

    if restore:
        ray.get(learner.restore.remote())
    Worker = import_worker_class(agent_config)
    workers = []
    pids = []
    for worker_id in range(agent_config['n_workers']):
        worker = create_worker(
            Worker=Worker, name='Worker', worker_id=worker_id, 
            model_fn=model_fn, config=agent_config, 
            model_config=model_config, env_config=env_config, 
            buffer_config=replay_config)
        worker.pull_weights.remote(learner)
        pids.append(worker.run.remote(learner, replay))
        workers.append(worker)

    while not ray.get(replay.good_to_learn.remote()):
        time.sleep(1)

    learner.start_learning.remote()

    ray.get(pids)

    ray.get(learner.save.remote())
    
    ray.shutdown()
