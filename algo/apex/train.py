import time
import ray

from utility.signal import sigint_shutdown_ray
from env.gym_env import create_gym_env
from replay.func import create_replay_center


def import_agent(config):
    algo = config['algorithm']
    if algo.endswith('sac'):
        from algo.sac.nn import create_model
        from algo.sac.agent import Agent
    elif algo.endswith('dqn'):
        from algo.d3qn.nn import create_model
        from algo.d3qn.agent import Agent
    else:
        raise ValueError(f'Unknown algorithm: {algo}')

    return create_model, Agent

def get_worker_fn(agent_config):
    if agent_config['algorithm'].startswith('apex-es2'):
        from algo.apex_es2.worker import create_worker
    elif agent_config['algorithm'].startswith('apex-es'):
        from algo.apex_es.worker import create_worker
    elif agent_config['algorithm'].startswith('apex'):
        from algo.apex.worker import create_worker
    else:
        raise NotImplementedError

    return create_worker

def get_learner_fn(agent_config):
    if agent_config['algorithm'].startswith('apex-es2'):
        from algo.apex_es2.learner import create_learner
    elif agent_config['algorithm'].startswith('apex-es'):
        from algo.apex.learner import create_learner
    elif agent_config['algorithm'].startswith('apex'):
        from algo.apex.learner import create_learner
    else:
        raise NotImplementedError

    return create_learner

def main(env_config, model_config, agent_config, replay_config, restore=False, render=False):
    if agent_config['n_workers'] == 4:
        ray.init(memory=8*1024**3, object_store_memory=4*1024**3, num_cpus=6)
    else:
        ray.init()
    sigint_shutdown_ray()

    env_config_copy = env_config.copy()
    env_config_copy['n_workers'] = env_config_copy['n_envs'] = 1
    env = create_gym_env(env_config)

    replay_keys = ['state', 'action', 'reward', 'done', 'steps']
    replay = create_replay_center(replay_config, *replay_keys, state_shape=env.state_shape)
    env.close()

    create_learner = get_learner_fn(agent_config)
    model_fn, Agent = import_agent(agent_config)
    learner = create_learner(
        BaseAgent=Agent, name='Learner', model_fn=model_fn, 
        replay=replay, config=agent_config, 
        model_config=model_config, env_config=env_config,
        replay_config=replay_config)

    if restore:
        ray.get(learner.restore.remote())
    create_worker = get_worker_fn(agent_config)
    workers = []
    pids = []
    for worker_id in range(agent_config['n_workers']):
        worker = create_worker(
            name='Worker', worker_id=worker_id, 
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

    ray.shutdown()
