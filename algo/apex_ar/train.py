import time
import ray

from utility.signal import sigint_shutdown_ray
from env.gym_env import create_gym_env
from algo.sacar.replay.func import create_replay_center
from algo.apex_ar.learner import create_learner
from algo.apex_ar.worker import create_worker


def import_agent(config):
    algo = config['algorithm']
    if algo.endswith('sac'):
        from algo.sacar.nn import create_model
        from algo.sacar.agent import Agent
    elif algo.endswith('dqn'):
        from algo.d3qn.nn import create_model
        from algo.d3qn.agent import Agent
    else:
        raise ValueError(f'Unknown algorithm: {algo}')

    return create_model, Agent

def main(env_config, model_config, agent_config, replay_config, restore=False, render=False):
    ray.init()
    sigint_shutdown_ray()

    env_config_copy = env_config.copy()
    env_config_copy['n_workers'] = env_config_copy['n_envs'] = 1
    env = create_gym_env(env_config)

    replay_keys = ['state', 'action', 'n_ar', 'reward', 'done', 'steps']
    replay = create_replay_center(replay_config, *replay_keys, state_shape=env.state_shape)
    env.close()

    model_fn, Agent = import_agent(agent_config)
    learner = create_learner(
        BaseAgent=Agent, name='Learner', model_fn=model_fn, 
        replay=replay, config=agent_config, 
        model_config=model_config, env_config=env_config,
        replay_config=replay_config)

    if restore:
        ray.get(learner.restore.remote())
    
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
