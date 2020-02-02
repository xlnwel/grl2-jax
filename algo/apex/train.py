import time
import ray

from utility.signal import sigint_shutdown_ray
from utility.yaml_op import load_config
from env.gym_env import create_gym_env
from replay.func import create_replay_center


def import_agent(config):
    algo = config['algorithm']
    if algo.endswith('-il-sac'):
        from algo.sac_il.nn import create_model
        from algo.sac_il.agent import Agent
    elif algo.endswith('sac'):
        from algo.sac.nn import create_model
        from algo.sac.agent import Agent
    elif algo.endswith('dqn'):
        from algo.d3qn.nn import create_model
        from algo.d3qn.agent import Agent
    else:
        raise ValueError(f'Unknown algorithm: {algo}')

    return create_model, Agent

def get_worker_fn(agent_config):
    if agent_config['algorithm'].startswith('asap2'):
        from algo.asap2.worker import create_worker
    elif agent_config['algorithm'].startswith('asap'):
        from algo.asap.worker import create_worker
    elif agent_config['algorithm'].startswith('apex'):
        from algo.apex.worker import create_worker
    else:
        raise NotImplementedError

    return create_worker

def get_learner_fn(agent_config):
    if agent_config['algorithm'].startswith('asap2'):
        from algo.asap2.learner import create_learner
    elif agent_config['algorithm'].startswith('asap'):
        from algo.apex.learner import create_learner
    elif agent_config['algorithm'].startswith('apex'):
        from algo.apex.learner import create_learner
    else:
        raise NotImplementedError

    return create_learner

def get_bph_config(agent_config):
    """ get configure file for BipedalWalkerHardcore-v2 """
    if agent_config['algorithm'].startswith('asap'):
        config_file = 'algo/asap/bph_sac_config.yaml'
    elif agent_config['algorithm'].startswith('apex'):
        config_file = 'algo/apex/bph_sac_config.yaml'
    else:
        raise NotImplementedError
    
    return config_file

def main(env_config, model_config, agent_config, replay_config, restore=False, render=False):
    if env_config.get('is_deepmind_env'):
        ray.init()
    else:
        ray.init(memory=8*1024**3, object_store_memory=7*1024**3)
    
    if env_config['name'] == 'BipedalWalkerHardcore-v2':
        # Caveat: this keeps most default configuration
        algorithm = agent_config['algorithm']
        root_dir = agent_config['root_dir']
        video_path = env_config['video_path']
        config_file = get_bph_config(agent_config)
        config = load_config(config_file)
        env_config = config['env']
        model_config = config['model']
        agent_config = config['agent']
        replay_config = config.get('buffer') or config.get('replay')
        agent_config['algorithm'] = algorithm
        agent_config['root_dir'] = root_dir
        env_config['video_path'] = video_path
    sigint_shutdown_ray()

    replay = create_replay_center(replay_config)

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

    ray.get(learner.save.remote())
    
    ray.shutdown()
