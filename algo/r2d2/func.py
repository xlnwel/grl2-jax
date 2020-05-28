import tensorflow as tf
import ray

from utility.rl_utils import apex_epsilon_greedy
from algo.r2d2.buffer import create_local_buffer


def create_learner(Learner, model_fn, replay, config, model_config, env_config, replay_config, n_cpus):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    replay_config = replay_config.copy()
    
    env_config['n_workers'] = env_config['n_envs'] = 1
    config['model_name'] = 'learner'
    config['n_cpus'] = n_cpus

    if tf.config.list_physical_devices('GPU'):
        RayLearner = ray.remote(num_cpus=n_cpus, num_gpus=.5)(Learner)
    else:
        RayLearner = ray.remote(num_cpus=n_cpus)(Learner)

    learner = RayLearner.remote(
        config=config,
        model_config=model_config,
        env_config=env_config,
        model_fn=model_fn, 
        replay=replay)
    ray.get(learner.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        replay=replay_config
    )))

    return learner


def create_worker(Worker, worker_id, model_fn, config, model_config, 
                env_config, buffer_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()

    buffer_config['n_envs'] = env_config.get('n_envs', 1)
    buffer_fn = create_local_buffer

    env_config['seed'] += worker_id * 100
    
    if 'actor' not in model_config:
        config['act_eps'] = apex_epsilon_greedy(worker_id, config['n_workers'])
    RayWorker = ray.remote(num_cpus=1)(Worker)
    worker = RayWorker.remote(
        worker_id=worker_id, 
        config=config, 
        model_config=model_config, 
        env_config=env_config, 
        buffer_config=buffer_config,
        model_fn=model_fn, 
        buffer_fn=buffer_fn)

    return worker

def create_evaluator(Evaluator, model_fn, config, model_config, env_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()

    env_config['seed'] += 999
    
    RayEvaluator = ray.remote(num_cpus=1)(Evaluator)
    evaluator = RayEvaluator.remote(
        config=config,
        model_config=model_config,
        env_config=env_config,
        model_fn=model_fn)

    return evaluator