import logging
import numpy as np
import tensorflow as tf
import ray

from utility.rl_utils import apex_epsilon_greedy
from utility import pkg

logger = logging.getLogger(__name__)


def _compute_act_eps(config, worker_id, n_workers, envs_per_worker):
    if config.get('schedule_act_eps'):
        assert worker_id < n_workers, \
            f'worker ID({worker_id}) exceeds range. Valid range: [0, {config["n_workers"]})'
        act_eps_type = config.get('act_eps_type', 'apex')
        if act_eps_type == 'apex':
            config['act_eps'] = apex_epsilon_greedy(
                worker_id, envs_per_worker, n_workers, 
                epsilon=config['act_eps'], 
                sequential=config.get('seq_act_eps', True))
        elif act_eps_type == 'line':
            config['act_eps'] = np.linspace(
                0, config['act_eps'], n_workers * envs_per_worker)\
                    .reshape(n_workers, envs_per_worker)[worker_id]
        else:
            raise ValueError(f'Unknown type: {act_eps_type}')
        logger.info(f'worker_{worker_id} action epsilon: {config["act_eps"]}')
    return config

def _compute_act_temp(config, model_config, worker_id, n_workers, envs_per_worker):
    if config.get('schedule_act_temp'):
        n_exploit_envs = config.get('n_exploit_envs', 0)
        n_envs = n_workers * envs_per_worker
        n_exploit_envs = config.get('n_exploit_envs')
        if n_exploit_envs:
            act_temps = np.concatenate(
                [np.linspace(config['min_temp'], 1, n_exploit_envs), 
                np.logspace(0, np.log10(config['max_temp']), n_envs - n_exploit_envs)],
                axis=-1).reshape(n_workers, envs_per_worker)
        else:
            act_temps = np.logspace(
                np.log10(config['min_temp']), np.log10(config['max_temp']), 
                n_workers * envs_per_worker).reshape(n_workers, envs_per_worker)
        model_config['actor']['act_temp'] = act_temps[worker_id]
    return model_config

def create_learner(Learner, model_fn, replay, config, model_config, env_config, replay_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    replay_config = replay_config.copy()
    
    env_config['n_workers'] = env_config['n_envs'] = 1
    n_cpus = config.setdefault('n_learner_cpus', 3)

    if tf.config.list_physical_devices('GPU'):
        n_gpus = config.setdefault('n_learner_gpus', 1)
        RayLearner = ray.remote(num_cpus=n_cpus, num_gpus=n_gpus)(Learner)
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


def create_worker(
        Worker, worker_id, model_fn, 
        config, model_config, 
        env_config, buffer_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()

    n_workers = config['n_workers']
    n_envs = env_config.get('n_envs', 1)
    buffer_config['n_envs'] = env_config.get('n_envs', 1)
    buffer_fn = pkg.import_module(
        'buffer', config=config, place=0).create_local_buffer

    if 'seed' in env_config:
        env_config['seed'] += worker_id * 100
    
    config = _compute_act_eps(config, worker_id, n_workers, n_envs)
    model_config = _compute_act_temp(config, model_config, worker_id, n_workers, n_envs)
    
    n_cpus = config.get('n_worker_cpus', 1)
    n_gpus = config.get('n_worker_gpus', 0)
    RayWorker = ray.remote(num_cpus=1, num_gpus=n_gpus)(Worker)
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

    if 'seed' in env_config:
        env_config['seed'] += 999
    env_config['n_workers'] = env_config['n_envs'] = 1
    model_config['actor']['act_temp'] = config.pop('eval_act_temp', .5)

    RayEvaluator = ray.remote(num_cpus=1)(Evaluator)
    evaluator = RayEvaluator.remote(
        config=config,
        model_config=model_config,
        env_config=env_config,
        model_fn=model_fn)

    return evaluator
