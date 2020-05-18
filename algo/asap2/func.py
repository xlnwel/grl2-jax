import tensorflow as tf
import ray

from algo.apex.buffer import create_local_buffer


def create_learner(Learner, name, model_fn, replay, config, model_config, env_config, replay_config, n_cpus):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    replay_config = replay_config.copy()
    
    env_config['n_workers'] = env_config['n_envs'] = 1
    config['model_name'] = 'learner'
    config['n_steps'] = replay_config['n_steps']
    config['n_cpus'] = n_cpus

    if tf.config.list_physical_devices('GPU'):
        RayLearner = ray.remote(num_cpus=n_cpus, num_gpus=.5)(Learner)
    else:
        RayLearner = ray.remote(num_cpus=n_cpus)(Learner)

    learner = RayLearner.remote(
        name, model_fn, replay, config, model_config, env_config)
    ray.get(learner.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        replay=replay_config
    )))

    return learner


def create_worker(Worker, name, worker_id, model_fn, config, model_config, 
                env_config, buffer_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()

    buffer_config['n_envs'] = env_config.get('n_envs', 1)
    buffer_fn = create_local_buffer

    env_config['seed'] += worker_id * 100
    
    RayWorker = ray.remote(num_cpus=1)(Worker)
    worker = RayWorker.remote(name, worker_id, model_fn, buffer_fn, config, 
                        model_config, env_config, buffer_config)

    return worker
