import tensorflow as tf
import ray


def create_learner(Learner, name, model_fn, config, model_config, env_config, replay_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    replay_config = replay_config.copy()
    
    config['model_name'] = 'learner'
    if tf.config.list_physical_devices('GPU'):
        RayLearner = ray.remote(num_cpus=2, num_gpus=1)(Learner)
    else:
        RayLearner = ray.remote(num_cpus=2)(Learner)
    learner = RayLearner.remote(
        name, model_fn, config, model_config, env_config, replay_config)
    ray.get(learner.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        replay=replay_config
    )))

    return learner


def create_worker(Worker, name, worker_id, env_config):
    env_config = env_config.copy()
    env_config['seed'] += worker_id * 100
    
    RayWorker = ray.remote(num_cpus=1)(Worker)
    worker = RayWorker.remote(name, worker_id, env_config)

    return worker
