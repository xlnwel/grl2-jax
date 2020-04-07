import importlib
import tensorflow as tf
import ray

from run.pkg import get_package


def create_learner(Learner, name, model_fn, config, model_config, env_config, replay_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()

    config['model_name'] = 'learner'
    config['n_cpus'] = n_cpus = 4

    if tf.config.list_physical_devices('GPU'):
        RayLearner = ray.remote(num_cpus=n_cpus, num_gpus=.5)(Learner)
    else:
        RayLearner = ray.remote(num_cpus=n_cpus)(Learner)
        
    learner = RayLearner.remote(
        name, model_fn, config, model_config, env_config, replay_config)
    ray.get(learner.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config
    )))

    return learner


def create_worker(Worker, name, worker_id, env_config):
    env_config = env_config.copy()
    env_config['seed'] += worker_id * 100
    
    RayWorker = ray.remote(num_cpus=1)(Worker)
    worker = RayWorker.remote(name, worker_id, env_config)

    return worker

def create_actor(Actor, name, model_fn, config, model_config, env_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()

    config['model_name'] = 'actor'
    config['display_var'] = False
    config['writer'] = False
    if tf.config.list_physical_devices('GPU'):
        RayActor = ray.remote(num_cpus=1, num_gpus=.5)(Actor)
    else:
        RayActor = ray.remote(num_cpus=2)(Actor)

    actor = RayActor.remote(
        name, model_fn, config, model_config, env_config)
    
    return actor
