import numpy as np
import tensorflow as tf
import ray

from core.tf_config import *
from utility.display import pwc
from utility.signal import sigint_shutdown_ray
from env.gym_env import create_env
from algo.sac.run import evaluate
from algo.sac.nn import SoftPolicy


def main(env_config, model_config, agent_config, n, render=False):
    silence_tf_logs()
    configure_gpu()

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()
        
    env = create_env(env_config)
    n_envs = env_config['n_envs'] * env_config['n_workers']

    actor = SoftPolicy(model_config['actor'],
                        env.obs_shape,
                        env.action_dim,
                        env.is_action_discrete,
                        'actor')

    ckpt = tf.train.Checkpoint(actor=actor)
    ckpt_path = f'{agent_config["root_dir"]}/{agent_config["model_name"]}/models'
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, 5)

    path = ckpt_manager.latest_checkpoint
    ckpt.restore(path).expect_partial()
    if path:
        pwc(f'Params are restored from "{path}".', color='cyan')
        scores, epslens = evaluate(env, agent, n, render=render)
        pwc(f'After running {n_envs} episodes:',
            f'Score: {np.mean(scores)}\tEpslen: {np.mean(epslens)}', color='cyan')
    else:
        pwc(f'No model is found at "{ckpt_path}"!', color='magenta')

    ray.shutdown()