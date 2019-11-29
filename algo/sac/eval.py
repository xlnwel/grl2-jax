import numpy as np
import tensorflow as tf
import ray

from core.tf_config import configure_gpu
from utility.display import pwc
from utility.utils import set_global_seed
from utility.signal import sigint_shutdown_ray
from env.gym_env import create_gym_env
from algo.run import run_trajectories
from algo.sac.nn import SoftPolicy


def main(env_config, model_config, agent_config, render=False):
    set_global_seed()
    configure_gpu()

    env = create_gym_env(env_config)

    actor = SoftPolicy(model_config['actor'],
                        env.state_shape,
                        env.action_dim,
                        env.is_action_discrete,
                        'actor')

    ckpt = tf.train.Checkpoint(actor=actor)
    ckpt_path = f'{agent_config["model_root_dir"]}/{agent_config["model_name"]}'
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, 5)

    path = ckpt_manager.latest_checkpoint
    ckpt.restore(path).expect_partial()
    if path:
        pwc(f'Params are restored from "{path}".', color='cyan')
        scores, epslens = run_trajectories(env, actor, evaluation=True)
        pwc(f'After running 100 episodes:',
            f'Score: {np.mean(scores)}\tEpslen: {np.mean(epslens)}', color='cyan')
    else:
        pwc(f'No model is found at "{ckpt_path}"!', color='magenta')

    ray.shutdown()