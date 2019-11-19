import numpy as np
import tensorflow as tf
import ray

from utility.display import pwc
from utility.utils import set_global_seed
from utility.tf_utils import configure_gpu
from utility.signal import sigint_shutdown_ray
from env.gym_env import create_gym_env
from algo.ppo.nn import PPOAC


def evaluate(env, model):
    i = 0

    scores = []
    epslens = []
    while i < 10:
        i += env.n_envs
        state = env.reset()
        for _ in range(env.max_episode_steps):
            action = model.det_action(tf.convert_to_tensor(state, tf.float32))
            state, _, done, _ = env.step(action.numpy())

            if np.all(done):
                break
        scores.append(env.get_score())
        epslens.append(env.get_score())

    pwc(f'Score: {np.mean(scores)}\tEpslen: {np.mean(epslens)}')

def main(env_config, agent_config, render=False):
    set_global_seed()
    configure_gpu()

    if render:
        env_config['n_workers'] = env_config['n_envs'] = 1
    else:
        env_config['n_envs'] = 10
        env_config['n_workers'] = 10
    env_config['seed'] = np.random.randn()
    if env_config['n_workers'] > 1:
        ray.init()
        sigint_shutdown_ray()
    
    env = create_gym_env(env_config)

    ac = PPOAC(env.state_shape,
                env.action_dim,
                env.is_action_discrete,
                env.n_envs, 
                'ac')

    ckpt = tf.train.Checkpoint(ac=ac)
    ckpt_path = f'{agent_config["model_root_dir"]}/{agent_config["model_name"]}'
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, 5)

    path = ckpt_manager.latest_checkpoint
    ckpt.restore(path).expect_partial()
    if path:
        pwc(f'Params are restored from "{path}".', color='cyan')
        evaluate(env, ac)
    else:
        pwc(f'No model is found at "{ckpt_path}"!', color='magenta')

    ray.shutdown()