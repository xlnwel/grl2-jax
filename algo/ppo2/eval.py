import numpy as np
import tensorflow as tf
import ray

from utility.display import pwc
from utility.utils import set_global_seed
from core.tf_config import configure_gpu
from utility.signal import sigint_shutdown_ray
from env.gym_env import create_gym_env
from algo.ppo2.nn import create_model


def evaluate(env, agent):
    pwc('Evaluation starts', color='cyan')
    i = 0

    scores = []
    epslens = []
    while i < 100:
        i += env.n_envs
        agent.reset_states(batch_size=env.n_envs)
        state = env.reset()
        for _ in range(env.max_episode_steps):
            action = agent.step(state, deterministic=True)
            state, _, done, _ = env.step(action.numpy())

            if np.all(done):
                break
            
        scores.append(env.get_score())
        epslens.append(env.get_epslen())

    return scores, epslens

def main(env_config, model_config, agent_config, render=False):
    set_global_seed()
    configure_gpu()

    env = create_gym_env(env_config)

    ac = create_model(
        model_config, 
        obs_shape=env.obs_shape, 
        action_dim=env.action_dim, 
        is_action_discrete=env.is_action_discrete,
        n_envs=env.n_envs
    )['ac']

    ckpt = tf.train.Checkpoint(ac=ac)
    ckpt_path = f'{agent_config["root_dir"]}/{agent_config["model_name"]}/models'
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, 5)

    path = ckpt_manager.latest_checkpoint
    ckpt.restore(path).expect_partial()
    if path:
        pwc(f'Params are restored from "{path}".', color='cyan')
        scores, epslens = evaluate(env, ac)
        pwc(f'After running 100 episodes',
            f'Score: {np.mean(scores)}\tEpslen: {np.mean(epslens)}', color='cyan')
    else:
        pwc(f'No model is found at "{ckpt_path}"!', color='magenta')
