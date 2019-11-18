import numpy as np
import tensorflow as tf

from utility.display import pwc
from utility.utils import set_global_seed
from utility.tf_utils import configure_gpu
from env.gym_env import create_gym_env
from algo.ppo.nn import PPOAC


def evaluate(env, model, name=None, logger=None, step=None):
    i = 0

    while i < 10:
        i += env.n_envs
        state = env.reset()
        for j in range(env.max_episode_steps):
            print(j)
            action = model.det_action(tf.convert_to_tensor(state, tf.float32))
            state, _, done, _ = env.step(action.numpy())

            if np.all(done):
                break



def main(env_config, agent_config, buffer_config, render=False):
    set_global_seed()
    configure_gpu()

    env_config['n_envs'] = 10
    env_config['n_workers'] = 10
    env = create_gym_env(env_config)

    ac = PPOAC(env.state_shape,
                env.action_dim,
                env.is_action_discrete,
                env.n_envs, 
                'ac')

    ckpt = tf.train.Checkpoint(ac=PPOAC)
    ckpt_path = f'{agent_config["model_root_dir"]}/{agent_config["model_name"]}'
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path)

    path = ckpt_manager.latest_checkpoint
    ckpt.restore(path)
    if path:
        pwc(f'Params for {self.name} are restored from "{path}".', color='cyan')
    else:
        pwc(f'No model for {self.name} is found at "{ckpt_path}"!', color='magenta')
        pwc(f'Continue or Exist (c/e):', color='magenta')
        ans = input()
        if ans.lower() == 'e':
            import sys
            sys.exit()
        else:
            pwc(f'Start training from scratch.', color='magenta')