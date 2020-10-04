# import numpy as np
# import tensorflow as tf
# import ray

# from core.tf_config import *
# from utility.display import pwc
# from utility.ray_setup import sigint_shutdown_ray
# from utility.run import evaluate
# from utility import pkg
# from env.func import create_env


# def main(env_config, model_config, agent_config, n, record=False, size=(128, 128)):
#     silence_tf_logs()
#     configure_gpu()

#     use_ray = env_config.get('n_workers', 0) > 1
#     if use_ray:
#         ray.init()
#         sigint_shutdown_ray()
        
#     if record:
#         env_config['log_episode'] = True
#         env_config['n_workers'] = env_config['n_envs'] = 1

#     env = create_env(env_config)

#     create_model, Agent = pkg.import_agent(config=agent_config)

#     actor = create_model(model_config, env)['actor']

#     ckpt = tf.train.Checkpoint(actor=actor)
#     ckpt_path = f'{agent_config["root_dir"]}/{agent_config["model_name"]}/models'
#     ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, 5)

#     path = ckpt_manager.latest_checkpoint
#     ckpt.restore(path).expect_partial()
#     if path:
#         pwc(f'Params are restored from "{path}".', color='cyan')
#         scores, epslens, video = evaluate(env, actor, n, record=record)
#         pwc(f'After running {n} episodes:',
#             f'Score: {np.mean(scores)}\tEpslen: {np.mean(epslens)}', color='cyan')
#     else:
#         pwc(f'No model is found at "{ckpt_path}"!', color='magenta')

#     ray.shutdown()