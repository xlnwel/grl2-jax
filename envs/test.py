import time
import numpy as np
import ray

from envs.func import create_env
from tools.display import print_dict


if __name__ == '__main__':
  config = dict(
    env_name='spiel-leduc_poker',
    uid2aid=[0, 1], 
    n_envs=1, 
    timeout_done=True, 
    seed=0
    # n_envs=10,
    # max_episode_steps=100,
    # n_units=2,
    # num_good_agents=3,
    # num_adversaries=2,
    # num_landmarks=3
    # use_state_agent=True,
    # use_mustalive=True,
    # add_center_xy=True,
    # timeout_done=True,
    # add_agent_id=False,
    # debug=True
  )

  if config.get('n_runners', 1) > 1:
    ray.init()
  env = create_env(config)
  print(env)
  for i in range(10):
    a = env.random_action()
    
    out = env.step(a)
    if out[0].obs:
      o, r, d, re = out[0]
    else:
      o, r, d, re = out[1]
    info = env.info()
    print(r, d, info)
    
  # print(env.shared_state_space)

  # def run(env):
  #   st = time.time()
  #   for _ in range(100):
  #     a = env.random_action()
  #     _, r, d, _ = env.step(a)
  #     env.render()
  #     print('reward', r)
  #     time.sleep(.1)
  #     if not np.any(d):
  #       env.reset()
  #   return time.time() - st

  # print("Ray env:", run(env))

  if ray.is_initialized():
    ray.shutdown()
