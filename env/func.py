import numpy as np
import gym

from env.env import Env, EnvVec, RayEnvVec, make_env


def create_env(config, env_fn=None, force_envvec=False):
    config = config.copy()
    env_fn = env_fn or make_env
    if config.get('n_workers', 1) <= 1:
        EnvType = EnvVec if force_envvec or config.get('n_envs', 1) > 1 else Env
        env = EnvType(config, env_fn)
    else:
        EnvType = EnvVec if config.get('n_envs', 1) > 1 else Env
        env = RayEnvVec(EnvType, config, env_fn)

    return env


if __name__ == '__main__':
    import ray
    # performance test
    config = dict(
        name='atari_breakout',
        wrapper='baselines',
        sticky_actions=True,
        frame_stack=4,
        life_done=True,
        np_obs=False,
        seed=0,
    )
    import time
    ray.init()
    config['n_envs'] = 2
    config['n_workers'] = 4
    env = create_env(config)
    st = time.time()
    s = env.reset()
    for _ in range(1000):
        a = env.random_action()
        s, r, d, re = env.step(a)
        if np.any(re):
            idx = [i for i, rr in enumerate(re) if rr]
            info = env.info(idx)
            for i in info:
                print(idx, info, i)
    print(f'RayEnvVec({config["n_workers"]}, {config["n_envs"]})', time.time() - st)
    
    ray.shutdown()
    config['n_envs'] = config['n_workers'] * config['n_envs']
    config['n_workers'] = 1
    env = create_env(config)
    s = env.reset()
    for _ in range(1000):
        a = env.random_action()
        s, r, d, re = env.step(a)
        if np.any(re):
            idx = [i for i, rr in enumerate(re) if rr]
            info = env.info(idx)
            for i in info:
                print(i)
    print(f'EnvVec({config["n_workers"]}, {config["n_envs"]})', time.time() - st)
    