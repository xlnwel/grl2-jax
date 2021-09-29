import numpy as np

from env.cls import Env, EnvVec
from env import make_env


def create_env(config, env_fn=None, force_envvec=True):
    """ Creates an Env/EnvVec from config """
    config = config.copy()
    env_fn = env_fn or make_env
    if config.get('n_workers', 1) <= 1:
        EnvType = EnvVec if force_envvec or config.get('n_envs', 1) > 1 else Env
        env = EnvType(config, env_fn)
    else:
        from env.ray_env import RayEnvVec
        EnvType = EnvVec if config.get('n_envs', 1) > 1 else Env
        env = RayEnvVec(EnvType, config, env_fn)

    return env


if __name__ == '__main__':
    def run(config):
        env = create_env(config)
        start = time.time()
        env.reset()
        return time.time() - start
        # st = time.time()
        # for _ in range(10000):
        #     a = env.random_action()
        #     _, _, d, _ = env.step(a)
        #     if np.any(d == 0):
        #         idx = [i for i, dd in enumerate(d) if dd == 0]
        #         # print(idx)
        #         env.reset(idx)
        # return time.time() - st
        env.close()
    import ray
    # performance test
    config = dict(
        name='smac_6h_vs_8z',
        n_workers=8,
        n_envs=1,
        use_state_agent=True,
        use_mustalive=True,
        add_center_xy=True,
        timeout_done=True,
        add_agent_id=False,
        obs_agent_id=False,
    )
    import time
    ray.init()
    config['n_envs'] = 2
    config['n_workers'] = 8
    duration1 = run(config)
    ray.shutdown()
    ray.init()
    config['n_envs'] = 1
    config['n_workers'] = 16
    duration2 = run(config)
    print(f'RayEnvVec({config["n_workers"]}, {config["n_envs"]})', duration1)
    print(f'EnvVec({config["n_workers"]}, {config["n_envs"]})', duration2)
    ray.shutdown()
    