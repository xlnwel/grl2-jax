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

def get_env_stats(config):
    # TODO (cxw): store env_stats in a standalone file for costly environments
    tmp_env_config = config.copy()
    tmp_env_config['n_workers'] = 1
    tmp_env_config['n_envs'] = 1
    env = create_env(tmp_env_config)
    env_stats = env.stats()
    env.close()
    return env_stats

if __name__ == '__main__':
    import time
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
        name='BipedalWalker-v3',
        n_workers=1,
        n_envs=1,
        use_state_agent=True,
        use_mustalive=True,
        add_center_xy=True,
        timeout_done=True,
        add_agent_id=False,
        obs_agent_id=False,
    )
    