from types import FunctionType
from env.cls import Env, VecEnv, MASimVecEnv, MATBVecEnv
from env import make_env
from utility.display import print_dict


def is_ma_suite(env_name):
    ma_suites = ['overcooked', 'smac', 'grf']
    for suite in ma_suites:
        if env_name.startswith(suite):
            return True
    return False

def is_matb_suite(env_name):
    matb_suites = ['spiel']
    for suite in matb_suites:
        if env_name.startswith(suite):
            return True
    return False

def create_env(
    config: dict, 
    env_fn: FunctionType=None, 
    agents={}, 
    force_envvec=True, 
    no_remote=False
):
    """ Creates an Env/VecEnv from config """
    config = config.copy()
    env_fn = env_fn or make_env
    if config['env_name'].startswith('unity'):
        # Unity handles vectorized environments by itself
        env = Env(config, env_fn, agents=agents)
    elif no_remote or config.get('n_runners', 1) <= 1:
        config['n_runners'] = 1
        if force_envvec or config.get('n_envs', 1) > 1:
            if is_matb_suite(config['env_name']):
                EnvType = MATBVecEnv
            elif is_ma_suite(config['env_name']):
                EnvType = MASimVecEnv
            else:
                EnvType = VecEnv
        else:
            EnvType = Env
        env = EnvType(config, env_fn, agents=agents)
    else:
        from env.ray_env import RayVecEnv
        EnvType = VecEnv if config.get('n_envs', 1) > 1 else Env
        env = RayVecEnv(EnvType, config, env_fn)
    return env

def get_env_stats(config):
    # TODO (cxw): store env_stats in a standalone file for costly environments
    tmp_env_config = config.copy()
    tmp_env_config['n_runners'] = 1
    # we cannot change n_envs for unity environments
    if not config.env_name.startswith('unity'):
        tmp_env_config['n_envs'] = 1
    env = create_env(tmp_env_config, force_envvec=False, no_remote=True)
    env_stats = env.stats()
    env_stats.n_runners = config.get('n_runners', 1)
    env_stats.n_envs = env_stats.n_runners * config.n_envs
    print('Get env stats')
    print_dict(env_stats, 'env stats')
    env.close()
    return env_stats


if __name__ == '__main__':
    import time
    import numpy as np
    import collections
    State = collections.namedtuple('State', 'h c')
    def run(config):
        env = create_env(config)
        start = time.time()
        env.record_default_state(np.zeros(2), State(np.zeros((2, 2)), np.ones((2, 2))))
        env.record_default_state(np.ones(2), State(np.zeros((2, 2)), np.ones((2, 2))))
        obs = env.reset()
        print(obs)
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
        env_name='overcooked-asymmetric_advantages',
        max_episode_steps=400,
        n_envs=4,
        dense_reward=True,
        featurize=False,
        record_state=True,
        uid2aid=[0, 1]
    )
    run(config)
