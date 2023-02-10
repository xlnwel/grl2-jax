import logging
from types import FunctionType

from core.log import do_logging
from env.cls import Env, VecEnv, MASimVecEnv, MATBVecEnv
from env import make_env

logger = logging.getLogger(__name__)


def is_ma_suite(env_name):
    # ma_suites = ['overcooked', 'smac', 'grf', 'grid_world']
    ma_suites = ['smac', 'grf', 'grid_world']
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
    config.setdefault('n_envs', 1)
    env_fn = env_fn or make_env
    if config['env_name'].startswith('unity'):
        # Unity handles vectorized environments by itself
        env = Env(config, env_fn, agents=agents)
    elif no_remote or config.get('n_runners', 1) <= 1:
        config['n_runners'] = 1
        if force_envvec or config.n_envs > 1:
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
        EnvType = VecEnv if config.n_envs > 1 else Env
        env = RayVecEnv(EnvType, config, env_fn)
    return env

def get_env_stats(config):
    # TODO (cxw): store env_stats in a standalone file for costly environments
    config.setdefault('n_envs', 1)
    tmp_env_config = config.copy()
    tmp_env_config['n_runners'] = 1
    # we cannot change n_envs for unity environments
    if not config.env_name.startswith('unity'):
        tmp_env_config['n_envs'] = 1
    env = create_env(tmp_env_config, force_envvec=False, no_remote=True)
    env_stats = env.stats()
    env_stats.n_runners = config.get('n_runners', 1)
    env_stats.n_envs = env_stats.n_runners * config.n_envs
    do_logging(
        env_stats, 
        prefix='env stats', 
        logger=logger, 
        color='blue'
    )
    env.close()
    return env_stats


if __name__ == '__main__':
    import numpy as np
    config = dict(
        env_name='ma_mujoco-HalfCheetah_2x3',
        n_runners=1,
        n_envs=1,
        to_multi_agent=True,
    )
    env = create_env(config)
    for k in range(1000):
        a = env.random_action()
        o, r, d, re = env.step(a)
        if np.any(re):
            print('info', env.info())
