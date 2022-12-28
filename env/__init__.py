import os
import importlib

# from gym.envs.registration import register

# register(
#     id='Overcooked-v0',
#     entry_point='env.overcooked.overcooked:OvercookedMultiEnv',
# )


def retrieve_all_make_env():
    env_dict = {}
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    for i in range(1, 10):
        pkg = 'env' if i == 1 else f'env{i}'
        if importlib.util.find_spec(pkg) is not None:
            if os.path.exists(os.path.join(root_dir, pkg, 'make.py')):
                make_pkg = f'{pkg}.make'
                m = importlib.import_module(make_pkg)
                for attr in dir(m):
                    if attr.startswith('make'):
                        env_dict[attr.split('_', maxsplit=1)[1]] = getattr(m, attr)
    
    return env_dict


def make_env(config, eid=None, agents={}):
    config = config.copy()
    env_name = config['env_name'].lower()

    env_dict = retrieve_all_make_env()
    suite = env_name.split('-', 1)[0]
    builtin_env = env_dict.pop('gym')
    env_func = env_dict.get(suite, builtin_env)
    if eid is not None:
        config['eid'] = eid
    if agents != {}:
        config.update(agents)
    env = env_func(config)
    
    return env
