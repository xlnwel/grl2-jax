import os, sys

from utility import pkg
from utility.display import pwc
from utility.utils import eval_str, dict2AttrDict, modify_config
from utility.yaml_op import load_config


def get_configs_dir(algo):
    algo_dir = pkg.get_package_from_algo(algo, 0, '/')
    if algo_dir is None:
        raise RuntimeError(f'Algorithm({algo}) is not implemented')
    configs_dir = f'{algo_dir}/configs'

    return configs_dir


def get_filename_with_env(env):
    env_split = env.split('-')  # TODO: extra care need to be taken when we call env ending with version number (-vx)
    if len(env_split) == 1:
        filename = 'builtin'
    elif len(env_split) == 2:
        filename = env_split[0]
    else:   # we may use intermediate characters in env to specify the configuration filename
        filename = '_'.join(env_split[1:-1])

    return filename


def get_env_name(env):
    env_split = env.split('-')
    if len(env_split) == 1:
        env_name = env_split[0]
    else:
        # we may specify the configuration filename in between
        env_name = f'{env_split[0]}-{env_split[-1]}'
    return env_name


def change_config(kw, config, model_name=''):
    """ Changes configs based on kw. model_name will
    be modified accordingly to embody changes 
    """
    def change_dict(config, key, value, prefix):
        modified_configs = []
        for k, v in config.items():
            config_name = f'{prefix}:{k}' if prefix else k
            if key == k:
                config[k] = value
                modified_configs.append(config_name)
            if isinstance(v, dict):
                modified_configs += change_dict(v, key, value, config_name)
        return modified_configs
            
    if kw:
        for s in kw:
            key, value = s.split('=', 1)
            value = eval_str(value)
            if model_name != '':
                model_name += '-'
            model_name += s

            # change kwargs in config
            modified_configs = change_dict(config, key, value, '')
            pwc(f'All "{key}" appeared in the following configs will be changed: '
                + f'{modified_configs}.', color='cyan')
    return model_name


def load_config_with_algo_env(algo, env, filename=None, to_attrdict=True):
    configs_dir = get_configs_dir(algo)
    if filename is None:
        filename = get_filename_with_env(env)
    filename = filename + '.yaml'
    path = f'{configs_dir}/{filename}'

    config = load_config(path)
    if config is None:
        raise RuntimeError('No configure is loaded')
    env_name = get_env_name(env)

    config = modify_config(config, overwrite_existed=True, algorithm=algo, env_name=env_name)

    if to_attrdict:
        config = dict2AttrDict(config)

    return config


def search_for_all_configs(directory, to_attrdict=True):
    if not os.path.exists(directory):
        return []
    directory = directory
    config_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if 'src' in root:
                break
            elif f.endswith('config.yaml'):
                config_files.append(os.path.join(root, f))

    configs = []
    for f in config_files:
        configs.append(load_config(f, to_attrdict=to_attrdict))
    
    return configs


def search_for_config(directory, to_attrdict=True):
    if isinstance(directory, tuple):
        directory = '/'.join(directory)
    if not os.path.exists(directory):
        raise ValueError(f'Invalid directory: {directory}')
    directory = directory
    config_file = None
    for root, _, files in os.walk(directory):
        for f in files:
            if 'src' in root:
                break
            elif f.endswith('config.yaml') and config_file is None:
                config_file = os.path.join(root, f)
                break
            elif f.endswith('config.yaml') and config_file is not None:
                pwc(f'Get multiple "config.yaml": "{config_file}" and "{os.path.join(root, f)}"')
                sys.exit()

    if config_file is None:
        raise RuntimeError(f'No configure file is found in {directory}')
    config = load_config(config_file, to_attrdict=to_attrdict)
    
    return config


def load_and_run(directory):
    # load model and log path
    config_file = None
    for root, _, files in os.walk(directory):
        for f in files:
            if 'src' in root:
                break
            if f == 'config.yaml' and config_file is None:
                config_file = os.path.join(root, f)
                break
            elif f =='config.yaml' and config_file is not None:
                pwc(f'Get multiple "config.yaml": "{config_file}" and "{os.path.join(root, f)}"')
                sys.exit()

    config = load_config(config_file)
    configs = dict2AttrDict(config)
    
    main = pkg.import_main('train', config=configs.agent)

    main(configs)
