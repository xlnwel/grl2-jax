import os, sys

from utility import pkg
from utility.display import pwc
from utility.file import search_for_all_files, search_for_file
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
        filename = 'gym'
    elif len(env_split) == 2:
        filename = env_split[0]
    else:
        raise ValueError(f'Cannot extract filename from env: {env}')

    return filename


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
            assert modified_configs != [], modified_configs

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

    config = modify_config(
        config, overwrite_existed=True, algorithm=algo, env_name=env)

    if to_attrdict:
        config = dict2AttrDict(config)

    return config


def search_for_all_configs(directory, to_attrdict=True):
    if not os.path.exists(directory):
        return []

    config_files = search_for_all_files(directory, 'config.yaml')
    if config_files == []:
        raise RuntimeError(f'No configure file is found in {directory}')
    configs = [load_config(f, to_attrdict=to_attrdict) for f in config_files]

    return configs


def search_for_config(directory, to_attrdict=True, check_duplicates=True):
    if isinstance(directory, tuple):
        directory = '/'.join(directory)
    if not os.path.exists(directory):
        raise ValueError(f'Invalid directory: {directory}')
    
    config_file = search_for_file(directory, 'config.yaml', check_duplicates)
    if config_file is None:
        raise RuntimeError(f'No configure file is found in {directory}')
    config = load_config(config_file, to_attrdict=to_attrdict)
    
    return config
