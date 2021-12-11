import os, sys

from core.typing import ModelPath
from utility import pkg
from utility.display import pwc
from utility.utils import eval_str, dict2AttrDict
from utility.yaml_op import load_config


def get_config(algo, env):
    def search_add(word, files, filename):
        if [f for f in files if word in f]:
            # if suffix meets any config in the dir, we add it to filename
            filename = f'{word}_{filename}'
        return filename
    algo_dir = pkg.get_package_from_algo(algo, 0, '/')
    if algo_dir is None:
        raise RuntimeError(f'Algorithm({algo}) is not implemented')
    configs_dir = f'{algo_dir}/configs'
    files = [f for f in os.listdir(configs_dir) if 'config.yaml' in f]
    env_split = env.split('_')
    if len(env_split) == 1:
        filename = 'builtin.yaml'
    elif len(env_split) == 2:
        filename = f'{env_split[0]}.yaml'
    else:
        filename = '_'.join(env_split[1:-1]) + '.yaml'
    if '-' in algo:
        suffix = algo.split('-')[-1]
        if [f for f in files if suffix in f]:
            filename = search_add(suffix, files, filename)
        elif suffix[-1].isdigit():
            suffix = suffix[:-1]
            filename = search_add(suffix, files, filename)
    path = f'{configs_dir}/{filename}'

    config = load_config(path)
    if config is None:
        raise RuntimeError('No configure is loaded')

    config['algorithm'] = algo
    config['env']['name'] = f'{env_split[0]}_{env_split[-1]}'   # we may have version number in between

    return config


def change_config(kw, configs, model_name=''):
    """ Changes configs based on kw. model_name will
    be modified accordingly to embody changes 
    """
    def extract_dicts(config):
        keys = []
        values = []
        for k, v in config.items():
            if isinstance(v, dict):
                keys.append(k)
                values.append(v)
                ks, vs = extract_dicts(v)
                keys += ks
                values += vs
        return keys, values
    
    config_keys, config_values = extract_dicts(configs)
    if kw:
        for s in kw:
            key, value = s.split('=', 1)
            value = eval_str(value)
            if model_name != '':
                model_name += '-'
            model_name += s

            # change kwargs in config
            key_configs = [('config', configs)] if key in configs else []
            for name, config in zip(config_keys, config_values):
                if key in config:
                    key_configs.append((name, config))
            assert key_configs, f'"{s}" does not appear in any config!'
            if len(key_configs) > 1:
                pwc(f'All {key} appeared in the following configs will be changed: '
                        + f'{list([n for n, _ in key_configs])}.', color='cyan')
                
            for _, c in key_configs:
                c[key]  = value

    return model_name


def load_configs_with_algo_env(algo, env, to_attrdict=True):
    config = get_config(algo, env)
    
    if to_attrdict:
        config = dict2AttrDict(config)

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


def set_path(config, model_path: ModelPath, recursive=True):
    config['root_dir'] = model_path.root_dir
    config['model_name'] = model_path.model_name
    if recursive:
        for v in config.values():
            if not isinstance(v, dict):
                continue
            v['root_dir'] = model_path.root_dir
            v['model_name'] = model_path.model_name
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

    config = load_config(config_file, to_attrdict=to_attrdict)
    
    return config

def get_other_path(other_dir):
    if other_dir:
        other_config = search_for_config(other_dir)
        other_path = ModelPath(other_config.root_dir, other_config.model_name)
    else:
        other_path = None

    return other_path
