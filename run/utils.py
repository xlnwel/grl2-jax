import os, sys
import logging

from utility import pkg
from utility.display import pwc
from utility.typing import AttrDict
from utility.utils import eval_str, deep_update
from utility.yaml_op import load_config


def get_config(algo, env):
    def search_add(word, files, filename):
        if [f for f in files if word in f]:
            # if suffix meets any config in the dir, we add it to filename
            filename = f'{word}_{filename}'
        return filename
    algo_dir = pkg.get_package_from_algo(algo, 0, '/')
    if env == '' and '-' in algo:
        pwc('Config Warning: set Procgen as the default env, otherwise specify env explicitly', color='green')
        env = 'procgen_'
    files = [f for f in os.listdir(algo_dir) if 'config.yaml' in f]
    filename = 'config.yaml'
    if '_' in env:
        prefix = env.split('_')[0]
        filename = search_add(prefix, files, filename)
    if '-' in algo:
        suffix = algo.split('-')[-1]
        if [f for f in files if suffix in f]:
            filename = search_add(suffix, files, filename)
        elif suffix[-1].isdigit():
            suffix = suffix[:-1]
            filename = search_add(suffix, files, filename)
    path = f'{algo_dir}/{filename}'
    
    config = load_config(path)
    if config:
        pwc(f'Config path: {path}', color='green')
    
    return config


def decompose_config(config: dict):
    configs = AttrDict(**config)
    for k, v in configs.items():
        if isinstance(v, dict):
            configs[k] = decompose_config(v)

    return configs


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
            key, value = s.split('=')
            value = eval_str(value)
            if model_name != '':
                model_name += '-'
            model_name += s

            # change kwargs in config
            key_configs = []
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


def load_configs_with_algo_env(algo, env):
    if '-' in algo:
        config = get_config(algo.split('-')[-1], env)
        dist_config = get_config(algo, env)
        assert config or dist_config, (config, dist_config)
        assert dist_config, dist_config
        if config == {}:
            config = dist_config
        config = deep_update(config, dist_config)
    else:
        config = get_config(algo, env)
    configs = decompose_config(config)
    configs.agent['algorithm'] = algo
    if env:
        configs.env['name'] = env

    return configs


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
    configs = decompose_config(config)
    
    main = pkg.import_main('train', config=configs.agent)

    main(configs)
