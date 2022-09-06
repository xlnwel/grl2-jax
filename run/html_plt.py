import argparse
import os, sys
import json
from pathlib import Path
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.typing import ModelPath
from tools.file import search_for_dirs
from tools import yaml_op
from tools.utils import flatten_dict, recursively_remove


def get_model_path(dirpath) -> ModelPath:
    d = dirpath.split('/')
    model_path = ModelPath('/'.join(d[:3]), '/'.join(d[3:]))
    return model_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',
                        type=str,
                        default='.')
    parser.add_argument('--prefix', '-p', 
                        type=str, 
                        default=['seed'], 
                        nargs='*')
    parser.add_argument('--name', '-n', 
                        type=str, 
                        default=None)
    parser.add_argument('--target', '-t', 
                        type=str,
                        default='html-logs')
    args = parser.parse_args()

    return args


def remove_lists(d):
    to_remove_keys = []
    dicts = []
    for k, v in d.items():
        if isinstance(v, list):
            to_remove_keys.append(k)
        elif isinstance(v, dict):
            dicts.append((k, v))
    for k in to_remove_keys:
        del d[k]
    for k, v in dicts:
        d[k] = remove_lists(v)
    return d


def remove_redundancies(config: dict):
    redundancies = [k for k in config.keys() if k.endswith('id') and '/' in k]
    redundancies += [k for k in config.keys() if k.endswith('algorithm') and '/' in k]
    redundancies += [k for k in config.keys() if k.endswith('env_name') and '/' in k]
    redundancies += [k for k in config.keys() if k.endswith('model_name') and '/' in k]
    for k in redundancies:
        del config[k]
    return config


def rename_env(config: dict):
    env_name = config['env/env_name']
    suite = env_name.split('-', 1)[0]
    raw_env_name = env_name.split('-', 1)[1]
    config['env_name'] = env_name
    config['env_suite'] = suite
    config['raw_env_name'] = raw_env_name
    return config


if __name__ == '__main__':
    args = parse_args()
    
    config_name = 'config.yaml' 
    js_name = 'parameter.json'
    record_name = 'record.txt'
    process_name = 'progress.csv'
    for p in args.prefix:
        for d in search_for_dirs(args.directory, p, is_suffix=False, name=args.name):
            target_dir = d.replace(args.directory, args.target)
            print(f'copy from {d} to {target_dir}')
            if not os.path.isdir(target_dir):
                Path(target_dir).mkdir(parents=True)
            
            # define paths
            yaml_path = '/'.join([d, config_name])
            json_path = '/'.join([target_dir, js_name])
            record_path = '/'.join([d, record_name])
            process_path = '/'.join([target_dir, process_name])
            print('yaml path', yaml_path)
            if not os.path.exists(yaml_path) or not os.path.exists(record_path):
                print(f'{yaml_path} does not exist')
                continue
                
            # save config
            config = yaml_op.load_config(yaml_path)
            to_remove_keys = ['root_dir', 'seed']
            seed = config['seed']
            config = recursively_remove(config, to_remove_keys)
            config['seed'] = seed
            config = remove_lists(config)
            config = flatten_dict(config)
            config = rename_env(config)
            config = remove_redundancies(config)
            config['model_name'] = config['model_name'].split('/')[0]

            with open(json_path, 'w') as json_file:
                json.dump(config, json_file)

            # save stats
            try:
                data = pd.read_table(record_path, on_bad_lines='skip')
            except:
                print(f'Record path ({record_path}) constains no data')
                continue
            if len(data.keys()) == 1:
                data = pd.read_csv(record_path)
            for k in ['expl', 'latest_expl', 'nash_conv', 'latest_nash_conv']:
                if k not in data.keys():
                    try:
                        data[k] = (data[f'{k}1'] + data[f'{k}2']) / 2
                    except:
                        pass
            data.to_csv(process_path)
