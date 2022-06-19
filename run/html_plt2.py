import argparse
import os, sys
import re
import json
from pathlib import Path
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.typing import ModelPath
from utility.file import search_for_dirs
from utility import yaml_op
from utility.utils import flatten_dict, recursively_remove


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
                        type=str)
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


if __name__ == '__main__':
    args = parse_args()
    
    dirs = search_for_dirs(args.directory, args.prefix, is_suffix=False)
    for d in dirs:
        # if d.split('/')[-1].startswith('0602'):
        #     print(d.split('/')[-1])
        #     continue
        target_dir = '/'.join([args.target, d])
        print(f'copy from {d} to {target_dir}')
        if not os.path.isdir(target_dir):
            Path(target_dir).mkdir(parents=True)
        
        # define paths
        yaml_path = '/'.join([d, 'config_p0.yaml'])
        json_path = '/'.join([target_dir, 'parameter.json'])
        record_path = '/'.join([d, 'nash_conv.txt'])
        process_path = '/'.join([target_dir, 'progress.csv'])
        print('yaml path', yaml_path)
        print('record path', record_path)
        if not os.path.exists(yaml_path) or not os.path.exists(record_path):
            continue
            
        # save config
        config = yaml_op.load_config(yaml_path)
        to_remove_keys = ['root_dir', 'model_name', 'seed']
        if isinstance(config.controller.max_steps_per_iteration, list):
            config['info'] = 'mspi=' + '-'.join([f'{x[0]}_{x[1]:g}' for x in config.controller.max_steps_per_iteration])
        else:
            config['info'] = f'mspi={config.controller.max_steps_per_iteration:g}'
        config = recursively_remove(config, to_remove_keys)
        config = remove_lists(config)

        config = flatten_dict(config)
        env_names = [(k, v) for k, v in config.items() if k.endswith('env_name')]
        for k, v in env_names:
            prefix = k.rsplit('/', 1)[0]
            suite = v.split('-', 1)[0]
            env_name = v.split('-', 1)[1]
            config[f'{prefix}/suite'] = suite
            config[k] = env_name
        with open(json_path, 'w') as json_file:
            json.dump(config, json_file)

        # save stats
        try:
            data = pd.read_table(record_path, on_bad_lines='skip')
        except:
            print(f'Record path ({record_path}) constains no data')
        if len(data.keys()) == 1:
            data = pd.read_csv(record_path)

        data.to_csv(process_path)
