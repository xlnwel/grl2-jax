import argparse
import os, sys
from pathlib import Path
import json
from pathlib import Path
import pandas as pd
import subprocess
import collections

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.log import do_logging
from tools.file import yield_dirs
from tools import yaml_op
from tools.utils import flatten_dict, recursively_remove

ModelPath = collections.namedtuple('ModelPath', 'root_dir model_name')

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
                        default=None, 
                        nargs='*')
    parser.add_argument('--target', '-t', 
                        type=str, 
                        default='~/Documents/html-logs')
    parser.add_argument('--date', '-d', 
                        type=str, 
                        default=None)
    parser.add_argument('--sync', 
                        action='store_true')
    parser.add_argument('--ignore', '-i',
                        type=str, 
                        default=None)
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
    player0_config_name = 'config_p0.yaml' 
    js_name = 'parameter.json'
    record_name = 'record.txt'
    process_name = 'progress.csv'
    date = args.date
    do_logging(f'Loading logs on date: {date}')

    directory = os.path.abspath(args.directory)
    target = os.path.expanduser(args.target)
    sync_dest = os.path.expanduser(args.target)
    do_logging(f'Directory: {directory}')
    do_logging(f'Target: {target}')

    while directory.endswith('/'):
        directory = directory[:-1]
    
    if directory.startswith('/'):
        strs = directory.split('/')
    process = None
    if args.sync:
        old_logs = '/'.join(strs)
        new_logs = f'~/Documents/' + '/'.join(strs[8:])
        if not os.path.exists(new_logs):
            Path(new_logs).mkdir(parents=True)
        cmd = ['rsync', '-avz', old_logs, new_logs, '--exclude', 'src']
        for n in args.name:
            cmd += ['--include', n]
        do_logging(' '.join(cmd))
        process = subprocess.Popen(cmd)

    for p in args.prefix:
        do_logging(f'Finding directories with prefix {p} in {directory}')
        for d in yield_dirs(directory, p, is_suffix=False, matches=args.name):
            if date is not None and date not in d:
                print(f'Pass directory {d} due to mismatch date')
                continue
                
            if args.ignore and args.ignore in d:
                print(f'Pass directory {d} as it contains ignore pattern "{args.ignore}"')
                continue

            # load config
            yaml_path = '/'.join([d, config_name])
            if not os.path.exists(yaml_path):
                new_yaml_path = '/'.join([d, player0_config_name])
                if os.path.exists(new_yaml_path):
                    yaml_path = new_yaml_path
                else:
                    do_logging(f'{yaml_path} does not exist', color='magenta')
                    continue
            config = yaml_op.load_config(yaml_path)
            root_dir = config.root_dir
            model_name = config.model_name
            strs = f'{root_dir}/{model_name}'.split('/')
            for s in strs[::-1]:
                if directory.endswith(s):
                    directory = directory.removesuffix(f'/{s}')

            target_dir = d.replace(directory, target)
            do_logging(f'Copy from {d} to {target_dir}')
            if not os.path.isdir(target_dir):
                Path(target_dir).mkdir(parents=True)
            assert os.path.isdir(target_dir), target_dir
            
            # define paths
            json_path = '/'.join([target_dir, js_name])
            record_path = '/'.join([d, record_name])
            process_path = '/'.join([target_dir, process_name])
            # do_logging(f'yaml path: {yaml_path}')
            if not os.path.exists(record_path):
                do_logging(f'{record_path} does not exist', color='magenta')
                continue
            # save config
            to_remove_keys = ['root_dir', 'seed']
            seed = config['seed']
            config = recursively_remove(config, to_remove_keys)
            config['seed'] = seed
            config = remove_lists(config)
            config = flatten_dict(config)
            config = rename_env(config)
            config = remove_redundancies(config)
            config['model_name'] = config['model_name'].split('/')[1]

            with open(json_path, 'w') as json_file:
                json.dump(config, json_file)

            # save stats
            try:
                data = pd.read_table(record_path, on_bad_lines='skip')
            except:
                do_logging(f'Record path ({record_path}) constains no data', color='magenta')
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

    if process is not None:
        do_logging('Waiting for rsync to complete...')
        process.wait()

    do_logging('Transfer completed')
