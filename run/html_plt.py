import argparse
import os, sys
from enum import Enum
from pathlib import Path
import json
import pandas as pd
import subprocess
import collections
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.log import do_logging
from core.mixin.monitor import is_nonempty_file, merge_data
from tools import yaml_op
from tools.utils import flatten_dict, recursively_remove

ModelPath = collections.namedtuple('ModelPath', 'root_dir model_name')
DataPath = collections.namedtuple('data_path', 'path data')


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
                        default=[], 
                        nargs='*')
    parser.add_argument('--target', '-t', 
                        type=str, 
                        default='~/Documents/html-logs')
    parser.add_argument('--date', '-d', 
                        type=str, 
                        default=[], 
                        nargs='*')
    parser.add_argument('--n_processes', '-np', 
                        type=int, 
                        default=16)
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
    if 'env_name' in config:
        env_name = config['env_name']
    else:
        env_name = config['env/env_name']
    suite = env_name.split('-', 1)[0]
    raw_env_name = env_name.split('-', 1)[1]
    config['env_name'] = env_name
    config['env_suite'] = suite
    config['raw_env_name'] = raw_env_name
    return config


def process_data(data):
    if 'model_error/ego&train-trans' in data:
        k1_err = data[f'model_error/ego-trans']
        train_err = data[f'model_error/train-trans']
        k1_train_err = np.abs(k1_err - train_err)
        data[f'model_error/ego&train-trans'] = k1_train_err
        data[f'model_error/norm_ego&train-trans'] = np.where(train_err != 0,
            k1_train_err / train_err, k1_train_err)
    if 'cos_lka_pi' in data:
        data['cos_lka_mu_diff'] = data['cos_lka_pi'] - data['cos_mu_pi']
        data['cos_mix_mu_diff'] = data['cos_mix_pi'] - data['cos_mu_pi']
    if 'js_mu_lka_diff' in list(data):
        data = data.loc[data['js_mu_lka_diff']<np.log(2)]
        data = data.loc[-np.log(2)<data['js_mu_lka_diff']]
    # if 'lka_ego_score_diff' in list(data):
    #     data.loc[:, 'lka_ego_score_sign'] = np.sign(data['lka_ego_score_diff'])
    return data

def to_csv(env_name, v):
    SCORE = 'metrics/score'
    if v == []:
        return
    scores = [vv.data[SCORE] for vv in v if SCORE in vv.data]
    if scores:
        scores = np.concatenate(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
    print(f'env: {env_name}\tmax={max_score}\tmin={min_score}')
    for csv_path, data in v:
        if SCORE in data:
            data[SCORE] = (data[SCORE] - min_score) / (max_score - min_score)
            print(f'\t{csv_path}. norm score max={np.max(data[SCORE])}, min={np.min(data[SCORE])}')
        data.to_csv(csv_path)


class DirLevel(Enum):
    ROOT = 0
    LOGS = 1
    ENV = 2
    ALGO = 3
    DATE = 4
    MODEL = 5
    SEED = 6
    FINAL = 7
    
    def next(self):
        v = self.value + 1
        if v > 7:
            raise ValueError(f'Enumeration ended')
        return DirLevel(v)



def join_dir_name(filedir, filename):
    return '/'.join([filedir, filename])


def get_level(search_dir, last_prefix):
    for d in os.listdir(search_dir):
        if d.endswith('logs'):
            return DirLevel.ROOT
    all_names = search_dir.split('/')
    last_name = all_names[-1]
    if any([last_name.startswith(p) for p in last_prefix]):
        return DirLevel.FINAL
    if last_name.endswith('logs'):
        return DirLevel.LOGS
    if last_name.startswith('seed'):
        return DirLevel.SEED
    suite = None
    for name in search_dir.split('/'):
        if name.endswith('-logs'):
            suite = name.split('-')[0]
    if last_name.startswith(f'{suite}'):
        return DirLevel.ENV
    if last_name.isdigit():
        return DirLevel.DATE
    # find algorithm name
    algo = None
    model = None
    for i, name in enumerate(all_names):
        if name.isdigit():
            algo = all_names[i-1]
            if len(all_names) == i+2:
                return DirLevel.MODEL
    if algo is None:
        return DirLevel.ALGO
    
    return DirLevel.FINAL


def fixed_pattern_search(search_dir, level=DirLevel.LOGS, matches=[], ignores=[]):
    if level != DirLevel.FINAL:
        if not os.path.isdir(search_dir):
            return []
        for d in os.listdir(search_dir):
            for f in fixed_pattern_search(
                join_dir_name(search_dir, d), 
                level=level.next(), 
                matches=matches, 
                ignores=ignores
            ):
                yield f
        return []
    if matches:
        for m in matches:
            if m in search_dir:
                yield search_dir
        return []
    for i in ignores:
        if i in search_dir:
            return []
    yield search_dir


if __name__ == '__main__':
    args = parse_args()
    
    config_name = 'config.yaml' 
    player0_config_name = 'config_p0.yaml' 
    js_name = 'parameter.json'
    record_name = 'record'
    process_name = 'progress.csv'
    name = args.name
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
        for n in name:
            cmd += ['--include', n]
        do_logging(' '.join(cmd))
        process = subprocess.Popen(cmd)

    search_dir = directory
    level = get_level(search_dir, args.prefix)
    print('Search directory level:', level)
    # all_data = collections.defaultdict(list)
    # for d in yield_dirs(search_dir, args.prefix, is_suffix=False, root_matches=args.name):
    matches = args.name + args.date
    ignores = args.ignore

    for d in fixed_pattern_search(search_dir, level=level, matches=matches, ignores=ignores):
        last_name = d.split('/')[-1]
        if not any([last_name.startswith(p) for p in args.prefix]):
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
            if s.endswith('logs'):
                break
            if directory.endswith(s):
                directory = directory.removesuffix(f'/{s}')

        target_dir = d.replace(directory, target)
        do_logging(f'Copy from {d} to {target_dir}')
        if not os.path.isdir(target_dir):
            Path(target_dir).mkdir(parents=True)
        assert os.path.isdir(target_dir), target_dir
        
        # define paths
        json_path = '/'.join([target_dir, js_name])
        record_filename = '/'.join([d, record_name])
        record_path = record_filename + '.txt'
        csv_path = '/'.join([target_dir, process_name])
        # do_logging(f'yaml path: {yaml_path}')
        if not is_nonempty_file(record_path):
            do_logging(f'Bypass {record_path} due to its non-existence', color='magenta')
            continue
        # save config
        to_remove_keys = ['root_dir', 'seed']
        seed = config.get('seed', 0)
        config = recursively_remove(config, to_remove_keys)
        config['seed'] = seed
        config = remove_lists(config)
        config = flatten_dict(config)
        config = rename_env(config)
        config = remove_redundancies(config)
        config = {k: str(v) for k, v in config.items()}
        # config['model_name'] = config['model_name'].split('/')[1]
        config['buffer/sample_keys'] = []

        # save stats
        data = merge_data(record_filename, '.txt')
        data = process_data(data)
        for k in ['expl', 'latest_expl', 'nash_conv', 'latest_nash_conv']:
            if k not in data.keys():
                try:
                    data[k] = (data[f'{k}1'] + data[f'{k}2']) / 2
                except:
                    pass

        with open(json_path, 'w') as json_file:
            json.dump(config, json_file)
        data.to_csv(csv_path, index=False)
        # all_data[config.env_name].append(DataPath(csv_path, data))
        # to_csv(config.env_name, DataPath(csv_path, data))

    # for k, v in all_data.items():
    #     to_csv(k, v)
    # all_data.clear()
        
    if process is not None:
        do_logging('Waiting for rsync to complete...')
        process.wait()

    do_logging('Transfer completed')
