import argparse
import os, sys, glob
from pathlib import Path
import json
import functools
import multiprocessing
import numpy as np
import pandas as pd
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.log import do_logging
from core.names import PATH_SPLIT
from core.mixin.monitor import is_nonempty_file, merge_data
from tools import yaml_op
from tools.utils import flatten_dict, recursively_remove
from tools.logops import *


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('directory',
            type=str)
  parser.add_argument('--final_level', '-fl', 
            type=int, 
            default=DirLevel.VERSION)
  parser.add_argument('--final_name', '-fn', 
            type=str, 
            default=['a0'], 
            nargs='*')
  parser.add_argument('--target', '-t', 
            type=str, 
            default='~/Documents/html-logs')
  parser.add_argument('--multiprocessing', '-mp', 
            action='store_true')
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


def process_data(data, plt_config):
  # if name != 'happo':
  new_data = {}
  for k in data.keys():
    if 'agent0_first_epoch' in k:
      k2 = k.replace('first', 'last')
      new_key = k.split('agent0_first_epoch/') + ['_diff']
      new_key = ''.join(new_key)
      new_data[new_key] = data[k2] - data[k]
  data = pd.concat([data, pd.DataFrame(new_data)], axis=1)
  normal_rename = plt_config.rename
  normal_filter = [k for k in normal_rename.keys() if k in data]
  normal_columns = data[normal_filter].rename(columns=normal_rename)
  regex_rename = plt_config.regex_rename
  matches = lambda x: any([re.match(name, x) for name in regex_rename])
  regex_filter = [k for k in data.columns if matches(k)]
  regex_columns = data[regex_filter].rename(columns=lambda x: x.split('/', 1)[-1] if matches(x) else x)
  final_data = pd.concat([normal_columns, regex_columns], axis=1)
  final_data['steps'] = data['steps']
  return final_data


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


def convert_data(d, directory, target, plt_config):
  config_name = 'config.yaml' 
  agent0_config_name = 'config_a0.yaml' 
  js_name = 'parameter.json'
  record_name = 'record'
  progress_name = 'progress.csv'

  # load config
  yaml_path = os.path.join(d, config_name)
  if not os.path.exists(yaml_path):
    new_yaml_path = os.path.join(d, agent0_config_name)
    if os.path.exists(new_yaml_path):
      yaml_path = new_yaml_path
    else:
      do_logging(f'{yaml_path} does not exist', color='magenta')
      return
  config = yaml_op.load_config(yaml_path, to_eval=False)
  root_dir = config.root_dir
  model_name = config.model_name
  strs = f'{root_dir}/{model_name}'.split('/')
  for s in strs[::-1]:
    if s.endswith('logs'):
      directory = directory.removesuffix(f'/{s}')
      break
    if directory.endswith(s):
      directory = directory.removesuffix(f'/{s}')

  target_dir = d.replace(directory, target)
  do_logging(f'Copy from "{d}" to "{target_dir}"')
  if not os.path.isdir(target_dir):
    Path(target_dir).mkdir(parents=True)
  assert os.path.isdir(target_dir), target_dir
  
  # define paths
  json_path = os.path.join(target_dir, js_name)
  record_filename = os.path.join(d, record_name)
  record_path = record_filename + '.txt'
  csv_path = os.path.join(target_dir, progress_name)
  # do_logging(f'yaml path: {yaml_path}')
  if not is_nonempty_file(record_path):
    do_logging(f'Bypass {record_path} due to its non-existence', color='magenta')
    return
  # save config
  to_remove_keys = ['root_dir', 'seed']
  seed = config.get('seed', 0)
  config = recursively_remove(config, to_remove_keys)
  config['seed'] = seed
  config = remove_lists(config)
  config = flatten_dict(config)
  config = rename_env(config)
  config = remove_redundancies(config)
  # config = {k: str(v) for k, v in config.items()}
  # config['model_name'] = config['model_name'].split('/')[1]
  config['buffer/sample_keys'] = []

  # save stats
  data = merge_data(record_filename, '.txt')
  data = process_data(data, plt_config)
  for k in ['expl', 'latest_expl', 'nash_conv', 'latest_nash_conv']:
    if k not in data.keys():
      try:
        data[k] = (data[f'{k}1'] + data[f'{k}2']) / 2
      except:
        pass
  
  with open(json_path, 'w') as json_file:
    json.dump(config, json_file)
  data.to_csv(csv_path, index=False)


def transfer_data(args, search_dir, level, plt_config=None):
  date = set()
  env = set()
  algo = set()
  model = set()
  plt_config = yaml_op.load_config(plt_config, to_eval=False)
  for data in plt_config.data:
    if 'date' in data:
      date.add(str(data.date))
    if 'env_suite' in data:
      env.add(f'{data.env_suite}-{data.raw_env_name}' if 'raw_env_name' in data else data.env_suite)
    if 'name' in data:
      algo.add(str(data.name))
    if 'model' in data:
      model.add(data.model)
  do_logging(f'Loading logs with')
  do_logging(f'\tdate={date}')
  do_logging(f'\tenv={env}')
  do_logging(f'\talgo={algo}')
  do_logging(f'\tmodel={model}')
  if args.multiprocessing:
    files = [
      d for d in fixed_pattern_search(
      search_dir, 
      level=level, 
      env=env, 
      algo=algo, 
      date=date, 
      model=model, 
      final_level=DirLevel(args.final_level), 
      final_name=args.final_name
    )]
    pool = multiprocessing.Pool()
    func = functools.partial(convert_data, 
      directory=directory, target=target, plt_config=plt_config)
    pool.map(func, files)
    pool.close()
    pool.join()
  else:
    for d in fixed_pattern_search(
      search_dir, 
      level=level, 
      env=env, 
      algo=algo, 
      date=date, 
      model=model, 
      final_level=DirLevel(args.final_level), 
      final_name=args.final_name
    ):
      convert_data(d, directory, target=target, plt_config=plt_config)


if __name__ == '__main__':
  args = parse_args()

  directory = os.path.abspath(args.directory)
  target = os.path.expanduser(args.target)
  sync_dest = os.path.expanduser(args.target)
  do_logging(f'Directory: {directory}')
  do_logging(f'Target: {target}')

  while directory.endswith(PATH_SPLIT):
    directory = directory[:-1]
  
  search_dir = directory
  print(search_dir)
  level = get_level(search_dir)
  print('Search directory level:', level)

  # transfer_data(args, search_dir, level)
  for f in glob.glob(f'plt_configs/*'):
    if f.endswith('.yaml'):
      transfer_data(args, search_dir, level, f)

  do_logging('Transfer completed')
