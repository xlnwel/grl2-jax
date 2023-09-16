import argparse
import os, sys
import json
import itertools
import pandas as pd
import collections
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.log import do_logging
from core.mixin.monitor import is_nonempty_file, merge_data
from tools.file import yield_dirs
from tools import yaml_op

ModelPath = collections.namedtuple('ModelPath', 'root_dir model_name')
Data = collections.namedtuple('data', 'mean std')


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
  parser.add_argument('--key', '-k', 
            type=str, 
            nargs='*', 
            default=['model_info'])
  parser.add_argument('--var', '-v', 
            nargs='*', 
            type=str)
  parser.add_argument('--date', '-d', 
            type=str, 
            default=None)
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
    

if __name__ == '__main__':
  args = parse_args()
  
  config_name = 'config.yaml' 
  record_name = 'record'
  date = args.date
  do_logging(f'Loading logs on date: {date}')

  directory = os.path.abspath(args.directory)
  do_logging(f'Directory: {directory}')

  while directory.endswith('/'):
    directory = directory[:-1]
  
  if directory.startswith('/'):
    strs = directory.split('/')

  search_dir = directory
  
  data = collections.defaultdict(lambda: collections.defaultdict(list))
  for k, v in itertools.product(args.key, args.var):
    for d in yield_dirs(search_dir, args.prefix, is_suffix=False, root_matches=args.name):
      if date is not None and date not in d:
        do_logging(f'Bypass directory "{d}" due to mismatch date')
        continue
        
      if args.ignore and args.ignore in d:
        do_logging(f'Bypass directory "{d}" as it contains ignore pattern "{args.ignore}"')
        continue

      # load config
      yaml_path = '/'.join([d, config_name])
      if not os.path.exists(yaml_path):
        do_logging(f'{yaml_path} does not exist', color='magenta')
        continue
      config = yaml_op.load_config(yaml_path)
      name = config[k]

      # save stats
      record_filename = '/'.join([d, record_name])
      df = merge_data(record_filename, '.txt')
      if v in df:
        data[v][name].append(df[v].to_numpy())
      # all_data[config.env_name].append(DataPath(csv_path, data))
      # to_csv(config.env_name, DataPath(csv_path, data))

  max_key_len = max([len(k) for k in data.values()])
  max_key_len = max(max_key_len, 25)
  for title, d in data.items():
    print('title', title)
    print('-'*60)
    for k, v in d.items():
      def get_val_str(val):
        return f"{val:5.2g}" if hasattr(val, "__float__") else str(val)
      v = np.concatenate(v)
      print(f' | {k:<{max_key_len}} | mean={get_val_str(v.mean()):<3s}\tstd={get_val_str(v.std()):<3s} |')
    print('-'*60)