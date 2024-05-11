import argparse
import os, sys
from enum import Enum
import collections

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.log import do_logging
from core.names import PATH_SPLIT
from tools import yaml_op

ModelPath = collections.namedtuple('ModelPath', 'root_dir model_name')
DataPath = collections.namedtuple('data_path', 'path data')


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('directory',
            type=str,
            default='.')
  parser.add_argument('--key', '-k', 
            type=str, 
            default='.')
  parser.add_argument('--value', '-v', 
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
  parser.add_argument('--date', '-d', 
            type=str, 
            default=[], 
            nargs='*')
  parser.add_argument('--ignore', '-i',
            type=str, 
            default=[])
  args = parser.parse_args()

  return args


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
  return os.path.join(filedir, filename)


def get_level(search_dir, last_prefix):
  for d in os.listdir(search_dir):
    if d.endswith('logs'):
      return DirLevel.ROOT
  all_names = search_dir.split(PATH_SPLIT)
  last_name = all_names[-1]
  if any([last_name.startswith(p) for p in last_prefix]):
    return DirLevel.FINAL
  if last_name.endswith('logs'):
    return DirLevel.LOGS
  if last_name.startswith('seed'):
    return DirLevel.SEED
  suite = None
  for name in search_dir.split(PATH_SPLIT):
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
  do_logging(f'Directory: {directory}')

  while directory.endswith(PATH_SPLIT):
    directory = directory[:-1]
  
  if directory.startswith(PATH_SPLIT):
    strs = directory.split(PATH_SPLIT)

  search_dir = directory
  level = get_level(search_dir, args.prefix)
  print('Search directory level:', level)
  # all_data = collections.defaultdict(list)
  # for d in yield_dirs(search_dir, args.prefix, is_suffix=False, root_matches=args.name):
  matches = args.name + args.date
  ignores = args.ignore

  for d in fixed_pattern_search(search_dir, level=level, matches=matches, ignores=ignores):
    last_name = d.split(PATH_SPLIT)[-1]
    if not any([last_name.startswith(p) for p in args.prefix]):
      continue
    # load config
    yaml_path = os.path.join(d, config_name)
    if not os.path.exists(yaml_path):
      new_yaml_path = os.path.join(d, player0_config_name)
      if os.path.exists(new_yaml_path):
        yaml_path = new_yaml_path
      else:
        do_logging(f'{yaml_path} does not exist', color='magenta')
        continue
    config = yaml_op.load_config(yaml_path)
    if args.key:
      config[args.key] = args.value
    do_logging(f'Rewrite config at "{yaml_path}"')
    yaml_op.save_config(config, path=yaml_path)

  do_logging('Rewrite completed')
