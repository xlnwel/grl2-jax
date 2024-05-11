import argparse
import os, sys
import pandas as pd
import collections

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.log import do_logging
from core.names import PATH_SPLIT
from core.mixin.monitor import is_nonempty_file, merge_data
from tools import plot, yaml_op
from tools.display import print_dict_info
from tools.file import mkdir
from tools.logops import *


ModelPath = collections.namedtuple('ModelPath', 'root_dir model_name')
DataPath = collections.namedtuple('data_path', 'path data')


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('directory',
            type=str,
            default='.')
  parser.add_argument('--target_dir', '-td', 
            type=str, 
            default='~/Documents/plots')
  parser.add_argument('--title', '-t', 
            type=str, 
            default=None)
  parser.add_argument('--last_name', '-ln', 
            type=str, 
            default='seed')
  parser.add_argument('--x', '-x', 
            type=str, 
            nargs='*', 
            default=['steps'])
  parser.add_argument('--y', '-y', 
            type=str, 
            default='metrics/score')
  parser.add_argument('--env', '-e', 
            type=str, 
            default=None, 
            nargs='*')
  parser.add_argument('--algo', '-a', 
            type=str, 
            default=None, 
            nargs='*')
  parser.add_argument('--date', '-d', 
            type=str, 
            default=None, 
            nargs='*')
  parser.add_argument('--model', '-m', 
            type=str, 
            default=None, 
            nargs='*')
  parser.add_argument('--figsize', '-f', 
            type=int, 
            default=None, 
            nargs='*')
  args = parser.parse_args()

  return args


def load_dataset(search_dir, args, date):
  level = get_level(search_dir)
  record_name = 'record'
  print('Search directory level:', level)

  dataset = collections.defaultdict(list)
  levels = collections.defaultdict(set)
  for d in fixed_pattern_search(
    search_dir, 
    level=level, 
    env=args.env, 
    algo=args.algo, 
    date=date, 
    model=args.model, 
    final_level=DirLevel.MODEL
  ):
    env, _, _, model = d.split(PATH_SPLIT)[-4:]
    for dd in fixed_pattern_search(
      d, 
      level=DirLevel.MODEL, 
      env=args.env, 
      algo=args.algo, 
      date=date, 
      model=args.model, 
      final_name=args.last_name
    ):
      print('directory', dd)
      # load config
      yaml_path = os.path.join(dd, config_name)
      if not os.path.exists(yaml_path):
        do_logging(f'{yaml_path} does not exist', color='magenta')
        continue
      config = yaml_op.load_config(yaml_path)

      # define paths
      record_filename = os.path.join(dd, record_name)
      record_path = record_filename + '.txt'
      # do_logging(f'yaml path: {yaml_path}')
      if not is_nonempty_file(record_path):
        do_logging(f'Bypass {record_path} due to its non-existence', color='magenta')
        continue

      do_logging(f'Loading dataset from {record_path}')
      # save stats
      data = merge_data(record_filename, '.txt')
      if data is not None:
        data['legend'] = [config.model_info] * len(data.index)
        dataset[env].append(data)
        levels['model'].add(model)
  for env in dataset:
    dataset[env] = pd.concat(dataset[env])

  return dataset, levels


if __name__ == '__main__':
  args = parse_args()
  
  config_name = 'config.yaml' 
  date = get_date(args.date)
  do_logging(f'Loading logs on date: {date}')

  directory = os.path.abspath(args.directory)
  target_dir = os.path.expanduser(args.target_dir)
  do_logging(f'Directory: {directory}')
  do_logging(f'Target: {target_dir}')

  while directory.endswith(PATH_SPLIT):
    directory = directory[:-1]

  dataset, levels = load_dataset(directory, args, date)
  envs = list(dataset.keys())
  fig, axs = plot.setup_figure(n=len(envs))
  if len(envs) == 1:
    axs = [axs]
  else:
    axs = axs.flat
  for i, (env, ax) in enumerate(zip(envs, axs)):
    data = dataset[env]
    plot.lineplot_dataframe(
      data=data, title=env, y=args.y, fig=fig, ax=ax)
    ax.get_legend().remove()
  fig.legend(labels=levels['model'], loc='lower left', bbox_to_anchor=(.1, .9))
  mkdir(target_dir)
  fig_path = os.path.join(target_dir, f'{args.title}.png')
  fig.savefig(fig_path, bbox_inches='tight')
  print(f'File saved at "{fig_path}"')

  do_logging('Plotting completed')
