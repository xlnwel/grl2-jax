import argparse
import os, sys
from pathlib import Path
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.log import do_logging
from core.names import PATH_SPLIT
from tools import yaml_op
from tools.utils import modify_config
from tools.logops import *


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('directory',
            type=str,
            default='.')
  parser.add_argument('--model_rename', '-mr', 
            type=str, 
            nargs='*', 
            default=[])
  parser.add_argument('--new_root', '-nr', 
            type=str, 
            default=None)
  parser.add_argument('--new_date', '-nd', 
            type=str, 
            default=None)
  parser.add_argument('--new_name', '-nn', 
            type=str, 
            default=None)
  parser.add_argument('--last_name', '-ln', 
            type=str, 
            default=['a0', 'dynamics'], 
            nargs='*')
  parser.add_argument('--name', '-n', 
            type=str, 
            default=[], 
            nargs='*')
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
            default=[], 
            nargs='*')
  parser.add_argument('--model', '-m', 
            type=str, 
            default=None, 
            nargs='*')
  parser.add_argument('--copy', '-cp', 
            action='store_true')
  args = parser.parse_args()

  return args


if __name__ == '__main__':
  args = parse_args()
  
  config_name = 'config.yaml' 
  player0_config_name = 'config_p0.yaml' 
  date = get_date(args.date)
  do_logging(f'Loading logs on date: {date}')

  directory = os.path.abspath(args.directory)
  do_logging(f'Directory: {directory}')

  while directory.endswith(PATH_SPLIT):
    directory = directory[:-1]
  
  if directory.startswith(PATH_SPLIT):
    strs = directory.split(PATH_SPLIT)

  search_dir = directory
  level = get_level(search_dir)
  print('Search directory level:', level)
  # all_data = collections.defaultdict(list)
  # for d in yield_dirs(search_dir, args.last_name, is_suffix=False, root_matches=args.name):
  model_rename = args.model_rename
  new_name = args.new_name
  new_date = args.new_date

  for d in fixed_pattern_search(
    search_dir, 
    level=level, 
    env=args.env, 
    algo=args.algo, 
    date=date, 
    model=args.model, 
    final_level=DirLevel.MODEL
  ):
    root, env, algo, date, model = d.rsplit(PATH_SPLIT, 4)
    root = root if args.new_root is None else args.new_root
    prev_dir = os.path.join(root, env, algo, date)
    new_date = args.new_date if args.new_date else date
    new_model = model
    if model_rename:
      for s in model_rename:
        old, new = s.split('=')
        new_model = new_model.replace(old, new)
    new_dir = os.path.join(root, env, algo, new_date, new_model)
    do_logging(f'Moving directory from \n{d} to \n{new_dir}')
    if not os.path.isdir(prev_dir):
      Path(prev_dir).mkdir(parents=True)
    # if os.path.isdir(new_dir):
    #   shutil.rmtree(new_dir)
    if args.copy:
      shutil.copytree(d, new_dir, ignore=shutil.ignore_patterns('src'), dirs_exist_ok=True)
    else:
      os.rename(d, new_dir)
    for d2 in fixed_pattern_search(
      new_dir, 
      level=DirLevel.MODEL, 
      env=args.env, 
      algo=args.algo, 
      date=new_date, 
      model=args.model, 
    ):
      last_name = d2.split(PATH_SPLIT)[-1]
      if not any([last_name.startswith(p) for p in args.last_name]):
        continue
      # load config
      yaml_path = os.path.join(d2, config_name)
      if not os.path.exists(yaml_path):
        new_yaml_path = os.path.join(d2, player0_config_name)
        if os.path.exists(new_yaml_path):
          yaml_path = new_yaml_path
        else:
          do_logging(f'{yaml_path} does not exist', color='magenta')
          continue
      config = yaml_op.load_config(yaml_path)
      model_name = config.model_name
      model_info = config.model_info
      root_dir = config.root_dir
      name = config.name
      if model_rename:
        for s in model_rename:
          old, new = s.split('=')
          model_name = model_name.replace(old, new)
          model_info = model_info.replace(old, new)
          model = model.replace(old, new)
      if new_date:
        model_name = model_name.replace(date, new_date)
        date = new_date
      if args.new_name:
        model_info = model_info.replace(name, new_name)
        name = name
      model_path = [root_dir, model_name]
      config = modify_config(
        config, 
        overwrite_existed_only=True, 
        model_name=model_name, 
        model_info=model_info, 
        date=date, 
        name=name, 
        model_path=model_path, 
        max_layer=3
      )
      do_logging(f'Rewriting config at "{yaml_path}"')
      yaml_op.save_config(config, path=yaml_path)

  do_logging('Move completed')
