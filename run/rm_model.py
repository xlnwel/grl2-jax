import argparse
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.log import do_logging
from tools.file import rm
from tools.logops import *


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('directory',
            type=str,
            default='/Users/chenxw/Documents/html-logs/')
  parser.add_argument('--model', '-m', 
            type=str, 
            nargs='*', 
            default=None)
  parser.add_argument('--date', '-d', 
            type=str, 
            default=None, 
            nargs='*')
  parser.add_argument('--env', '-e', 
            type=str, 
            default=None, 
            nargs='*')
  parser.add_argument('--algo', '-a', 
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

  while directory.endswith('/'):
    directory = directory[:-1]
  
  level = get_level(directory)
  print('Search directory level:', level)

  for d in fixed_pattern_search(
    directory, 
    level=level, 
    env=args.env, 
    algo=args.algo, 
    date=date, 
    final_level=DirLevel.MODEL
  ):
    root, env, algo, date, model = d.rsplit('/', 4)
    if args.env and not any([e in env for e in args.env]):
      continue
    if args.algo and algo not in args.algo:
      continue
    if args.date and date not in args.date:
      continue
    if args.model is None:
      do_logging(f'Removing model in {d}')
      rm(d)
      continue
    for m in args.model:
      if m in model:
        do_logging(f'Removing model in {d}')
        rm(d)

  do_logging('Removal completed')
