import argparse
from datetime import datetime


def parse_train_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--algorithms', '-a', 
    type=str, 
    nargs='*', 
    default=[])
  parser.add_argument(
    '--environment', '-e',
    type=str)
  parser.add_argument(
    '--configs', '-c',
    type=str,
    nargs='*',
    default=[])
  parser.add_argument(
    '--info', '-i',
    type=str,
    default=None)
  parser.add_argument(
    '--directory', '-d',
    type=str,
    default='',
    nargs='*', 
    help='directory where checkpoints and "config.yaml" exist')
  parser.add_argument(
    '--kwargs', '-kw',
    type=str,
    nargs='*',
    default=[],
    help='key-values in config.yaml required to be overwrite, '
      'we use "#" in key to specify configs, e.g., "0,1/key=value" overwrites key-value in the 0th and 1st configs only;'
      '":" in key to specify a nested config, e.g., "policy_opt:lr=1e-3" overwrites lr in policy_opt only;'
      '"," in value to denote a list of elements')
  parser.add_argument(
    '--kwidx', '-ki',
    type=int,
    nargs='*',
    default=[],
    help="key-values in config.yaml needed to be overwrite")
  parser.add_argument(
    '--trials', '-t',
    type=int,
    default=1,
    help='number of trials')
  """ Arguments for logdir """
  parser.add_argument(
    '--prefix', '-p',
    default='',
    help='directory prefix')
  dt = datetime.now()
  parser.add_argument(
    '--model_name', '-n',
    default=f'{dt.month:02d}{dt.day:02d}',
    help='model name')
  parser.add_argument(
    '--logdir', '-ld',
    type=str,
    default='logs',
    help='the logging directory. By default, all training data will be stored in logdir/env/algo/model_name')
  parser.add_argument(
    '--grid_search', '-gs',
    action='store_true')
  parser.add_argument(
    '--delay',
    default=1,
    type=int)
  parser.add_argument(
    '--verbose', '-v',
    type=str,
    default='warning',
    help="the verbose level for python's built-in logging")
  parser.add_argument(
    '--gpu',
    type=str,
    default=None)
  parser.add_argument(
    '--seed', '-s',
    type=int,
    default=None)
  parser.add_argument(
    '--multiprocess', '-mp', 
    action='store_true')
  parser.add_argument(
    '--n_agents', '-na', 
    type=int,
    default=1)
  parser.add_argument(
    '--train_entry', '-te', 
    type=str, 
    default='train')
  parser.add_argument(
    '--new_kw', 
    type=str,
    nargs='*',
    default=[],
    help='Add new key-values to config.yaml')
  parser.add_argument(
    '--exploiter', 
    action='store_true')
  args = parser.parse_args()

  return args


def parse_eval_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'directory',
    type=str,
    help='directory where checkpoints and "config.yaml" exist',
    nargs='*')
  parser.add_argument(
    '--record', '-r', 
    action='store_true')
  parser.add_argument(
    '--video_len', '-vl', 
    type=int, 
    default=None)
  parser.add_argument(
    '--n_episodes', '-n', 
    type=int, 
    default=1)
  parser.add_argument(
    '--n_envs', '-ne', 
    type=int, 
    default=0)
  parser.add_argument(
    '--n_runners', '-nr', 
    type=int, 
    default=0)
  parser.add_argument(
    '--size', '-s', 
    nargs='+', 
    type=int, 
    default=None)
  parser.add_argument(
    '--save', 
    action='store_true')
  parser.add_argument(
    '--fps', 
    type=int, 
    default=30)
  parser.add_argument(
    '--info', '-i', 
    type=str, 
    default='')
  parser.add_argument(
    '--verbose', '-v', 
    type=str, 
    default='warning')
  args = parser.parse_args()

  return args


def parse_rank_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'directory',
    type=str,
    help='directory where checkpoints and "config.yaml" exist')
  parser.add_argument(
    '--payoff', '-p', 
    type=str,
    default='eval_payoff', 
    help='payoff name')
  parser.add_argument(
    '--n_episodes', '-n', 
    type=int, 
    default=1000)
  parser.add_argument(
    '--n_envs', '-ne', 
    type=int, 
    default=None)
  parser.add_argument(
    '--n_runners', '-nw', 
    type=int, 
    default=None)
  parser.add_argument(
    '--verbose', '-v', 
    type=str, 
    default='warning')
  args = parser.parse_args()

  return args
