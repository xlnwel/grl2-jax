import warnings
import argparse
warnings.filterwarnings("ignore")

import os, sys
os.environ['XLA_FLAGS'] = "--xla_gpu_force_compilation_parallelism=1"

import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.log import do_logging
from core.typing import dict2AttrDict
from core.utils import configure_gpu
from tools.ray_setup import sigint_shutdown_ray
from tools.run import simple_evaluate
from tools import pkg
from env.func import create_env
from run.args import parse_eval_args
from run.utils import setup_configs, compute_episodes, build_agents


def main(configs, n, render):
  config = dict2AttrDict(configs[0])
  use_ray = config.env.get('n_runners', 0) > 1
  if use_ray:
    import ray
    ray.init()
    sigint_shutdown_ray()

  algo_name = config.algorithm

  try:
    make_env = pkg.import_module('env', algo=algo_name, place=-1).make_env
  except Exception as e:
    make_env = None
  
  env = create_env(config.env, env_fn=make_env)
  env_stats = env.stats()
  agents = build_agents(configs, env_stats)

  print('start evaluation')
  start = time.time()
  scores, epslens = simple_evaluate(env, agents, n, render)

  do_logging(f'After running {n} episodes', color='cyan')
  for i, score in enumerate(scores):
    do_logging(f'\tAgent{i} Score: {np.mean(score):.3g}', color='cyan')
  do_logging(f'\tEpslen: {np.mean(epslens):.3g}', color='cyan')
  do_logging(f'\tTime: {time.time()-start:.3g}', color='cyan')

  if use_ray:
    ray.shutdown()

  return scores, epslens


def parse_eval_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'directory',
    type=str,
    help='directory where checkpoints and "config.yaml" exist',
    nargs='*')
  parser.add_argument(
    '--render', '-r', 
    action='store_true')
  parser.add_argument(
    '--n_episodes', '-n', 
    type=int, 
    default=1)
  parser.add_argument(
    '--n_envs', '-ne', 
    type=int, 
    default=1)
  parser.add_argument(
    '--n_runners', '-nr', 
    type=int, 
    default=1)
  args = parser.parse_args()

  return args


if __name__ == '__main__':
  args = parse_eval_args()

  configure_gpu()
  configs = setup_configs(args)
  n = compute_episodes(args)

  main(configs, n=n, render=args.render)
