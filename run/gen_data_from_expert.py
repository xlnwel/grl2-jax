import os
import argparse
import collections
import numpy as np

from tools.pickle import restore
from core.typing import dict2AttrDict
from tools.yaml_op import load_config
from tools.utils import batch_dicts
from run.gen_data import save_data, save_stats


Stats = collections.namedtuple('Stats', 'score filename')


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_dir',
            type=str,
            default='datasets')
  parser.add_argument('--filedir',
            type=str,
            default='data')
  parser.add_argument('--n_elites', '-n',
            type=int,
            default=3)
  parser.add_argument('--n_runners', '-nr',
            type=int,
            default=1)
  parser.add_argument('--n_envs', '-ne',
            type=int,
            default=100)
  parser.add_argument('--n_steps', '-ns',
            type=int,
            default=1000)
  parser.add_argument('--from_algo',
            action='store_true', 
            default=False)

  args = parser.parse_args()

  return args


YAML_SUFFIX = '.yaml'
def main(args):
  dataset_dir = args.dataset_dir
  for filename in os.listdir(dataset_dir):
    if not filename.endswith(YAML_SUFFIX):
      continue
    filename = filename.replace(YAML_SUFFIX, '')
    stats_path = os.path.join(dataset_dir, filename)
    stats = load_config(stats_path)
    stats = {k: v for k, v in stats.items() if isinstance(v, dict)}
    
    elite_stats_keys = sorted(stats, key=lambda x: stats[x].score, reverse=True)[:args.n_elites]
    elite_stats = dict2AttrDict({k: stats[k] for k in elite_stats_keys})
    elite_filedir = args.filedir
    elite_stats_path = f'{elite_filedir}/{filename}{YAML_SUFFIX}'
    data = batch_dicts([
      restore(filedir=dataset_dir, filename=f)
      for f in elite_stats_keys
    ], np.concatenate)
    data_filename = '-'.join(filename.split('-')[:2])
    save_data(data, data_filename, filedir=elite_filedir)
    save_stats(elite_stats, elite_stats_path)


if __name__ == '__main__':
  args = parse_args()

  main(args)
