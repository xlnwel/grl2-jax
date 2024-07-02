import argparse
import time
from datetime import datetime
import subprocess

from run.args import parse_train_args
from tools.file import load_config_with_algo_env


def _args2str(args: argparse.Namespace):
  cmd_args = []
  for k, v in vars(args).items():
    if not v or v is None:
      continue
    cmd_args.append(f'--{k}')
    if isinstance(v, (int, float)):
      v = str(v)
    if isinstance(v, str):
      v = [v]
    cmd_args += v
  return cmd_args


if __name__ == '__main__':
  args = parse_train_args()

  cmd = ['python', 'run/train.py', *_args2str(args)]
  print('Command', ' '.join(cmd))
  process = subprocess.Popen(cmd)
  pid = process.pid

  config = load_config_with_algo_env(
    args.algorithms[0], args.environments[0], args.configs[0])
  period = config.controller.store_period + 60
  while True:
    time.sleep(period)
    with open('check.txt', 'r') as f:
      x = f.read()
    diff = datetime.now() - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    if diff.seconds > period:
      subprocess.Popen(['kill', '-9', f'{pid}'])

      process = subprocess.Popen(cmd)
      pid = process.pid
