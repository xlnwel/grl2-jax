import os, sys
import ray

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.log import setup_logging
from tools import pkg
from run.args import parse_rank_args
from run.utils import search_for_config
from distributed.common.local.controller import Controller


def main(config, payoff_name, n):
  ray.init()

  if config.env.env_name.startswith('grl'):
    config.env.write_video = True
    config.env.dump_frequency = 1
    config.env.write_full_episode_dumps = True
    config.env.render = True
    config.runner.n_steps = config.env.max_episode_steps = 3000

  controller = Controller(config, to_restore=False)
  controller.build_managers_for_evaluation(config)

  controller.evaluate_all(n, payoff_name)

  ray.shutdown()


if __name__ == '__main__':
  args = parse_rank_args()

  config = search_for_config(args.directory, check_duplicates=False)
  setup_logging(args.verbose)

  # set up env_config
  n = args.n_episodes
  if args.n_runners:
    if 'runner' in config:
      config.runner.n_runners = args.n_runners
    config.env.n_runners = args.n_runners
  if args.n_envs:
    config.env.n_envs = args.n_envs
  n = max(config.runner.n_runners * config.env.n_envs, n)

  main(config, args.payoff, n=n)
