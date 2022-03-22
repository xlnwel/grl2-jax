import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.log import setup_logging
from core.tf_config import *
from utility import pkg
from run.args import parse_rank_args
from run.utils import search_for_config


def main(config, payoff_name, n):
    pass

if __name__ == '__main__':
    args = parse_rank_args()

    config = search_for_config(args.directory, check_duplicates=False)
    setup_logging(args.verbose)

    try:
        main = pkg.import_main('rank', config=config)
    except Exception as e:
        print('Default main is used for ranking due to :', e)

    silence_tf_logs()
    configure_gpu()
    configure_precision(config.precision)

    # set up env_config
    n = args.n_episodes
    if args.n_workers:
        if 'runner' in config:
            config.runner.n_runners = args.n_workers
        config.env.n_workers = args.n_workers
    if args.n_envs:
        config.env.n_envs = args.n_envs
    n = max(config.runner.n_runners * config.env.n_envs, n)

    main(config, args.payoff, n=n)
