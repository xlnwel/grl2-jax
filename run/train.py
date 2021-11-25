import os, sys
import numpy as np
import logging
import time
import itertools
from multiprocessing import Process

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.log import do_logging
from utility import pkg
from run.args import parse_train_args
from run.grid_search import GridSearch
from run.utils import *


def _get_algo_name(algo):
    algo_mapping = {
        'r2d2': 'apex-mrdqn',
        'impala': 'apg-impala',
        'appo': 'apg-ppo',
        'appo2': 'apg-ppo2',
    }
    if algo in algo_mapping:
        return algo_mapping[algo]
    return algo


def set_path(config, root_dir, model_name):
    config['root_dir'] = root_dir
    config['model_name'] = model_name
    for v in config.values():
        if not isinstance(v, dict):
            continue
        v['root_dir'] = root_dir
        v['model_name'] = model_name
    return config


if __name__ == '__main__':
    cmd_args = parse_train_args()

    verbose = getattr(logging, cmd_args.verbose.upper())
    logging.basicConfig(
        level=verbose, 
        format=f'%(asctime)s: %(levelname)s: %(name)s: %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    processes = []
    if cmd_args.directory != '':
        load_and_run(cmd_args.directory)
    else:
        algorithm = list(cmd_args.algorithm)
        environment = list(cmd_args.environment)
        algo_env = list(itertools.product(algorithm, environment))

        logdir = cmd_args.logdir
        prefix = cmd_args.prefix
        model_name = cmd_args.model_name

        for algo, env in algo_env:
            algo = _get_algo_name(algo)
            config = load_configs_with_algo_env(algo, env)
            model_name = change_config(cmd_args.kwargs, config, model_name)
            if model_name == '':
                model_name = 'baseline'

            main = pkg.import_main('train', algo)
            if cmd_args.grid_search or cmd_args.trials > 1:
                gs = GridSearch(
                    config, main, n_trials=cmd_args.trials, 
                    logdir=logdir, dir_prefix=prefix,
                    separate_process=len(algo_env) > 1, 
                    delay=cmd_args.delay)

                if cmd_args.grid_search:
                    processes += gs(batch_size=[32, 64, 128, 256], lr=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
                    # processes += gs(batch_size=[32, 64])
                else:
                    processes += gs()
            else:
                dir_prefix = prefix + '-' if prefix else prefix
                root_dir=f'{logdir}/{dir_prefix}{config.env.name}/{config.algorithm}'
                config = set_path(config, root_dir, model_name)
                config.buffer['root_dir'] = config.buffer['root_dir'].replace('logs', 'data')
                do_logging(config, level='DEBUG')
                if len(algo_env) > 1:
                    p = Process(target=main, args=(config,))
                    p.start()
                    time.sleep(1)
                    processes.append(p)
                else:
                    main(config)
    [p.join() for p in processes]
