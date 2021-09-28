import os, sys
import logging
import time
import itertools
from multiprocessing import Process

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility import pkg
from utility.utils import deep_update
from run.args import parse_train_args
from run.grid_search import GridSearch
from run.utils import *

logger = logging.getLogger(__name__)


def get_algo_name(algo):
    algo_mapping = {
        'r2d2': 'apex-mrdqn',
        'impala': 'apg-impala',
        'appo': 'apg-ppo',
        'appo2': 'apg-ppo2',
    }
    if algo in algo_mapping:
        return algo_mapping[algo]
    return algo


def load_configs(algo, env):
    algo = get_algo_name(algo)
    if '-' in algo:
        config = get_config(algo.split('-')[-1], env)
        dist_config = get_config(algo, env)
        assert config or dist_config, (config, dist_config)
        assert dist_config, dist_config
        if config == {}:
            config = dist_config
        config = deep_update(config, dist_config)
    else:
        config = get_config(algo, env)
    configs = decompose_config(config)
    configs.agent['algorithm'] = algo
    if env:
        configs.env['name'] = env

    return configs


def set_path(configs, root_dir, model_name):
    for k, v in configs.items():
        assert isinstance(v, dict), (k, v)
        v['root_dir'] = root_dir
        v['model_name'] = model_name
    return configs


if __name__ == '__main__':
    cmd_args = parse_train_args()

    verbose = getattr(logging, cmd_args.verbose.upper())
    logging.basicConfig(level=verbose)
    
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
            configs = load_configs(algo, env)
            model_name = change_config(cmd_args.kwargs, configs, model_name)
            if model_name == '':
                model_name = 'baseline'

            main = pkg.import_main('train', algo)
            if cmd_args.grid_search or cmd_args.trials > 1:
                gs = GridSearch(
                    configs, main, n_trials=cmd_args.trials, 
                    logdir=logdir, dir_prefix=prefix,
                    separate_process=len(algo_env) > 1, 
                    delay=cmd_args.delay)

                if cmd_args.grid_search:
                    processes += gs()
                else:
                    processes += gs()
            else:
                dir_prefix = prefix + '-' if prefix else prefix
                root_dir=f'{logdir}/{dir_prefix}{configs.env["name"]}/{configs.agent["algorithm"]}'
                configs = set_path(configs, root_dir, model_name)
                configs.buffer['dir'] = configs.agent['root_dir'].replace('logs', 'data')
                if len(algo_env) > 1:
                    p = Process(target=main, args=configs)
                    p.start()
                    time.sleep(1)
                    processes.append(p)
                else:
                    main(configs)
    [p.join() for p in processes]
