import os, sys

# try:
#     from tensorflow.python.compiler.mlcompute import mlcompute
#     mlcompute.set_mlc_device(device_name='gpu')
#     print("----------M1----------")
# except:
#     print("----------Not M1-----------")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.log import setup_logging, do_logging
from utility import pkg
from utility.utils import modify_config
from run.args import parse_train_args
from run.grid_search import GridSearch
from run.utils import *


def _get_algo_name(algo):
    # shortcuts for distributed algorithms
    algo_mapping = {
        'ssppo': 'sync_sim-ppo',
    }
    if algo in algo_mapping:
        return algo_mapping[algo]
    return algo


def _get_algo_env_config(cmd_args):
    algorithms = list(cmd_args.algorithms)
    environments = list(cmd_args.environments)
    configs = list(cmd_args.configs)
    if len(algorithms) < len(configs):
        assert len(algorithms) == 1, algorithms
        algorithms = algorithms * len(configs)
    if len(environments) < len(configs):
        assert len(environments) == 1, environments
        environments = environments * len(configs)
    assert len(algorithms) == len(environments) == len(configs), (algorithms, environments, configs)
    algo_env_config = list(zip(algorithms, environments, configs))
    
    return algo_env_config


def run_all(cmd_args):
    algo_env_config = _get_algo_env_config(cmd_args)

    logdir = cmd_args.logdir
    prefix = cmd_args.prefix
    model_name = cmd_args.model_name

    configs = []
    for algo, env, config in algo_env_config:
        algo = _get_algo_name(algo)
        config = load_config_with_algo_env(algo, env, config)
        model_name = change_config(cmd_args.kwargs, config, model_name)
        if model_name == '':
            model_name = 'baseline'

        main = pkg.import_main('train', algo)
        
        dir_prefix = prefix + '-' if prefix else prefix
        root_dir = f'{logdir}/{dir_prefix}{config.env.env_name}/{config.algorithm}'
        config = modify_config(config, root_dir=root_dir, model_name=model_name)
        config.buffer['root_dir'] = config.buffer['root_dir'].replace('logs', 'data')

        configs.append(config)

    if cmd_args.grid_search or cmd_args.trials > 1:
        assert len(configs) == 1, len(configs)
        gs = GridSearch(
            config, main, n_trials=cmd_args.trials, 
            logdir=logdir, dir_prefix=prefix,
            separate_process=True, 
            delay=cmd_args.delay)

        processes = []
        if cmd_args.grid_search:
            processes += gs()
        else:
            processes += gs()
        [p.join() for p in processes]
    else:
        do_logging(config, level='DEBUG')
        main(configs)


if __name__ == '__main__':
    cmd_args = parse_train_args()

    setup_logging(cmd_args.verbose)
    if cmd_args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]= f"{cmd_args.gpu}"

    processes = []
    if cmd_args.directory != '':
        load_and_run(cmd_args.directory)
    else:
        run_all(cmd_args)
