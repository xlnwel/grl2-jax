import os, sys
from datetime import datetime

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
        'ssppo': 'sync-ppo',
    }
    if algo in algo_mapping:
        return algo_mapping[algo]
    return algo


def _get_algo_env_config(cmd_args):
    algos = list(cmd_args.algorithms)
    envs = list(cmd_args.environments)
    configs = list(cmd_args.configs)
    if len(algos) < len(configs):
        assert len(algos) == 1, algos
        configurations = [read_config(algos[0], envs[0], c) for c in configs]
        algos = [c.algorithm for c in configurations]
    if len(envs) < len(configs):
        assert len(envs) == 1, envs
        envs = [envs[0] for _ in configs]
    if len(configs) == 0:
        configs = [get_filename_with_env(e) for e in envs]
    assert len(algos) == len(envs) == len(configs), (algos, envs, configs)
    algo_env_config = list(zip(algos, envs, configs))
    
    return algo_env_config


def _grid_search(config, main, cmd_args):
    gs = GridSearch(
        config, 
        main, 
        n_trials=cmd_args.trials, 
        logdir=cmd_args.logdir, 
        dir_prefix=cmd_args.prefix,
        separate_process=True, 
        delay=cmd_args.delay, 
        multiprocess=cmd_args.multiprocess
    )

    processes = []
    processes += gs(
        lr=[1e-1, 1e-2, 1e-3], 
        entropy_coef=[1e-3, 1e-4], 
        # seed=[0, 1, 2]
    )
    [p.join() for p in processes]


def _run_with_configs(cmd_args):
    algo_env_config = _get_algo_env_config(cmd_args)

    logdir = cmd_args.logdir
    prefix = cmd_args.prefix
    if cmd_args.model_name[:4].isdigit():
        raw_model_name = cmd_args.model_name
    else:
        dt = datetime.now().strftime("%m%d")
        raw_model_name = f'{dt}-{cmd_args.model_name}'

    configs = []
    for algo, env, config in algo_env_config:
        algo = _get_algo_name(algo)
        config = load_config_with_algo_env(algo, env, config)
        model_name = change_config(cmd_args.kwargs, config, raw_model_name)
        if model_name == '':
            model_name = 'baseline'

        if cmd_args.seed is not None:
            model_name = f'{model_name}-seed={cmd_args.seed}'

        main = pkg.import_main('train', algo)
        
        dir_prefix = prefix + '-' if prefix else prefix
        root_dir = f'{logdir}/{dir_prefix}{config.env.env_name}/{config.algorithm}'
        config = modify_config(
            config, 
            root_dir=root_dir, 
            model_name=model_name, 
            seed=cmd_args.seed
        )
        config.buffer['root_dir'] = config.buffer['root_dir'].replace('logs', 'data')

        config['info'] = cmd_args.info
        configs.append(config)

    if cmd_args.grid_search or cmd_args.trials > 1:
        assert len(configs) == 1, 'No support for multi-agent grid search.'
        _grid_search(config, main, cmd_args)
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
        configs = [search_for_config(d) for d in cmd_args.directory]
        main = pkg.import_main('train', config=configs[0])
    else:
        _run_with_configs(cmd_args)
