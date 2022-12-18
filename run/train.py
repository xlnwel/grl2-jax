import os, sys
os.environ["XLA_FLAGS"] = '--xla_dump_to=/tmp/foo'
# os.environ['XLA_FLAGS'] = "--xla_gpu_force_compilation_parallelism=1"

from datetime import datetime
import numpy as np

# try:
#     from tensorflow.python.compiler.mlcompute import mlcompute
#     mlcompute.set_mlc_device(device_name='gpu')
#     print("----------M1----------")
# except:
#     print("----------Not M1-----------")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.log import setup_logging, do_logging
from tools import pkg
from tools.utils import modify_config
from run.args import parse_train_args
from run.grid_search import GridSearch
from run.utils import *
from tools.timer import get_current_datetime


def _get_algo_name(algo):
    # shortcuts for distributed algorithms
    algo_mapping = {
        'sppo': 'sync-ppo',
    }
    if algo in algo_mapping:
        return algo_mapping[algo]
    return algo


def _get_algo_env_config(cmd_args):
    algos = cmd_args.algorithms
    env = cmd_args.environment
    configs = list(cmd_args.configs)
    if len(algos) < len(configs):
        envs = [env for _ in configs]
    else:
        envs = [env for _ in algos]
    if len(algos) < len(envs):
        assert len(algos) == 1, algos
        algos = [algos[0] for _ in envs]
    assert len(algos) == len(envs), (algos, envs)

    if len(configs) == 0:
        configs = [get_filename_with_env(env) for env in envs]
    elif len(configs) < len(envs):
        configs = [configs[0] for _ in envs]
    assert len(algos) == len(envs) == len(configs), (algos, envs, configs)

    if len(algos) == 1 and cmd_args.n_agents > 1:
        algos = [algos[0] for _ in range(cmd_args.n_agents)]
        envs = [envs[0] for _ in range(cmd_args.n_agents)]
        configs = [configs[0] for _ in range(cmd_args.n_agents)]
    else:
        cmd_args.n_agents = len(algos)
    assert len(algos) == len(envs) == len(configs) == cmd_args.n_agents, (algos, envs, configs, cmd_args.n_agents)
    
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
        kw_dict={
            # 'optimizer:lr': np.linspace(1e-4, 1e-3, 2),
            # 'meta_opt:lr': np.linspace(1e-4, 1e-3, 2),
            # 'value_coef:default': [.5, 1]
        }, 
    )
    [p.join() for p in processes]


def config_info(config, infos, model_name):
    if len(infos) > 1:
        for i, info in enumerate(infos):
            config[f'info{i}'] = info
    elif len(infos) == 1:
        config['info'] = infos[0]
    else:
        config['info'] = None
    info = model_name.split('/', 1)[0]
    if not config['info'] or config['info'] in info:
        config['info'] = info
    else:
        config['info'] = config['info'] + info
    return config


def setup_configs(cmd_args, algo_env_config):
    logdir = cmd_args.logdir
    prefix = cmd_args.prefix
    
    if cmd_args.model_name[:4].isdigit():
        date = cmd_args.model_name[:4]
        if len(cmd_args.model_name) == 4:
            raw_model_name = ''
        else:
            assert cmd_args.model_name[4] in ['-', '_', '/'], cmd_args.model_name[4]
            raw_model_name = f'{cmd_args.model_name[5:]}'
    else:
        date = datetime.now().strftime("%m%d")
        raw_model_name = cmd_args.model_name

    model_name = get_model_name_from_kw_string(
        cmd_args.kwargs, raw_model_name)

    configs = []
    kwidx = cmd_args.kwidx
    if kwidx == []:
        kwidx = list(range(len(algo_env_config)))
    current_time = str(get_current_datetime())
    for i, (algo, env, config) in enumerate(algo_env_config):
        do_logging(f'Setup configs for algo({algo}) and env({env})', color='yellow')
        algo = _get_algo_name(algo)
        config = load_config_with_algo_env(algo, env, config)
        if i in kwidx:
            change_config_with_kw_string(cmd_args.kwargs, config)
        if model_name == '':
            model_name = 'baseline'

        if not cmd_args.grid_search and not cmd_args.trials > 1:
            model_name = f'{model_name}/seed={cmd_args.seed}'
        
        dir_prefix = prefix + '-' if prefix else prefix
        root_dir = f'{logdir}/{dir_prefix}{config.env.env_name}/{config.algorithm}'
        config = modify_config(
            config, 
            max_layer=1, 
            root_dir=root_dir, 
            model_name=f'{date}/{model_name}', 
            seed=cmd_args.seed
        )
        config.date = date
        config.buffer.root_dir = config.buffer.root_dir.replace('logs', 'data')

        config = config_info(config, cmd_args.info, model_name)
        config.launch_time = current_time
        configs.append(config)
    
    if len(configs) < cmd_args.n_agents:
        assert len(configs) == 1, len(configs)
        configs[0]['n_agents'] = cmd_args.n_agents
        configs = [dict2AttrDict(configs[0], to_copy=True) 
            for _ in range(cmd_args.n_agents)]
    elif len(configs) == cmd_args.n_agents:
        configs = [dict2AttrDict(c, to_copy=True) for c in configs]
    else:
        raise NotImplementedError

    if cmd_args.n_agents > 1:
        for i, c in enumerate(configs):
            modify_config(
                c, 
                aid=i,
                seed=i*100 if cmd_args.seed is None else cmd_args.seed+i*100
            )
    
    return configs


def _run_with_configs(cmd_args):
    algo_env_config = _get_algo_env_config(cmd_args)
    main = pkg.import_main('train', cmd_args.algorithms[0])

    configs = setup_configs(cmd_args, algo_env_config)

    if cmd_args.grid_search or cmd_args.trials > 1:
        assert len(configs) == 1, 'No support for multi-agent grid search.'
        _grid_search(configs[0], main, cmd_args)
    else:
        do_logging(configs, level='info')
        main(configs)


if __name__ == '__main__':
    cmd_args = parse_train_args()

    setup_logging(cmd_args.verbose)
    if not (cmd_args.grid_search and cmd_args.multiprocess) and cmd_args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{cmd_args.gpu}"

    processes = []
    if cmd_args.directory != '':
        configs = [search_for_config(d) for d in cmd_args.directory]
        main = pkg.import_main('train', config=configs[0])
        main(configs)
    else:
        _run_with_configs(cmd_args)
