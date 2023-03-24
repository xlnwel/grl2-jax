import os, sys
import argparse
import numpy as np
import jax

from core.ckpt.pickle import save, restore
from core.log import do_logging
from core.typing import AttrDict
from core.utils import configure_gpu
from tools.display import print_dict, print_dict_info
from tools.utils import modify_config, batch_dicts
from tools.yaml_op import load_config, dump
from algo.ma_common.run import Runner

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.log import do_logging
from tools import pkg
from run.utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',
                        type=str,
                        default='',
                        help='directory where checkpoints and "config.yaml" exist')
    parser.add_argument('--filename', '-f',
                        type=str,
                        default='uniform')
    parser.add_argument('--n_runners', '-nr',
                        type=int,
                        default=1)
    parser.add_argument('--n_envs', '-ne',
                        type=int,
                        default=100)
    parser.add_argument('--n_steps', '-ns',
                        type=int,
                        default=1000)

    args = parser.parse_args()

    return args


def main(config, args):
    root_dir = args.directory.replace(config.model_name, '')
    config = modify_config(
        config, 
        root_dir=root_dir
    )
    configure_gpu()

    config.env.n_runners = args.n_runners
    config.env.n_envs = args.n_envs
    runner = Runner(config.env)
    env_stats = runner.env_stats()

    m = pkg.import_module('train', config=config)
    config.buffer = AttrDict(
        type='ac', 
        n_runners=args.n_runners, 
        n_envs=args.n_envs, 
        n_steps=args.n_steps, 
        sample_keys=[], 
    )
    agent = m.build_agent(config, env_stats)

    runner.run(agent, n_steps=args.n_steps, lka_aids=[])
    data = agent.buffer.get_data()
    data = jax.tree_util.tree_map(
        lambda x: x.reshape(-1, 1, *x.shape[2:]), data)

    # filedir = f'data/{config.algorithm}-{config.env.env_name}'
    filedir = f'data'
    filename = config.env.env_name
    prev_data = restore(filedir=filedir, filename=filename)
    if prev_data:
        data = batch_dicts([prev_data, data], np.concatenate)
    do_logging('Data:')
    print_dict_info(data, '\t')
    save(data, filedir=filedir, filename=filename)

    stats = agent.get_raw_stats()
    for k, v in stats.items():
        stats[k] = np.stack(v)
    do_logging('-'*100)
    do_logging('Running Statistics:')
    print_dict_info(stats, '\t')
    
    start = prev_data.obs.shape[0] if prev_data else 0
    end = data.obs.shape[0]

    simple_stats = AttrDict()
    for k, v in stats.items():
        simple_stats[k] = float(np.mean(v))
        simple_stats[f'{k}_std'] = float(np.std(v))
    simple_stats.root_dir = root_dir
    simple_stats.model_name = config.model_name

    stats_path = 'data/stats.yaml'
    all_stats = load_config(stats_path)
    all_stats[f'{start}-{end}'] = simple_stats
    print_dict(all_stats)
    dump(stats_path, all_stats.asdict())


if __name__ == '__main__':
    args = parse_args()
    config = search_for_config(args.directory)
    main(config, args)
