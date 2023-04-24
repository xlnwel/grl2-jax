import os, sys
import argparse
import numpy as np
import jax

from core.ckpt.pickle import save
from core.log import do_logging
from core.typing import AttrDict
from core.utils import configure_gpu
from tools.display import print_dict, print_dict_info
from tools.utils import modify_config
from tools.yaml_op import load_config, dump
from tools import pkg, yaml_op
from algo.ma_common.run import Runner
from run.utils import *


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',
                        type=str,
                        default='',
                        help='directory where checkpoints and "config.yaml" exist')
    parser.add_argument('--filedir',
                        type=str,
                        default='datasets')
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


def load_config_from_filedir(filedir, from_algo=False):
    assert filedir.split('/')[-1].startswith('a'), filedir
    
    names = filedir.split('/')
    algo = names[-5]
    if from_algo:
        algo = names[-5]
        env = names[-6]
        env_suite, env_name = env.split('-')
        filename = env_suite if env_suite == 'ma_mujoco' else env_name
        # load config from algo/*/configs/*.yaml
        config = load_config_with_algo_env(algo, env, filename)
    else:
        # load config from the logging directory
        config = search_for_config(filedir)
    root_dir = '/'.join(names[:-5])
    model_name = '/'.join(names[-5:])
    config = modify_config(
        config, 
        root_dir=root_dir, 
        model_name=model_name, 
        name=algo, 
        seed=int(names[-2][-1])
    )
    # print_dict(config)
    # yaml_path = f'{filedir}/config'
    # yaml_op.save_config(config, path=yaml_path)

    return config


def collect_data(config, n_runners, n_envs, n_steps):
    config.env.n_runners = n_runners
    config.env.n_envs = n_envs
    runner = Runner(config.env)
    env_stats = runner.env_stats()

    m = pkg.import_module('train', config=config)
    config.buffer = AttrDict(
        type='ac', 
        n_runners=n_runners, 
        n_envs=n_envs, 
        n_steps=n_steps, 
        sample_keys=config.buffer.sample_keys, 
    )
    agent = m.build_agent(
        config, 
        env_stats, 
        save_monitor_stats_to_disk=False, 
        save_config=False
    )

    runner.run(agent, n_steps=n_steps, lka_aids=[])
    data = agent.buffer.get_data()
    data = jax.tree_util.tree_map(
        lambda x: x.reshape(-1, 1, *x.shape[2:]), data)
    stats = dict2AttrDict(agent.get_raw_stats())

    return data, stats


def save_data(data, filename, filedir):
    do_logging('Data:')
    print_dict_info(data, '\t')
    save(data, filedir=filedir, filename=filename)


def get_stats_path(filename, filedir):
    return f'{filedir}/{filename}-stats.yaml'


def summarize_stats(stats, config):
    for k, v in stats.items():
        stats[k] = np.stack(v)
    # print_dict_info(stats, '\t')

    simple_stats = AttrDict()
    for k in ['score', 'epslen']:
        v = stats[k]
        simple_stats[k] = float(np.mean(v))
        simple_stats[f'{k}_std'] = float(np.std(v))
    simple_stats.root_dir = config.root_dir
    simple_stats.model_name = config.model_name

    filename = config.env.env_name
    env_suite, env_name = filename.split('-')
    simple_stats.algorithm = config.algorithm
    simple_stats.env_suite = env_suite
    simple_stats.env_name = env_name

    return simple_stats


def save_stats(all_stats, stats_path):
    do_logging('Running Statistics:')
    print_dict(all_stats, '\t')
    dump(stats_path, all_stats)


def main(config, args):
    configure_gpu()
    data, stats = collect_data(config, args.n_runners, args.n_envs, args.n_steps)
    simple_stats = summarize_stats(stats, config)

    filename = config.env.env_name
    stats_path = get_stats_path(filename, filedir=args.filedir)
    all_stats = load_config(stats_path)
    start = all_stats.get('data_size', 0)
    end = start + data['obs'].shape[0]
    data_filename = f'{filename}-{start}-{end}'
    all_stats[data_filename] = simple_stats
    all_stats.data_size = end

    save_stats(all_stats, stats_path)
    do_logging('-'*100)
    save_data(data, filename=data_filename, filedir=args.filedir)
    

if __name__ == '__main__':
    args = parse_args()
    config = load_config_from_filedir(args.directory, args.from_algo)

    main(config, args)
