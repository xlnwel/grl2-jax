import os, sys
import importlib
import argparse
import logging
from copy import deepcopy
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.display import pwc
from utility.yaml_op import load_config
from run.pkg import get_package


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d',
                        type=str,
                        help='directory where checkpoints and "config.yaml" exist')
    parser.add_argument('--record', '-r',
                        action='store_true')
    parser.add_argument('--n_episodes', '-n', type=int, default=1)
    parser.add_argument('--n_envs', '-ne', type=int, default=0)
    parser.add_argument('--n_workers', '-nw', type=int, default=0)
    parser.add_argument('--size', '-s', nargs='+', type=int, default=[128, 128])
    args = parser.parse_args()

    return args

def import_main(algorithm):
    pkg = get_package(algorithm, -1)
    m = importlib.import_module(f'{pkg}.eval')

    return m.main


if __name__ == '__main__':
    args = parse_cmd_args()

    # search for config.yaml
    directory = args.directory
    config_file = None
    for root, _, files in os.walk(directory):
        for f in files:
            if 'src' in root:
                break
            elif f.endswith('config.yaml') and config_file is None:
                config_file = os.path.join(root, f)
                break
            elif f.endswith('config.yaml') and config_file is not None:
                pwc(f'Get multiple "config.yaml": "{config_file}" and "{os.path.join(root, f)}"')
                sys.exit()

    # load respective config
    config = load_config(config_file)
    env_config = config['env']
    model_config = config['model']
    agent_config = config['agent']

    # get the main function
    algorithm = config['agent']['algorithm']
    main = import_main(algorithm)
        
    record = args.record

    # set up env_config
    n = args.n_episodes
    if args.n_workers:
        env_config['n_workers'] = args.n_workers
    if args.n_envs:
        env_config['n_envs'] = args.n_envs
    n = max(args.n_workers * args.n_envs, n)
    env_config['seed'] = np.random.randint(1000)
    
    main(env_config, model_config, agent_config, n=n, record=record, size=tuple(args.size))
