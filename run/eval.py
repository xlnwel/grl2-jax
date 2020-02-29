import os, sys
import argparse
import logging
from copy import deepcopy
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.display import pwc
from utility.yaml_op import load_config


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d',
                        type=str,
                        help='directory where checkpoints and "config.yaml" exist')
    parser.add_argument('--render', '-r',
                        action='store_true')
    parser.add_argument('--n_envs', '-ne', default=100)
    parser.add_argument('--n_workers', '-nw', default=1)
    args = parser.parse_args()

    return args

def import_main(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.eval import main
    elif algorithm == 'ppo2':
        from algo.ppo2.eval import main
    elif 'sac' in algorithm:
        from algo.sac.eval import main
    else:
        raise NotImplementedError

    return main


if __name__ == '__main__':
    cmd_args = parse_cmd_args()

    # search for config.yaml
    directory = cmd_args.directory
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
        
    render = cmd_args.render

    # set up env_config
    if render:
        env_config['n_workers'] = env_config['n_envs'] = 1
    else:
        env_config['n_workers'] = cmd_args.n_workers
        env_config['n_envs'] = cmd_args.n_envs
    env_config['seed'] = np.random.randint(1000)
    
    main(env_config, model_config, agent_config, render)
