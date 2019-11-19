import os, sys
import argparse
import logging
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.display import pwc
from utility.utils import str2bool
from utility.yaml_op import load_config
from utility.display import assert_colorize


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d',
                        type=str,
                        help='directory where checkpoints and "config.yaml" exist')
    parser.add_argument('--render', '-r',
                        action='store_true')
    args = parser.parse_args()

    return args

def import_main(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.eval import main
    else:
        raise NotImplementedError

    return main


if __name__ == '__main__':
    cmd_args = parse_cmd_args()

    directory = cmd_args.directory
    config_file = None
    for root, _, files in os.walk(directory):
        for f in files:
            if f == 'config.yaml' and config_file is None:
                config_file = os.path.join(root, f)
                break
            elif f =='config.yaml' and config_file is not None:
                pwc(f'Get multiple "config.yaml": "{config_file}" and "{os.path.join(root, f)}"')
                sys.exit()

    config = load_config(config_file)
    algorithm = config['algorithm']
    main = import_main(algorithm)
        
    render = cmd_args.render
    env_config = dict(name=config['env_name'])

    main(env_config, config, render)
