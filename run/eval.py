import os, sys
import argparse
import logging
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run.grid_search import GridSearch
from utility.utils import str2bool, pwc
from utility.yaml_op import load_config
from utility.display import assert_colorize


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', '-a',
                        type=str,
                        nargs='*')
    parser.add_argument('--render', '-r',
                        action='store_true')
    parser.add_argument('--trials', '-t',
                        type=int,
                        default=1,
                        help='number of trials')
    parser.add_argument('--prefix', '-p',
                        default='',
                        help='prefix for model dir')
    parser.add_argument('--checkpoint', '-c',
                        type=str,
                        default='',
                        help='checkpoint path to restore')
    args = parser.parse_args()

    return args

def import_main(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.eval import main
    else:
        raise NotImplementedError

    return main
    
def get_arg_file(algorithm):
    if algorithm == 'ppo':
        arg_file = 'algo/ppo/config.yaml'
    else:
        raise NotImplementedError

    return arg_file

if __name__ == '__main__':
    cmd_args = parse_cmd_args()
    algorithm = list(cmd_args.algorithm)
    
    processes = []
    arg_file = get_arg_file(algorithm)
    main = import_main(algorithm)
        
    render = cmd_args.render

    if cmd_args.checkpoint != '':
        config = load_config(arg_file)
        env_config = config['env']
        agent_config = config['agent']
        checkpoint = cmd_args.checkpoint

    main(env_config, agent_config, render)
