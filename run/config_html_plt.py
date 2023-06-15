import argparse
import os, sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.log import do_logging
from tools.display import print_dict
from tools import yaml_op


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str,
                        default='/Users/chenxw/work/Polixir/cache/WEB_ROM/configs')
    parser.add_argument('--config', '-c', 
                        type=str, 
                        default=None)
    parser.add_argument('--target', '-t', 
                        type=str, 
                        default=None)
    parser.add_argument('--date', '-d', 
                        type=str, 
                        default=None, 
                        nargs='*')
    args = parser.parse_args()

    return args


def select_data(config, plt_config):
    config['DATA_SELECT'] = plt_config.data
    config['DATA_SELECT_PROPERTY'] = [{}] * len(config['DATA_SELECT'])

    return config


def rename_data(config, plt_config):
    config['DATA_KEY_RENAME_CONFIG'] = plt_config.rename

    return config

def plot_data(config, plt_config):
    names = plt_config.rename
    plot_xy = []
    for m in names.values():
        plot_xy.append(['steps', m])
    plot_xy += plt_config.get('plot_xy', [])
    config['PLOTTING_XY'] = plot_xy

    return config

if __name__ == '__main__':
    args = parse_args()
    
    plt_config = yaml_op.load_config('run/plt_config', to_eval=False)
    old_config = plt_config.config.old if args.config is None else args.config
    new_config = plt_config.config.new if args.target is None else args.target
    
    config_path = os.path.join(args.directory, old_config) 
    with open(config_path, 'r') as f:
        config = json.load(f)
    config = rename_data(config, plt_config)
    plot_xy = plot_data(config, plt_config)
    config = select_data(config, plt_config)
    
    target_config_path = os.path.join(args.directory, new_config)
    with open(target_config_path, 'w') as f:
        json.dump(config, f)
    
    do_logging(f'New config generated at {target_config_path}')
