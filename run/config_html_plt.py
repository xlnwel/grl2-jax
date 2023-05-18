import argparse
import os, sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.log import do_logging
from tools.display import print_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str,
                        default='/Users/chenxw/work/Polixir/cache/WEB_ROM/configs')
    parser.add_argument('--config', '-c', 
                        type=str, 
                        default='tmp')
    parser.add_argument('--target', '-t', 
                        type=str, 
                        default='tmp')
    parser.add_argument('--date', '-d', 
                        type=str, 
                        default=None, 
                        nargs='*')
    args = parser.parse_args()

    return args


def select_data(config):
    config['DATA_SELECT'] = [
        # {'date': "517", 'env_suite': 'mujoco'},
        {'date': "519", 'env_suite': 'grf'}, 
        # {'date': "518", 'env_suite': 'grf'}, 
        # {'date': "516", 'env_suite': 'grf', 'name': 'happo'}, 
        # {'date': "516", 'env_suite': 'grf', 'name': 'happo_lka2'}
    ]
    return config


def rename_data(config):
    rename_config = {}

    rename_config['metrics/score'] = 'score'
    rename_config['agent0_first_epoch/entropy'] = 'entropy'
    rename_config['agent0_first_epoch/clip_frac'] = 'first_epoch/clip_frac'
    rename_config['agent0_last_epoch/clip_frac'] = 'last_epoch/clip_frac'
    rename_config['clip_frac_diff'] = 'clip_frac_diff'
    # rename_config['agent0_first_epoch/adv_ratio_pp'] = 'first_epoch/adv_ratio_pp'
    # rename_config['agent0_first_epoch/adv_ratio_np'] = 'first_epoch/adv_ratio_np'
    # rename_config['agent0_first_epoch/adv_ratio_pn'] = 'first_epoch/adv_ratio_pn'
    # rename_config['agent0_first_epoch/adv_ratio_nn'] = 'first_epoch/adv_ratio_nn'
    # rename_config['agent0_last_epoch/adv_ratio_pp'] = 'last_epoch/adv_ratio_pp'
    # rename_config['agent0_last_epoch/adv_ratio_np'] = 'last_epoch/adv_ratio_np'
    # rename_config['agent0_last_epoch/adv_ratio_pn'] = 'last_epoch/adv_ratio_pn'
    # rename_config['agent0_last_epoch/adv_ratio_nn'] = 'last_epoch/adv_ratio_nn'
    rename_config['adv_ratio_pp_diff'] = 'adv_ratio_pp_diff'
    rename_config['adv_ratio_np_diff'] = 'adv_ratio_np_diff'
    rename_config['adv_ratio_pn_diff'] = 'adv_ratio_pn_diff'
    rename_config['adv_ratio_nn_diff'] = 'adv_ratio_nn_diff'
    rename_config['sample_reg_loss_diff'] = 'sample_reg_loss_diff'
    rename_config['agent0_first_epoch/pos_sample_reg_grads/max'] = 'first_epoch/pos_sample_reg_grads/max'
    rename_config['agent0_first_epoch/pos_sample_reg_grads/min'] = 'first_epoch/pos_sample_reg_grads/min'
    rename_config['agent0_first_epoch/sample_reg_loss'] = 'first_epoch/sample_reg_loss'

    config['DATA_KEY_RENAME_CONFIG'] = rename_config

    return config

def plot_data(config):
    plot_xy = []
    for m in config['DATA_KEY_RENAME_CONFIG'].values():
        plot_xy.append(['steps', m])
    
    config['PLOTTING_XY'] = plot_xy
    return config

if __name__ == '__main__':
    args = parse_args()
    
    config_path = os.path.join(args.directory, args.config) 
    with open(config_path, 'r') as f:
        config = json.load(f)
    print_dict(config)
    config = rename_data(config)
    plot_xy = plot_data(config)

    config = select_data(config)
    if args.target is None:
        target_config_path = config_path
    else:
        target_config_path = os.path.join(args.directory, args.target)

    with open(target_config_path, 'w') as f:
        json.dump(config, f)
    
    do_logging(f'New config generated at {target_config_path}')
