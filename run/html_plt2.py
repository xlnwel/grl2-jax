import argparse
import os, sys
import re
import json
from pathlib import Path
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.typing import ModelPath
from utility.file import search_for_dirs
from utility import yaml_op
from utility.utils import flatten_dict, recursively_remove
from run.html_plt import *


if __name__ == '__main__':
    args = parse_args()
    
    dirs = []
    for p in args.prefix:
        dirs += search_for_dirs(args.directory, p, is_suffix=False)

    for d in dirs:
        # if d.split('/')[-1].startswith('0602'):
        #     print(d.split('/')[-1])
        #     continue
        target_dir = '/'.join([args.target, d])
        print(f'copy from {d} to {target_dir}')
        if not os.path.isdir(target_dir):
            Path(target_dir).mkdir(parents=True)
        
        # define paths
        yaml_path = '/'.join([d, 'config_p0.yaml'])
        json_path = '/'.join([target_dir, 'parameter.json'])
        record_path = '/'.join([d, 'record.txt'])
        process_path = '/'.join([target_dir, 'progress.csv'])
        print('yaml path', yaml_path)
        print('record path', record_path)
        if not os.path.exists(yaml_path) or not os.path.exists(record_path):
            continue
            
        # save config
        config = yaml_op.load_config(yaml_path)
        to_remove_keys = ['root_dir', 'model_name', 'seed']
        config = recursively_remove(config, to_remove_keys)
        config = remove_lists(config)
        config = flatten_dict(config)
        config = rename_env(config)
        config = remove_redundancies(config)


        # save stats
        try:
            data = pd.read_table(record_path, on_bad_lines='skip')
        except:
            print(f'Record path ({record_path}) constains no data')
        if len(data.keys()) == 1:
            data = pd.read_csv(record_path)
        data.rename(columns={
            'latest_expl1': 'latest_nash_conv1', 
            'latest_expl2': 'latest_nash_conv2', 
        }, inplace=True)
        data.to_csv(process_path)
