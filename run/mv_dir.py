import argparse
import os, sys
from enum import Enum
from pathlib import Path
import collections
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.log import do_logging
from tools import yaml_op
from tools.utils import modify_config

ModelPath = collections.namedtuple('ModelPath', 'root_dir model_name')
DataPath = collections.namedtuple('data_path', 'path data')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',
                        type=str,
                        default='.')
    parser.add_argument('--model_rename', '-mr', 
                        type=str, 
                        nargs='*', 
                        default=[])
    parser.add_argument('--new_root', '-nr', 
                        type=str, 
                        default=None)
    parser.add_argument('--new_date', '-nd', 
                        type=str, 
                        default=None)
    parser.add_argument('--new_name', '-nn', 
                        type=str, 
                        default=None)
    parser.add_argument('--prefix', '-p', 
                        type=str, 
                        default=['a0', 'dynamics'], 
                        nargs='*')
    parser.add_argument('--name', '-n', 
                        type=str, 
                        default=[], 
                        nargs='*')
    parser.add_argument('--date', '-d', 
                        type=str, 
                        default=[], 
                        nargs='*')
    parser.add_argument('--ignore', '-i',
                        type=str, 
                        default=[])
    parser.add_argument('--copy', '-cp', 
                        action='store_true')
    args = parser.parse_args()

    return args


class DirLevel(Enum):
    ROOT = 0
    LOGS = 1
    ENV = 2
    ALGO = 3
    DATE = 4
    MODEL = 5
    SEED = 6
    FINAL = 7
    
    def next(self):
        v = self.value + 1
        if v > 7:
            raise ValueError(f'Enumeration ended')
        return DirLevel(v)



def join_dir_name(filedir, filename):
    return '/'.join([filedir, filename])


def get_level(search_dir, last_prefix):
    for d in os.listdir(search_dir):
        if d.endswith('logs'):
            return DirLevel.ROOT
    all_names = search_dir.split('/')
    last_name = all_names[-1]
    if any([last_name.startswith(p) for p in last_prefix]):
        return DirLevel.FINAL
    if last_name.endswith('logs'):
        return DirLevel.LOGS
    if last_name.startswith('seed'):
        return DirLevel.SEED
    suite = None
    for name in search_dir.split('/'):
        if name.endswith('-logs'):
            suite = name.split('-')[0]
    if last_name.startswith(f'{suite}'):
        return DirLevel.ENV
    if last_name.isdigit():
        return DirLevel.DATE
    # find algorithm name
    algo = None
    model = None
    for i, name in enumerate(all_names):
        if name.isdigit():
            algo = all_names[i-1]
            if len(all_names) == i+2:
                return DirLevel.MODEL
    if algo is None:
        return DirLevel.ALGO
    
    return DirLevel.FINAL


def fixed_pattern_search(search_dir, level=DirLevel.LOGS, matches=[], ignores=[], target_level=DirLevel.MODEL):
    if level != target_level:
        if not os.path.isdir(search_dir):
            return []
        for d in os.listdir(search_dir):
            for f in fixed_pattern_search(
                join_dir_name(search_dir, d), 
                level=level.next(), 
                matches=matches, 
                ignores=ignores, 
                target_level=target_level
            ):
                yield f
        return []
    if matches:
        for m in matches:
            if m in search_dir:
                yield search_dir
        return []
    for i in ignores:
        if i in search_dir:
            return []
    yield search_dir


if __name__ == '__main__':
    args = parse_args()
    
    config_name = 'config.yaml' 
    player0_config_name = 'config_p0.yaml' 
    name = args.name
    date = args.date
    do_logging(f'Loading logs on date: {args.date}')

    directory = os.path.abspath(args.directory)
    do_logging(f'Directory: {directory}')

    while directory.endswith('/'):
        directory = directory[:-1]
    
    if directory.startswith('/'):
        strs = directory.split('/')

    search_dir = directory
    level = get_level(search_dir, args.prefix)
    print('Search directory level:', level)
    # all_data = collections.defaultdict(list)
    # for d in yield_dirs(search_dir, args.prefix, is_suffix=False, root_matches=args.name):
    matches = args.name + args.date
    ignores = args.ignore
    model_rename = args.model_rename
    new_name = args.new_name
    new_date = args.new_date

    for d in fixed_pattern_search(search_dir, level=level, matches=matches, ignores=ignores, target_level=DirLevel.MODEL):
        root, env, algo, date, model = d.rsplit('/', 4)
        root = root if args.new_root is None else args.new_root
        prev_dir = '/'.join([root, env, algo, date])
        new_dir = '/'.join([root, env, algo, date, model])
        do_logging(f'Moving directory from {d} to {new_dir}')
        if not os.path.isdir(prev_dir):
            Path(prev_dir).mkdir(parents=True)
        # if os.path.isdir(new_dir):
        #     shutil.rmtree(new_dir)
        if args.copy:
            shutil.copytree(d, new_dir, ignore=shutil.ignore_patterns('src'), dirs_exist_ok=True)
        else:
            os.rename(d, new_dir)
        for d2 in fixed_pattern_search(new_dir, level=DirLevel.MODEL, matches=matches, ignores=ignores, target_level=DirLevel.FINAL):
            last_name = d2.split('/')[-1]
            if not any([last_name.startswith(p) for p in args.prefix]):
                continue
            # load config
            yaml_path = '/'.join([d2, config_name])
            if not os.path.exists(yaml_path):
                new_yaml_path = '/'.join([d2, player0_config_name])
                if os.path.exists(new_yaml_path):
                    yaml_path = new_yaml_path
                else:
                    do_logging(f'{yaml_path} does not exist', color='magenta')
                    continue
            config = yaml_op.load_config(yaml_path)
            model_name = config.model_name
            model_info = config.model_info
            root_dir = config.root_dir
            name = config.name
            if model_rename:
                for s in model_rename:
                    old, new = s.split('=')
                    model_name = model_name.replace(old, new)
                    model_info = model_info.replace(old, new)
                    model = model.replace(old, new)
            if new_date:
                model_name = model_name.replace(date, new_date)
                date = new_date
            if args.new_name:
                model_info = model_info.replace(name, new_name)
                name = name
            model_path = [root_dir, model_name]
            config = modify_config(
                config, 
                overwrite_existed_only=True, 
                model_name=model_name, 
                model_info=model_info, 
                date=date, 
                name=new_name, 
                model_path=model_path, 
                max_layer=3
            )
            do_logging(f'Rewriting config at "{yaml_path}"')
            yaml_op.save_config(config, path=yaml_path)

    do_logging('Move completed')
