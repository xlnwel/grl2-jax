import argparse
import os, sys, glob
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools import yaml_op

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--directory',
                      type=str,
                      default='/Users/chenxw/work/Polixir/cache/WEB_ROM/configs')
  args = parser.parse_args()

  return args


def select_data(config, plt_config, new_config=None):
  if new_config is None or f'{new_config}-data' not in plt_config:
    config['DATA_SELECT'] = plt_config['data']
  else:
    config['DATA_SELECT'] = plt_config[f'{new_config}-data']
  config['DATA_SELECT_PROPERTY'] = []
  for d in config['DATA_SELECT']:
    prop = {}
    for k, v in d.items():
      if isinstance(v, str) and '*' in v:
        prop[k] = {"manual": True}
      else:
        prop[k] = {"manual": False}
      
    config['DATA_SELECT_PROPERTY'].append(prop)

  return config


def rename_data(config, plt_config):
  config['DATA_KEY_RENAME_CONFIG'] = plt_config.rename

  return config


def plot_data(config, plt_config):
  names = plt_config.rename
  plot_xy = []
  for m in names.values():
    if ['steps', m] not in plot_xy:
      plot_xy.append(['steps', m])
  plot_xy += plt_config.get('plot_xy', [])
  config['PLOTTING_XY'] = plot_xy

  return config


def plot_config(config, plt_config):
  if 'plt' in plt_config:
    config.update(plt_config.plt)
  return config


def generate_config(args, f):
  plt_config = yaml_op.load_config(f, to_eval=True)
  old_config = str(plt_config['config']['old'])
  new_configs = plt_config['config']['new']
  if not isinstance(new_configs, (list, tuple)):
    new_configs = [new_configs]
  for new_config in new_configs:
    new_config = str(new_config)
    target_config_path = os.path.join(args.directory, new_config)
    if os.path.exists(target_config_path):
      config_path = target_config_path
    else:
      config_path = os.path.join(args.directory, old_config)
    config_path = os.path.expanduser(config_path)
    with open(config_path, 'r') as f:
      config = json.load(f)
    config = rename_data(config, plt_config)
    config = plot_data(config, plt_config)
    config = select_data(config, plt_config, new_config)
    config = plot_config(config, plt_config)
    print(config['DATA_KEY_RENAME_CONFIG'])
    target_config_path = os.path.join(args.directory, new_config)
    target_config_path = os.path.expanduser(target_config_path)
    with open(target_config_path, 'w') as f:
      json.dump(config, f)

    print(f'A new config generated at {target_config_path}')


if __name__ == '__main__':
  args = parse_args()
  
  for f in glob.glob('plt_configs/*'):
    generate_config(args, f)
