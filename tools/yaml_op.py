import yaml
from pathlib import Path

from core.log import do_logging
from core.typing import AttrDict, dict2AttrDict
from tools.utils import eval_config, flatten_dict


def default_path(path):
    if path.startswith('/'):
        return Path(path)
    else:
        return Path('.') / path

# load arguments from config.yaml
def load_config(path='config', to_attrdict=True):
    if not path.endswith('.yaml'):
        path = path + '.yaml'
    path = default_path(path)
    if not path.exists():
        do_logging(f'No configuration is found at: {path}', level='pwc', backtrack=4)
        return AttrDict()
    with open(path, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config = eval_config(config)
            if to_attrdict:
                return dict2AttrDict(config)
            else:
                return config
        except yaml.YAMLError as exc:
            do_logging(f'Fail loading configuration: {path}', level='pwc', backtrack=4)
            print(exc)

# save config to config.yaml
def save_config(config: dict, config_to_update={}, path='config.yaml'):
    assert isinstance(config, dict)
    if not path.endswith('.yaml'):
        path = path + '.yaml'
    
    path = default_path(path)
    if path.exists():
        if config_to_update is None:
            config_to_update = load_config(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    with path.open('w') as f:
        try:
            config_to_update.update(config)
            yaml.dump(config_to_update, f)
        except yaml.YAMLError as exc:
            print(exc)

def load(path: str):
    if not Path(path).exists():
        return {}
    with open(path, 'r') as f:
        try:
            data = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(f'Fail loading configuration: {path}')
            print(exc)
            return {}

    return data

def dump(path: str, kwargs):
    with open(path, 'w') as f:
        try:
            yaml.dump(kwargs, f)
        except yaml.YAMLError as exc:
            print(exc)

def yaml2json(yaml_path, json_path, flatten=False):
    config = load_config(yaml_path)
    if flatten:
        config = flatten_dict(config)
    import json
    with open(json_path, 'w') as json_file:
        json.dump(config, json_file)

    return config
