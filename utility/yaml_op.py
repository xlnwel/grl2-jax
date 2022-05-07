import yaml
from pathlib import Path

from utility.utils import dict2AttrDict, eval_config, flatten_dict


def default_path(filename):
    if filename.startswith('/'):
        return Path(filename)
    else:
        return Path('.') / filename

# load arguments from config.yaml
def load_config(filename='config.yaml', to_attrdict=True):
    if not Path(default_path(filename)).exists():
        raise RuntimeError(f'No configuration is found at: {filename}')
    with open(default_path(filename), 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config = eval_config(config)
            if to_attrdict:
                return dict2AttrDict(config)
            else:
                return config
        except yaml.YAMLError as exc:
            print(f'Fail loading configuration: {filename}')
            print(exc)

# save config to config.yaml
def save_config(config: dict, config_to_update={}, filename='config.yaml'):
    assert isinstance(config, dict)
    
    filepath = default_path(filename)
    if filepath.exists():
        if config_to_update is None:
            config_to_update = load_config(filename)
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.touch()

    with filepath.open('w') as f:
        try:
            config_to_update.update(config)
            yaml.dump(config_to_update, f)
        except yaml.YAMLError as exc:
            print(exc)

def load(path: str):
    with open(path, 'r') as f:
        try:
            data = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(f'Fail loading configuration: {path}')
            print(exc)
            return

    return data

def dump(path: str, **kwargs):
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
