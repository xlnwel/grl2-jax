import os
from typing import Dict
import cloudpickle

from core.log import do_logging
from core.typing import ModelPath


def save(data, filedir, filename):
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    
    path = f'{filedir}/{filename}.pkl'
    with open(path, 'wb') as f:
        cloudpickle.dump(data, f)
    do_logging(f'Saving parameters in "{path}"', level='pwt', backtrack=3)

def restore(filedir, filename):
    path = f'{filedir}/{filename}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = cloudpickle.load(f)
        do_logging(f'Restoring parameters from "{path}"', level='pwt', backtrack=3)
    else:
        data = {}
        do_logging(f'No such file: {path}', level='pwt', backtrack=3)

    return data

def set_weights_for_agent(
    agent, 
    model: ModelPath, 
    filename='params'
):
    weights = restore('/'.join(model), filename)
    agent.set_weights(weights)


class Checkpoint:
    def __init__(
        self, 
        config, 
        models: Dict, 
        name='ckpt'
    ):
        self._model_path = ModelPath(
            config.get('root_dir', None), 
            config.get('model_name', None)
        )
        self._models = models
        self._name = name

    """ Save & Restore Model """
    def reset_model_path(self, model_path: ModelPath):
        self._model_path = model_path

    def save(self):
        path = '/'.join([
            *self._model_path, 
            self._name
        ])
        for k, m in self._models.items():
            save(m.get_weights(), path, k)

    def restore(self):
        path = '/'.join([
            *self._model_path, 
            self._name
        ])
        for k, m in self._models.items():
            weights = restore(path, k)
            if weights:
                m.set_weights(weights)

    def get_filedir(self):
        return '/'.join([
            *self._model_path, 
            self._name
        ])
