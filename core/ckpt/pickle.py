import os
import cloudpickle

from core.elements.agent import Agent
from core.elements.model import Model
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
    agent: Agent, 
    model: ModelPath, 
    filename='params.pkl'
):
    weights = restore('/'.join(model), filename)
    agent.set_weights(weights)


class Checkpoint:
    def __init__(self, config, model: Model, name='ckpt'):
        self._model_path = ModelPath(config.get('root_dir', None), config.get('model_name', None))
        self._model = model
        self._name = name

    """ Save & Restore Model """
    def reset_model_path(self, model_path: ModelPath):
        self._model_path = model_path
        self.setup_checkpoint(force=True)
        self._has_ckpt = True

    def save(self):
        if self._has_ckpt:
            save(self._model.get_weights(), self._model_path, self._name)
        else:
            raise RuntimeError(
                'Cannot perform <save> as either root_dir or model_name was not specified at initialization')

    def restore(self):
        if self._has_ckpt:
            weights = restore(self._model_path, self._name)
            self._model.set_weights(weights)
        else:
            raise RuntimeError(
                'Cannot perform <restore> as either root_dir or model_name was not specified at initialization')
