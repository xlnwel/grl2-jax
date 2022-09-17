import os
import cloudpickle

from core.log import do_logging
from core.typing import ModelPath
from tools import yaml_op


def save(data, filedir, filename):
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    
    path = f'{filedir}/{filename}.pkl'
    with open(path, 'wb') as f:
        cloudpickle.dump(data, f)
    do_logging(f'Saving parameters in "{path}"', level='info', time=True, backtrack=3)

def restore(filedir, filename):
    path = f'{filedir}/{filename}.pkl'
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                data = cloudpickle.load(f)
                do_logging(f'Restoring parameters from "{path}"', backtrack=3)
        except Exception as e:
            do_logging(f'Failing restoring parameters from {path}', backtrack=3)
    else:
        data = {}
        do_logging(f'No such file: {path}', backtrack=3)

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
        name='ckpt'
    ):
        if 'root_dir' in config and 'model_name' in config:
            self._model_path = ModelPath(
                config.root_dir, config.model_name)
        else:
            self._model_path = None
        self._name = name
        if self._model_path is not None:
            self.filename_path = self.get_filedir('filename')
            self.filenames = yaml_op.load_config(self.filename_path)
        else:
            self.filename_path = None
            self.filenames = None
        if self.filenames:
            self.filenames = self.filenames['ckpt']
        else:
            self.filenames = []
        
    """ Save & Restore Model """
    def reset_model_path(self, model_path: ModelPath):
        self._model_path = model_path
        self.filename_path = self.get_filedir('filename')

    def save(self, params):
        path = self.get_filedir()
        if self.filenames != list(params):
            self.filenames = list(params)
            yaml_op.save_config({
                'ckpt': self.filenames}, path=self.filename_path)
        for k, v in params.items():
            save(v, path, k)

    def restore(self, filenames=None):
        if filenames is None:
            filenames = self.filenames
        params = {}
        if filenames:
            path = self.get_filedir()
            for filename in self.filenames:
                weights = restore(path, filename)
                if weights:
                    params[filename] = weights 
        return params

    def get_filedir(self, *args):
        assert self._model_path is not None, self._model_path
        return '/'.join([*self._model_path, self._name, *args])
