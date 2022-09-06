import os
from typing import Dict

from core.ckpt.pickle import Checkpoint
from core.typing import AttrDict, ModelPath, dict2AttrDict
from tools.utils import config_attr, set_path
from tools import yaml_op


class YAMLCheckpointBase:
    def save(self):
        raise NotImplementedError

    def restore(self):
        if os.path.exists(self._path):
            config = yaml_op.load(self._path)
            if config is not None:
                config_attr(
                    self, 
                    config, 
                    config_as_attr=False, 
                    private_attr=True
                )


class ParamsCheckpointBase:
    def __init__(self, config, name):
        self.name = name
        self.config = dict2AttrDict(config, to_copy=True)
        self.params: Dict[str, Dict] = AttrDict()
        self._ckpt = Checkpoint(self.config, name=self.name)

    """ Checkpoint Operations """
    def set_weights(self):
        raise NotImplementedError

    def reset_model_path(self, model_path: ModelPath):
        self.config = set_path(self.config, model_path, max_layer=0)
        if self._ckpt is not None:
            self._ckpt.reset_model_path(model_path)

    def restore(self):
        params = self._ckpt.restore(list(self.params))
        self.set_weights(params)
        
    def save(self):
        self._ckpt.save(self.params)
