import os
from typing import Dict, Sequence

from core.typing import AttrDict, ModelPath, dict2AttrDict
from tools.utils import config_attr, set_path
from tools import yaml_op, pickle


class CheckpointBase:
  def retrieve(self):
    data = {v[1:]: getattr(self, v) for v in vars(self) if v.startswith('_')}
    return data

  def load(self, data):
    config_attr(self, data, filter_dict=False, private_attr=True, check_overwrite=True)


class YAMLCheckpointBase(CheckpointBase):
  def __init__(self, dir, name):
    self.dir = dir
    self.name = name
    self.path = os.path.join(self.dir, f'{self.name}.yaml')

  def save(self):
    data = self.retrieve()
    yaml_op.dump(self.path, **data, atomic=True)

  def restore(self):
    config = yaml_op.load(self.path)
    self.load(config)


class PickleCheckpointBase(CheckpointBase):
  def __init__(self, dir, name):
    self.dir = dir
    self.name = name

  def save(self):
    data = self.retrieve()
    pickle.save(data, filedir=self.dir, filename=self.name, name=self.name)

  def restore(self):
    data = pickle.restore(filedir=self.dir, filename=self.name, name=self.name)
    self.load(data)


class ParamsCheckpointBase:
  def __init__(self, config, name, ckpt_name):
    self.name = name
    self.config = dict2AttrDict(config, to_copy=True)
    self.params: Dict[str, Dict] = AttrDict()
    self._ckpt = pickle.Checkpoint(self.config, name=f'params/{ckpt_name}')

  @property
  def filedir(self):
    return self._ckpt.get_filedir()

  """ Checkpoint Operations """
  def set_weights(self):
    raise NotImplementedError

  def reset_model_path(self, model_path: ModelPath):
    self.config = set_path(self.config, model_path, max_layer=0)
    if self._ckpt is not None:
      self._ckpt.reset_model_path(model_path)

  def restore(self):
    params = self._ckpt.restore()
    self.set_weights(params)
    return params
    
  def save(self, params=None):
    if params is None:
      params = self.params
    self._ckpt.save(params)
