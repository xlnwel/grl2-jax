import os
import torch

from th.core.typing import ModelPath, dict2AttrDict
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
  def __init__(self, config, name, ckpt_name):
    self.name = name
    self.config = dict2AttrDict(config, to_copy=True)
    self._model_path = ModelPath(self.config.root_dir, self.config.model_name)
    self._suffix_path = os.path.join('params', ckpt_name)
    self._saved_path = os.path.join(*self._model_path, self._suffix_path)

  @property
  def filedir(self):
    return self._saved_path

  """ Checkpoint Operations """
  def set_weights(self):
    raise NotImplementedError

  def reset_model_path(self, model_path: ModelPath):
    self.config = set_path(self.config, model_path, max_layer=0)
    self._model_path = model_path
    self._saved_path = os.path.join(*self._model_path, self._suffix_path)

  def restore(self, modules):
    if os.path.exists(self._saved_path):
      for k, v in modules.items():
        path = os.path.join(self._saved_path, f'{k}.pt')
        v.load_state_dict(torch.load(path))
    
  def save(self, modules):
    if not os.path.exists(self._saved_path):
      os.makedirs(self._saved_path)
    for k, m in modules.items():
      path = os.path.join(self._saved_path, f'{k}.pt')
      torch.save(m.state_dict(), path)
