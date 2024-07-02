import logging
import functools
import collections
import torch
import numpy as np

from tools.log import do_logging

DataFormat = collections.namedtuple('DataFormat', ('shape', 'dtype'))
logger = logging.getLogger(__name__)


def _transform_data(data_format):
  for k, v in data_format.items():
    if k == 'state':
      data_format[k] = (DataFormat(*vv[:2]) for vv in v)
    else:
      data_format[k] = DataFormat(*v[:2])
  return data_format


class Dataset:
  def __init__(
    self, 
    buffer, 
    data_format: dict, 
    process_fn=None, 
    batch_size=False, 
    **kwargs
  ):
    """ Create a tf.data.Dataset for data retrieval
    
    Args:
      buffer: buffer, a callable object that stores data
      data_format: dict, whose keys are keys of returned data
      values are tuple (type, shape) that passed to 
      tf.data.Dataset.from_generator
    """
    self._buffer = buffer
    assert isinstance(data_format, dict)
    data_format = _transform_data(data_format)
    self.data_format = data_format
    do_logging('Dataset info:', logger=logger)
    do_logging(data_format, prefix='\t', logger=logger)
    self.types = {k: v.dtype for k, v in self.data_format.items()}
    self.shapes = {k: v.shape for k, v in self.data_format.items()}
    self._iterator = self._prepare_dataset(process_fn, batch_size, **kwargs)

  def __getattr__(self, name):
    if name.startswith('_'):
      raise AttributeError("attempted to get missing private attribute '{}'".format(name))
    return getattr(self._buffer, name)

  def sample(self):
    while True:
      yield self._buffer.sample()


def process_with_env(data, env_stats, obs_range=None, 
    one_hot_action=False, dtype=np.float32):
  if env_stats['obs_dtype'] == np.uint8 and obs_range is not None:
    if obs_range == [0, 1]:
      for k in data:
        if 'obs' in k:
          data[k] = data[k] / 255.
    elif obs_range == [-.5, .5]:
      for k in data:
        if 'obs' in k:
          data[k] = data[k] / 255. - .5
    else:
      raise ValueError(obs_range)
  if env_stats['is_action_discrete'] and one_hot_action:
    for k in data:
      if k.endswith('action'):
        raise NotImplemented
  return data


def create_dataset(buffer, env_stats, data_format=None, 
    central_buffer=False, one_hot_action=False):
  process = functools.partial(process_with_env, 
    env_stats=env_stats, one_hot_action=one_hot_action)
  if central_buffer:
    from core.elements.ray_dataset import RayDataset
    DatasetClass = RayDataset
  else:
    DatasetClass = Dataset
  dataset = DatasetClass(buffer, data_format, process)
  return dataset
