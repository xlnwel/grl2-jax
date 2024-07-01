import os
import cloudpickle

from tools.log import do_logging
from core.names import PATH_SPLIT, MODEL, OPTIMIZER
from core.typing import ModelPath, exclude_subdict
from tools.file import search_for_all_files


def _ckpt_filename(filedir, filename):
  if filedir is not None:
    filename = os.path.join(filedir, filename)
  if not filename.endswith('.pkl'):
    filename = f'{filename}.pkl'
  return filename


def set_weights_for_agent(
  agent, 
  model: ModelPath, 
  name='params', 
  backtrack=5
):
  weights = restore_params(model, name, backtrack=backtrack)
  agent.set_weights(weights)


def save(data, *, filedir, filename, backtrack=3, name='data', to_print=True, atomic=False):
  if not os.path.isdir(filedir):
    os.makedirs(filedir)
  filename = _ckpt_filename(filedir, filename)
  assert os.path.isdir(os.path.dirname(filename)), os.path.dirname(filename)
  if atomic:
    tmp_filename = filename + '.tmp'
    with open(tmp_filename, 'wb') as f:
      cloudpickle.dump(data, f)
    os.rename(tmp_filename, filename)
  else:
    with open(filename, 'wb') as f:
      cloudpickle.dump(data, f)
  if to_print:
    do_logging(f'Saving {name} in "{filename}"', backtrack=backtrack, level='info')


def restore(*, filedir=None, filename, backtrack=3, default={}, name='data', to_print=True):
  """ Retrieve data from filedir/filename
  filename specifies the whole path if filedir is None
  """
  filename = _ckpt_filename(filedir, filename)
  data = default
  if os.path.exists(filename):
    try:
      with open(filename, 'rb') as f:
        data = cloudpickle.load(f)
      if to_print:
        do_logging(f'Restoring {name} from "{filename}"', backtrack=backtrack, level='info')
    except Exception as e:
      do_logging(f'Failing restoring {name} from {filename}: {e}', backtrack=backtrack, level='warning')
  else:
    do_logging(f'No such file: {filename}', backtrack=backtrack, level='info')

  return data


def get_filedir(model_path: ModelPath, name: str, *args):
  return os.path.join(*model_path, name, *args)


def save_hierarchical_params(params, model: ModelPath, name='params', to_print=True):
  if MODEL in params:
    save_params(params[MODEL], 
                model, f'{name}/model', to_print=to_print)
  if OPTIMIZER in params:
    save_params(params[OPTIMIZER], 
                model, f'{name}/opt', to_print=to_print)
  rest_params = exclude_subdict(params, MODEL, OPTIMIZER)
  if rest_params:
    save_params(rest_params, model, name, to_print=to_print)


def save_params(params, model_path: ModelPath, name, backtrack=4, to_print=True):
  filedir = get_filedir(model_path, name)
  for k, v in params.items():
    save(v, filedir=filedir, filename=k, backtrack=backtrack, 
         name='parameters', to_print=to_print)


def restore_params(model_path: ModelPath, name, filenames=None, backtrack=4, to_print=True):
  filedir = get_filedir(model_path, name)
  if filenames is None:
    filenames = search_for_all_files(filedir, '.pkl', remove_dir=True)
  if not isinstance(filenames, (list, tuple)):
    filenames = [filenames]
  params = {}
  for filename in filenames:
    weights = restore(filedir=filedir, filename=filename, backtrack=backtrack, 
                      name='parameters', to_print=to_print)
    filename = filename.replace('.pkl', '')
    if weights:
      if PATH_SPLIT in filename:
        name1, name2 = filename.split(PATH_SPLIT)
        if name1 not in params:
          params[name1] = {}
        params[name1][name2] = weights
      else:
        params[filename] = weights
  return params


class Checkpoint:
  def __init__(self, config, name='ckpt'):
    if 'root_dir' in config and 'model_name' in config:
      self._model_path = ModelPath(config.root_dir, config.model_name)
    else:
      self._model_path = None
    self._name = name
    
  """ Save & Restore Model """
  def reset_model_path(self, model_path: ModelPath):
    self._model_path = model_path

  def save(self, params):
    save_params(params, self._model_path, self._name)

  def restore(self, filenames=None):
    return restore_params(self._model_path, self._name, filenames)

  def get_filedir(self, *args):
    assert self._model_path is not None, self._model_path
    return get_filedir(self._model_path, self._name, *args)
