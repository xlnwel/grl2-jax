import time
import logging
import copy
import re
from multiprocessing import Process
import subprocess
import numpy as np

from core.typing import AttrDict2dict, dict2AttrDict
from tools.utils import modify_config, product_flatten_dict
from run.ops import change_config_with_kw_string

logger = logging.getLogger(__name__)


def find_key(key, config, results):
  if isinstance(config, dict):
    if key in config:
      results.append(config)
    else:
      for v in config.values():
        results = find_key(key, v, results)
  return results


class GridSearch:
  def __init__(
    self, 
    config, 
    train_func, 
    n_trials=1, 
    logdir='logs', 
    dir_prefix='', 
    separate_process=False, 
    delay=1,
    multiprocess=True
  ):
    self.config = dict2AttrDict(config)
    self.train_func = train_func
    self.n_trials = n_trials
    self.root_dir = logdir
    self.dir_prefix = dir_prefix
    self.separate_process = separate_process
    self.delay=delay
    self._pid = 0
    p = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
    gpus, _ = p.communicate()
    self.n_gpus = len(re.findall('GPU [0-9]', str(gpus)))

    if multiprocess:
      self.processes = []
    else:
      self.processes = None

  def __call__(self, kw_dict={}, **kwargs):
    self._setup_root_dir()
    kw_dict.update(kwargs)
    for k, v in kw_dict.items():
      if isinstance(v, np.ndarray):
        if np.issubdtype(v.dtype, np.floating):
          v_type = float 
        elif np.issubdtype(v.dtype, np.integer): 
          v_type = int
        else:
          raise TypeError(f'Unknown type for {v}: {v_type}')
        kw_dict[k] = [v_type(x) for x in v]
      elif isinstance(v, str):
        kw_dict[k] = [v]
      else:
        kw_dict[k] = list(v)
    if kw_dict == {} and self.n_trials == 1 and not self.separate_process:
      # if no argument is passed in, run the default setting
      if self.processes is None:
        self.train_func([self.config])
      else:
        for i in range(self.n_trials):
          config = self.config.copy()
          modify_config(
            config, 
            root_dir=self.root_dir, 
            model_name=f'{config.model_name}/seed={i}', 
            seed=i
          )
          self.processes.append(
            Process(target=self.train_func, 
            args=([self.config],), 
            kwargs={'gpu': self._pid % self.n_gpus}))
    else:
      # do grid search
      model_name = self.config.model_name
      self._change_config(model_name, kw_dict)

    return self.processes

  def _setup_root_dir(self):
    if self.dir_prefix:
      self.dir_prefix += '-'
    self.root_dir = (
      f'{self.root_dir}/'
      f'{self.config.env["env_name"]}/'
      f'{self.config.algorithm}'
    )

  def _change_config(self, model_name, kwargs):
    seed = kwargs.pop('seed', None)
    if seed is None:
      n_trials = self.n_trials
      seed = list(range(n_trials))
    else:
      n_trials = len(seed)
    assert len(seed) == n_trials, (seed, n_trials)
    kw_list = product_flatten_dict(**kwargs)
    
    for kw in kw_list:
      # deepcopy to avoid unintended conflicts
      config = copy.deepcopy(AttrDict2dict(self.config))
      kw_str = [f'{k}={v}' for k, v in kw.items()]
      new_model_name = change_config_with_kw_string(
        kw_str, config, model_name)

      for i in range(n_trials):
        config2 = dict2AttrDict(config, to_copy=True)

        if n_trials > 1:
          mn = f'{new_model_name}/seed={i}'
        else:
          mn = f'seed=None/{new_model_name}'
        if 'video_path' in config2['env']:
          config2['env']['video_path'] = \
            f'{self.root_dir}/{mn}/{config2["env"]["video_path"]}'
        
        config2 = modify_config(
          config2, 
          root_dir=self.root_dir, 
          model_name=mn, 
          seed=seed[i]
        )
        # print_dict(config2)
        if self.processes is None:
          self.train_func([config2])
        else:
          p = Process(target=self.train_func, args=([config2],), kwargs={'gpu': self._pid % self.n_gpus})
          p.start()
          self.processes.append(p)
          time.sleep(self.delay)   # ensure sub-processs starts in order
          self._pid += 1
