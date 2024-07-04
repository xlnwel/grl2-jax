import numpy as np
import jax
import haiku as hk
from typing import Union, List, Tuple

from tools.log import do_logging
from core.elements.model import Model as ModelBase
from nn.func import create_network
from tools.display import print_dict_info, summarize_arrays, int2str


class Model(ModelBase):
  """ A model, consisting of multiple modules, is a 
  self-contained unit for network inference. Its 
  subclass is expected to implement some methods 
  of practical meaning, such as <action> and 
  <compute_value> """
  def _prngkey(self, seed=None):
    if seed is None:
      if self.config.seed is None:
        self.config.seed = 42
      seed = self.config.seed
    do_logging(f'Model({self.name}) seed: {seed}', level='debug')
    return jax.random.PRNGKey(seed)

  def dl_init(self):
    self.rng = self._prngkey()
    self.act_rng = self.rng

  def build_net(self, *args, name, return_init=False, **kwargs):
    def build(*args, **kwargs):
      net = create_network(self.config[name], name)
      return net(*args, **kwargs)
    net = hk.transform(build)
    if return_init:
      return net.init, net.apply
    else:
      self.rng, rng = jax.random.split(self.rng)
      self.act_rng = self.rng
      return net.init(rng, *args, **kwargs), net.apply

  def print_params(self):
    if self.config.get('print_params', True):
      for k, v in self.params.items():
        do_logging(f'Module: {k}', level='info')
        print_dict_info(v, prefix='\t', level='info')
        n = summarize_arrays(v)
        n = int2str(n)
        do_logging(f'Total number of params of {k}: {n}', level='info')

  @property
  def theta(self):
    return self.params

  def compile_model(self):
    self.jit_action = jax.jit(self.raw_action, static_argnames=('evaluation'))
    # self.jit_action = jax.jit(self.raw_action, static_argnums=(3))

  def action(self, data, evaluation):
    self.act_rng, act_rng = jax.random.split(self.act_rng)
    action, stats, state = self.jit_action(self.params, act_rng, data, evaluation)
    action, stats = jax.tree_util.tree_map(np.asarray, (action, stats))
    return action, stats, state

  def raw_action(self, params, rng, data, evaluation=False):
    raise NotImplementedError

  def get_weights(self, name: Union[str, Tuple, List]=None):
    """ Returns a list/dict of weights

    Returns:
      If name is provided, it returns a dict of weights 
      for models specified by keys. Otherwise, it 
      returns a list of all weights
    """
    if name is None:
      name = list(self.params.keys())
    elif isinstance(name, str):
      name = [name]
    assert isinstance(name, (tuple, list))

    weights = {n: self.params[n] for n in name}
    weights = jax.device_get(weights)
    return weights

  def set_weights(self, weights: dict):
    """ Sets weights

    Args:
      weights: a dict or list of weights. If it's a dict, 
      it sets weights for models specified by the keys.
      Otherwise, it sets all weights 
    """
    assert set(weights).issubset(set(self.params)) or set(self.params).issubset(set(weights)), (list(self.params), list(weights))
    for name in self.params.keys():
      if name in weights:
        self.params[name] = weights[name]
      else:
        do_logging(f'Missing params: {name}', level='info')
