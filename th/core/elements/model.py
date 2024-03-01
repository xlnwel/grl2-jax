from torch import nn
from torch.utils._pytree import tree_map
from typing import Dict, Union

from core.log import do_logging
from core.names import MODEL
from th.core.ckpt.base import ParamsCheckpointBase
from core.ensemble import Ensemble, constructor
from core.typing import AttrDict, dict2AttrDict
from th.nn.func import create_network


class Model(ParamsCheckpointBase):
  """ A model, consisting of multiple modules, is a 
  self-contained unit for network inference. Its 
  subclass is expected to implement some methods 
  of practical meaning, such as <action> and 
  <compute_value> """
  def __init__(
    self, 
    *,
    config: AttrDict,
    env_stats: AttrDict,
    name: str,
  ):
    super().__init__(config, name, MODEL)
    self.env_stats = dict2AttrDict(env_stats, to_copy=True)
    self.modules: Dict[str, nn.Module] = AttrDict()

    self._initial_states = AttrDict()

    self.post_init()
    self.build_nets()
    self.print_params()

  def post_init(self):
    self.aid = self.config.get('aid', 0)
    self.gids = self.env_stats.aid2gids[self.aid]
    self.n_groups = len(self.gids)
    gid2uids = [self.env_stats.gid2uids[i] for i in self.gids]
    min_uid = gid2uids[0][0]
    self.gid2uids = [uid - min_uid for uid in gid2uids] # starts uids from zero
    self.is_action_discrete = self.env_stats.is_action_discrete[self.aid]

  def build_net(self, input_dim, name):
    net = create_network(input_dim, self.config[name])
    self.modules[name] = net
    return net

  def build_nets(self):
    raise NotImplementedError

  def print_params(self):
    if self.config.get('print_for_debug', True):
      for k, v in self.modules.items():
        do_logging(f'Module: {k}')
        do_logging(f'{v}')

        n = sum(p.numel() for p in v.parameters())
        do_logging(f'Total number of params of {k}: {n}')

  def action(self, data, evaluation):
    action = self.raw_action(data, evaluation)

    return action

  def raw_action(self, data, evaluation=False):
    raise NotImplementedError

  def get_weights(self, name: str=None):
    """ Returns a list/dict of weights

    Returns:
      If name is provided, it returns a dict of weights 
      for models specified by keys. Otherwise, it 
      returns a list of all weights
    """
    if name is None:
      name = list(self.modules.keys())
    elif isinstance(name, str):
      name = [name]
    assert isinstance(name, (tuple, list))

    weights = {n: self.modules[n].state_dict() for n in name}
    return weights

  def set_weights(self, weights: dict):
    """ Sets weights

    Args:
      weights: a dict or list of weights. If it's a dict, 
      it sets weights for models specified by the keys.
      Otherwise, it sets all weights 
    """
    assert set(weights).issubset(set(self.modules)) or set(self.modules).issubset(set(weights)), (list(self.params), list(weights))
    for name in self.modules.keys():
      if name in weights:
        self.modules[name].load_state_dict(weights[name])
      else:
        do_logging(f'Missing params: {name}')

  def get_states(self):
    pass
  
  def reset_states(self, state=None):
    pass

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return None

  @property
  def state_size(self):
    return None

  @property
  def state_keys(self):
    return None

  @property
  def state_type(self):
    return None


class ModelEnsemble(Ensemble):
  def __init__(
    self, 
    *, 
    config: dict, 
    env_stats: dict,
    constructor=constructor, 
    components=None, 
    name: str, 
    to_build=False, 
    to_build_for_eval=False,
    **classes
  ):
    super().__init__(
      config=config, 
      env_stats=env_stats, 
      constructor=constructor, 
      components=components, 
      name=name, 
      has_ckpt=False, 
      **classes
    )

  def get_weights(self, name: Union[dict, list]=None):
    weights = {}
    if name:
      if isinstance(name, dict):
        for model_name, comp_name in name.items():
          weights[model_name] = self.components[model_name].get_weights(comp_name)
      elif isinstance(name, list):
        for model_name in name:
          weights[model_name] = self.components[model_name].get_weights()
    else:
      for k, v in self.components.items():
        weights[k] = v.get_weights()
    return weights

  def set_weights(self, weights: Union[list, dict], default_initialization=None):
    for n, m in self.components.items():
      if n in weights:
        m.set_weights(weights[n], default_initialization)
      elif default_initialization:
        m.set_weights({}, default_initialization)

  def restore(self):
    for v in self.components.values():
      v.restore()

  def save(self):
    for v in self.components.values():
      v.save()
