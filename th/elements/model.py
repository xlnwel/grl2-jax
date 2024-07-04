from typing import Union, List, Tuple
import torch

from core.typing import dict2AttrDict
from core.elements.model import Model as ModelBase
from tools.log import do_logging
from tools.tree_ops import tree_map
from th.nn.func import create_network
from th.tools.th_utils import to_tensor
from th.utils import tpdv


class Model(ModelBase):
  """ A model, consisting of multiple modules, is a 
  self-contained unit for network inference. Its 
  subclass is expected to implement some methods 
  of practical meaning, such as <action> and 
  <compute_value> """
  def dl_init(self):
    self.device = self.config.device
    self.tpdv = tpdv(self.device)

  def build_net(self, input_dim, name):
    net = create_network(input_dim, self.config[name], self.device).to(**self.tpdv)
    self.modules[name] = net
    return net

  def print_params(self):
    if self.config.get('print_for_debug', True):
      for k, v in self.modules.items():
        do_logging(f'{v}', level='info')
        n = sum(p.numel() for p in v.parameters())
        do_logging(f'Total number of params of {k}: {n}', level='info')

  @torch.no_grad()
  def action(self, data, evaluation):
    for v in self.modules.values():
      v.eval()
    data = to_tensor(data, self.tpdv)
    action, stats, state = self.raw_action(data, evaluation)
    action, stats = tree_map(lambda x: x.cpu().numpy(), (action, stats))
    return action, stats, state

  def raw_action(self, data, evaluation=False):
    raise NotImplementedError

  def get_weights(self, name: Union[str, Tuple, List]=None):
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
        self.modules[name].load_state_dict(to_tensor(weights[name], self.tpdv))
      else:
        do_logging(f'Missing params: {name}', level='info')

  def save(self):
    self.params = self.get_weights()
    super().save(self.params)

  def train(self):
    for v in self.modules.values():
      v.train()
  
  def eval(self):
    for v in self.modules.values():
      v.eval()
