from torch import nn

from th.nn.mlp import MLP
from th.nn.registry import nn_registry


def create_network(input_dim, config, device='cpu'):
  """ Create a network according to config
  
  Args: 
    config[Dict]: must contain <nn_id>, which specifies the 
      class of network to create. The rest arguments are 
      passed to the network for initialization
  """
  config = config.copy()
  if 'nn_id' not in config:
    raise ValueError(f'No nn_id is specified in config: {config}')
  nn_id = config.pop('nn_id')
  if nn_id is None:
    registry = nn_registry
  elif '_' in nn_id:
    nn_type, nn_id = nn_id.split('_', 1)
    registry = nn_registry.get(nn_type)
  else:
    registry = nn_registry
  network = registry.get(nn_id)
  if not issubclass(network, nn.Module):
    raise TypeError(f'create_network returns invalid network: {network}')
  return network(input_dim, **config, device=device)


def mlp(input_dim, units_list=[], out_size=None, **kwargs) -> MLP:
  kwargs.pop('nn_id', None)
  return MLP(input_dim, units_list, out_size=out_size, **kwargs)
