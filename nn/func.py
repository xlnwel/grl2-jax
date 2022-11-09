import haiku as hk

from nn.mlp import MLP
from nn.registry import nn_registry


def create_network(config, name):
    """ Create a network according to config
    
    Args: 
        config[Dict]: must contain <nn_id>, which specifies the 
            class of network to create. The rest arguments are 
            passed to the network for initialization
        name[str]: the name of the network
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
    if not issubclass(network, hk.Module):
        raise TypeError(f'create_network returns invalid network: {network}')
    return network(**config, name=name)


def construct_components(config, name):
    from nn.func import create_network
    networks = {k: create_network(v, name=f'{name}/{k}') 
        for k, v in config.items() if isinstance(v, dict)}
    return networks


@hk.transparent
def mlp(units_list=[], out_size=None, **kwargs) -> MLP:
    kwargs.pop('nn_id', None)
    return MLP(units_list, out_size=out_size, **kwargs)
