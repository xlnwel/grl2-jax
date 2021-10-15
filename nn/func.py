from tensorflow.keras import layers

from nn.mlp import *
from nn.rnns.lstm import MLSTM
from nn.rnns.gru import MGRU
from nn.dnc.dnc import DNC
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
        nn_type, nn_id = nn_id.split('_')
        registry = nn_registry.get(nn_type)
    else:
        registry = nn_registry
    network = registry.get(nn_id)
    if not issubclass(network, tf.Module):
        raise TypeError(f'create_network returns invalid network: {network}')
    return network(**config, name=name)

def mlp(units_list=[], out_size=None, **kwargs):
    return MLP(units_list, out_size=out_size, **kwargs)

def rnn(config, name='rnn'):
    config = config.copy()
    rnn_name = config.pop('rnn_name')
    if rnn_name == 'gru':
        return layers.GRU(**config, name=name)
    elif rnn_name == 'mgru':
        return MGRU(**config, name=name)
    elif rnn_name == 'lstm':
        return layers.LSTM(**config, name=name)
    elif rnn_name == 'mlstm':
        return MLSTM(**config, name=name)
    else:
        raise ValueError(f'Unkown rnn: {rnn_name}')

def dnc_rnn(output_size, 
            access_config=dict(memory_size=128, word_size=16, num_reads=4, num_writes=1), 
            controller_config=dict(hidden_size=128),
            clip_value=20,
            name='dnc',
            rnn_config={}):
    """Return an RNN that encapsulates DNC
    
    Args:
        output_size: Output dimension size of dnc
        access_config: A dictionary of access module configuration. 
            memory_size: The number of memory slots
            word_size: The size of each memory slot
            num_reads: The number of read heads
            num_writes: The number of write heads
            name: name of the access module, optionally
        controller_config: A dictionary of controller(LSTM) module configuration
        clip_value: Clips controller and core output value to between
            `[-clip_value, clip_value]` if specified
        name: module name
        rnn_config: specifies extra arguments for keras.layers.RNN
    """
    dnc_cell = DNC(access_config, 
                controller_config, 
                output_size, 
                clip_value, 
                name)
    return layers.RNN(dnc_cell, **rnn_config)
