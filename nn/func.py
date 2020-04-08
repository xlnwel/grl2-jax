
from tensorflow.keras import layers

from nn.block.cnn import *
from nn.block.mlp import *
from nn.dnc.dnc import DNC


def mlp(units_list=[], 
        out_dim=None, 
        norm=None, 
        activation=None, 
        layer_type=layers.Dense, 
        kernel_initializer='glorot_uniform', 
        **kwargs):
    return MLP(units_list, out_dim=out_dim, layer_type=layer_type, 
                norm=norm, activation=activation, 
                kernel_initializer=kernel_initializer, **kwargs)


def cnn(name, **kwargs):
    if name.lower() == 'ftw':
        return FTWCNN(**kwargs)
    elif name.lower() == 'impala':
        return IMPALACNN(**kwargs)
    elif name.lower() == 'none':
        return None
    else:
        raise NotImplementedError(f'Unknown CNN structure: {name}')


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
