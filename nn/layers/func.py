import tensorflow as tf
from tensorflow.keras import layers

from nn.norm.func import get_norm
from nn.activation.func import get_activation
from nn.layers.dnc.dnc import DNC


def mlp_layers(units_list, out_dim=None, norm=None, name=None, activation=None, **kwargs):
    """Return a stack of Dense layers
    
    Args:
        units_list: A list of units for each layer
        out_dim: Output dimension, no activation is applied
        norm: Normalization layer following each hidden layer
        kwargs: kwargs for tf.keras.layers.Dense
    """
    if isinstance(norm, str):
        NormLayer = get_norm(norm)
    else:
        NormLayer = norm    # norm is a layer class
    
    if norm is not None:
        ActivationLayer = get_activation(activation)

    with tf.name_scope(name):
        layer_stack = []
        for u in units_list:
            if NormLayer is None:
                layer_stack.append(layers.Dense(u, activation=activation, **kwargs))
            else:
                layer_stack.append(layers.Dense(u, **kwargs))
                layer_stack.append(NormLayer())
                layer_stack.append(ActivationLayer())
        
        if out_dim:
            layer_stack.append(layers.Dense(out_dim))

        return layer_stack


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
