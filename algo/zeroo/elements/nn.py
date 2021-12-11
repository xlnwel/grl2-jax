import logging
import tensorflow as tf

from algo.zero.elements.nn import *
from core.module import Module
from nn.func import mlp, nn_registry
from utility.tf_utils import assert_rank
from utility.utils import dict2AttrDict


logger = logging.getLogger(__name__)

""" Source this file to register Networks """

@nn_registry.register('mencoder')
class MemoryEncoder(Module):
    def __init__(self, name='memory_encoder', **config):
        super().__init__(name)
        config = dict2AttrDict(config)

        self._layers = mlp(**config, name=name)
        self._maxpool = tf.keras.layers.MaxPool1D(3)
    
    def call(self, states):
        x = self._layers(states)
        t = x.shape[1]
        x = tf.reshape(x, (-1, *x.shape[2:]))
        x = self._maxpool(x)
        x = tf.squeeze(x, 1)
        x = tf.reshape(x, (-1, t, x.shape[-1]))
        return x


if __name__ == '__main__':
    from tensorflow.keras import layers
    from env.func import create_env
    from utility.yaml_op import load_config
    config = load_config('algo/gd/configs/guandan.yaml')
    env = create_env(config['env'])
    env_stats = env.stats()
    net = Encoder(**config['model']['action_encoder'])
    obs_shape = env_stats['obs_shape']
    s = 3
    x = {
        'numbers': layers.Input((s, *obs_shape['numbers'])),
        'jokers': layers.Input((s, *obs_shape['jokers'])),
        'others': layers.Input((s, 20)),
    }
    y = net(**x)
    model = tf.keras.Model(x, y)
    model.summary(200)
