import logging
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.log import do_logging
from core.module import Module
from nn.func import mlp, nn_registry, layer_registry
from utility.tf_utils import assert_rank
from utility.utils import dict2AttrDict


logger = logging.getLogger(__name__)

""" Source this file to register Networks """


class ConvFeedForward(Module):
    def __init__(self, name, **config):
        super().__init__(name)
        config = dict2AttrDict(config)

        self._layers = []
        units_list = config.pop('units_list')
        layer_type = layer_registry.get(config.pop('layer_type'))
        for u in units_list:
            self._layers += [
                layer_type(u, **config)]

@nn_registry.register('encoder')
class Encoder(Module):
    def __init__(self, name='encoder', **config):
        super().__init__(name)
        config = dict2AttrDict(config)

        self._numbers_layers = mlp(**config['numbers'], name=f'{name}/numbers')
        self._jokers_layers = mlp(**config['jokers'], name=f'{name}/jokers')
        self._others_layers = mlp(**config['others'], name=f'{name}/others') \
            if 'others' in config else None
    
    def call(self, numbers, jokers, others=None):
        # TODO: try dealing with stack
        t = numbers.shape[1]
        numbers = tf.reshape(numbers, [-1, *numbers.shape[2:]])
        x_n = self._numbers_layers(numbers)
        x_n = tf.reshape(x_n, [-1, t, np.prod(x_n.shape[1:])])
        x_j = self._jokers_layers(jokers)
        if others is not None:
            x_o = self._others_layers(others)
            x = tf.concat([x_n, x_j, x_o], axis=-1)
            do_logging(f'{self.name}, {x_n}, {x_j}, {x_o}, {x}', logger=logger, level='DEBUG')
        else:
            x = tf.concat([x_n, x_j], axis=-1)
            do_logging(f'{self.name}, {x_n}, {x_j}, {x}', logger=logger, level='DEBUG')
        return x


@nn_registry.register('policy')
class Policy(Module):
    def __init__(self, name='policy', **config):
        super().__init__(name=name)
        config = config.copy()

        self.eval_act_temp = config.pop('eval_act_temp', 1)
        assert self.eval_act_temp >= 0, self.eval_act_temp

        config.setdefault('out_gain', .01)
        self._layers = mlp(**config['head'], 
                        out_dtype='float32',
                        name=name)
        self._aux_layers = mlp(**config['aux'], name=f'{name}/aux') \
            if 'aux' in config else None

    def call(self, x, aux=None, action_mask=None, evaluation=False):
        if aux is not None:
            if self._aux_layers is not None:
                aux = self._aux_layers(aux)
            x = tf.concat([x, aux], axis=-1)
        x = self._layers(x)
        logits = x / self.eval_act_temp \
            if evaluation and self.eval_act_temp > 0 else x
        if action_mask is not None:
            assert logits.shape[1:] == action_mask.shape[1:], (logits.shape, action_mask.shape)
            logits = tf.where(action_mask, logits, -1e10)
        self.logits = logits
        act_dist = tfd.Categorical(logits)
        return act_dist

    def action(self, dist, evaluation):
        return dist.mode() if evaluation and self.eval_act_temp == 0 \
            else dist.sample()


@nn_registry.register('value')
class Value(Module):
    def __init__(self, name='value', **config):
        super().__init__(name=name)
        config = config.copy()
        
        config.setdefault('out_gain', 1)
        self._layers = mlp(**config,
                          out_size=1,
                          out_dtype='float32',
                          name=name)

    def call(self, x):
        value = self._layers(x)
        value = tf.squeeze(value, -1)
        return value


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
