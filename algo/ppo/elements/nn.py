import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.module import Module
from nn.func import mlp, nn_registry

""" Source this file to register Networks """


@nn_registry.register('policy')
class Policy(Module):
    def __init__(self, name='policy', **config):
        super().__init__(name=name)
        config = config.copy()

        self.action_dim = config.pop('action_dim')
        self.is_action_discrete = config.pop('is_action_discrete')
        self.eval_act_temp = config.pop('eval_act_temp', 1)
        self.attention_action = config.pop('attention_action', False)
        embed_dim = config.pop('embed_dim', 10)
        self._init_std = config.pop('init_std', 1)
        assert self.eval_act_temp >= 0, self.eval_act_temp

        if self.attention_action:
            self.embed = tf.Variable(
                tf.random.uniform((self.action_dim, embed_dim), -0.01, 0.01), 
                dtype='float32',
                trainable=True,
                name='embed')

        if not self.is_action_discrete:
            self.logstd = tf.Variable(
                initial_value=np.log(self._init_std)*np.ones(self.action_dim), 
                dtype='float32', 
                trainable=True, 
                name=f'policy/logstd')
        config.setdefault('out_gain', .01)
        self._layers = mlp(**config, 
                        out_size=embed_dim if self.attention_action else self.action_dim, 
                        out_dtype='float32',
                        name=name)

    def call(self, x, action_mask=None, evaluation=False):
        x = self._layers(x)
        if self.is_action_discrete:
            if self.attention_action:
                x = tf.matmul(x, self.embed, transpose_b=True)
            logits = x / self.eval_act_temp \
                if evaluation and self.eval_act_temp > 0 else x
            if action_mask is not None:
                assert logits.shape[1:] == action_mask.shape[1:], (logits.shape, action_mask.shape)
                logits = tf.where(action_mask, logits, -1e10)
            act_dist = tfd.Categorical(logits)
        else:
            std = tf.exp(self.logstd)
            if evaluation and self.eval_act_temp:
                std = std * self.eval_act_temp
            act_dist = tfd.MultivariateNormalDiag(x, std)
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
