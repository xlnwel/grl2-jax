import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.module import Module
from nn.func import mlp, nn_registry
from nn.index import IndexedModule
from nn.utils import get_activation

""" Source this file to register Networks """


@nn_registry.register('hpembed')
class HPEmbed(Module):
    def __init__(self, name='hp_embed', **config):
        super().__init__(name=name)
        config = config.copy()

        self._layers = mlp(
            **config, 
            use_bias=False, 
            out_dtype='float32',
            name=name
        )

    def call(self, x, *args):
        for v in args:
            tf.debugging.assert_all_finite(v, f'Bad value {x}')
        hp = tf.stop_gradient(tf.expand_dims(tf.stack(args), 0))
        embed = self._layers(hp)
        ones = (1 for _ in x.shape[:-1])
        embed = tf.reshape(embed, [*ones, embed.shape[-1]])
        zeros = tf.zeros_like(x)
        embed = embed + zeros[..., :1]
        x = tf.concat([x, embed], -1)

        return x


@nn_registry.register('policy')
class Policy(IndexedModule):
    def __init__(self, name='policy', **config):
        super().__init__(name=name)
        config = config.copy()

        self.action_dim = config.pop('action_dim')
        self.is_action_discrete = config.pop('is_action_discrete')
        self.clip_sample_action = config.pop('clip_sample_action', False)
        self.action_low = config.pop('action_low', None)
        self.action_high = config.pop('action_high', None)
        self.eval_act_temp = config.pop('eval_act_temp', 1)
        self.attention_action = config.pop('attention_action', False)
        self.out_act = config.pop('out_act', None)
        embed_dim = config.pop('embed_dim', 10)
        self.init_std = config.pop('init_std', 1)
        assert self.eval_act_temp >= 0, self.eval_act_temp

        if self.attention_action:
            self.embed = tf.Variable(
                tf.random.uniform((self.action_dim, embed_dim), -0.01, 0.01), 
                dtype='float32',
                trainable=True,
                name=f'{name}/embed')

        if not self.is_action_discrete:
            self.logstd = tf.Variable(
                initial_value=np.log(self.init_std)*np.ones(self.action_dim), 
                dtype='float32', 
                trainable=True, 
                name=f'{name}/policy/logstd')
        config.setdefault('out_gain', .01)

        self._build_nets(config, out_size=self.action_dim)

    def call(self, x, hx=None, action_mask=None, evaluation=False):
        tf.debugging.assert_all_finite(x, 'Bad input')
        if self.indexed == 'all':
            for l in self._layers:
                x = l(x, hx)
        elif self.indexed == 'head':
            x = self._layers(x)
            x = self._head(x, hx)
        else:
            x = self._layers(x)
        tf.debugging.assert_all_finite(x, 'Bad policy output')
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
            if self.out_act == 'tanh':
                x = tf.tanh(x)
            tf.debugging.assert_all_finite(self.logstd, 'Bad action logstd')
            std = tf.exp(self.logstd)
            if evaluation and self.eval_act_temp:
                std = std * self.eval_act_temp
            tf.debugging.assert_all_finite(x, 'Bad action mean')
            tf.debugging.assert_all_finite(std, 'Bad action std')
            std = tf.ones_like(x) * std
            act_dist = tfd.MultivariateNormalDiag(x, std)
        self.act_dist = act_dist
        return act_dist

    def get_distribution(self, *, logits, mean, std):
        act_dist = tfd.Categorical(logits) \
            if self.is_action_discrete else tfd.MultivariateNormalDiag(mean, std)
        return act_dist

    def action(self, dist, evaluation):
        if self.is_action_discrete:
            action = dist.mode() if evaluation and self.eval_act_temp == 0 \
                else dist.sample()
        else:
            action = dist.sample()
            if self.clip_sample_action:
                action = tf.clip_by_value(
                    action, self.action_low, self.action_high)
        return action


@nn_registry.register('value')
class Value(IndexedModule):
    def __init__(self, name='value', **config):
        super().__init__(name=name)
        config = config.copy()
        
        config.setdefault('out_gain', 1)
        self._out_act = config.pop('out_act', None)
        if self._out_act:
            self._out_act = get_activation(self._out_act)
        out_size = config.pop('out_size', 1)

        self._build_nets(config, out_size=out_size)

    def call(self, x, hx=None):
        if self.indexed == 'all':
            for l in self._layers:
                x = l(x, hx)
        elif self.indexed == 'head':
            x = self._layers(x)
            x = self._head(x, hx)
        else:
            x = self._layers(x)
        value = x
        if value.shape[-1] == 1:
            value = tf.squeeze(value, -1)
        if self._out_act is not None:
            value = self._out_act(value)
        return value
