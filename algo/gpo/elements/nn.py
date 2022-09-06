import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.ensemble import Module
from nn.func import mlp, nn_registry

""" Source this file to register Networks """


@nn_registry.register('policy')
class Policy(Module):
    def __init__(self, name='policy', **config):
        super().__init__(name=name)
        config = config.copy()

        self.action_dim = config.pop('action_dim')
        self.is_action_discrete = config.pop('is_action_discrete')
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
                name='embed')

        if not self.is_action_discrete:
            self.logstd = tf.Variable(
                initial_value=np.log(self.init_std)*np.ones(self.action_dim), 
                dtype='float32', 
                trainable=True, 
                name=f'policy/logstd')
        config.setdefault('out_gain', .01)
        self._layers = mlp(
            **config, 
            out_size=embed_dim if self.attention_action else self.action_dim, 
            out_dtype='float32',
            name=name
        )

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
            if self.out_act == 'tanh':
                x = tf.tanh(x)
            else:
                assert self.out_act is None, 'Unknown output activation '
            std = tf.exp(self.logstd)
            if evaluation and self.eval_act_temp:
                std = std * self.eval_act_temp
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
            if self.action_low is not None:
                action = tf.clip_by_value(
                    action, self.action_low, self.action_high)
        return action


@nn_registry.register('value')
class Value(Module):
    def __init__(self, name='value', **config):
        super().__init__(name=name)
        config = config.copy()
        
        config.setdefault('out_gain', 1)
        if 'out_size' not in config:
            config['out_size'] = 1
        self._layers = mlp(
            **config,
            out_dtype='float32',
            name=name
        )

    def call(self, x):
        value = self._layers(x)
        if value.shape[-1] == 1:
            value = tf.squeeze(value, -1)
        return value


@nn_registry.register('meta')
class MetaParams(Module):
    def __init__(self, config, name='meta_params'):
        super().__init__(name=name)

        for k, v in config.items():
            setattr(self, k, tf.Variable(
                v['init'], dtype='float32', trainable=v['trainable'], name=f'meta/{k}'))
            setattr(self, f'{k}_outer', v['outer'])
            setattr(self, f'{k}_act', tf.keras.activations.get(v['act']))

        self.params = list(config)
    
    def __call__(self, name):
        val = getattr(self, name)
        outer = getattr(self, f'{name}_outer')
        act = getattr(self, f'{name}_act')
        return outer * act(val)
