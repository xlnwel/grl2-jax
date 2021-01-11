import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import Encoder
from algo.iqn.nn import Quantile, Value
from algo.ppo.nn import Actor


class Quantile(Module):
    @config
    def __init__(self, name='phi'):
        super().__init__(name=name)

    def sample_tau(self, batch_size):
        tau_hat = tf.random.uniform([batch_size, self.N, 1], 
            minval=0, maxval=1, dtype=tf.float32)   # [B, N, 1]
        return tau_hat
    
    def call(self, x, n_qt=None, tau_hat=None):
        batch_size, cnn_out_size = x.shape
        # phi network
        n_qt = n_qt or self.N
        if tau_hat is None:
            
            tau_hat = tf.random.uniform([batch_size, n_qt, 1], 
                minval=0, maxval=1, dtype=tf.float32)   # [B, N, 1]
        pi = tf.convert_to_tensor(np.pi, tf.float32)
        degree = tf.cast(tf.range(1, self._tau_embed_size+1), tau_hat.dtype) * pi * tau_hat
        qt_embed = tf.math.cos(degree)              # [B, N, E]
        kwargs = dict(
            kernel_initializer=getattr(self, '_kernel_initializer', 'glorot_uniform'),
            activation=getattr(self, '_activation', 'relu'),
            out_dtype='float32',
        )
        qt_embed = self.mlp(
            qt_embed, 
            [cnn_out_size], 
            name=self.name,
            **kwargs)                  # [B, N, cnn.out_size]
        tf.debugging.assert_shapes([
            [qt_embed, (batch_size, n_qt, cnn_out_size)],
        ])
        return tau_hat, qt_embed


class PPO(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, tau_hat, evaluation=False, return_eval_stats=False):
        x = self.encoder(x)
        act_dist = self.actor(x, evaluation=evaluation)
        action = self.actor.action(act_dist, evaluation)

        if evaluation:
            return action
        else:
            _, qt_embed = self.quantile(x, tau_hat=tau_hat)
            value = self.value(x, qt_embed)
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi, 'value': value}
            return action, terms    # keep the batch dimension for later use

    @tf.function
    def compute_value(self, x, tau_hat):
        x = self.encoder(x)
        _, qt_embed = self.quantile(x, tau_hat=tau_hat)
        value = self.value(x, qt_embed)
        return value
    
    def reset_states(self, **kwargs):
        return

    @property
    def state_keys(self):
        return None

def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    return dict(
        encoder=Encoder(config['encoder']), 
        actor=Actor(config['actor'], action_dim, is_action_discrete),
        quantile=Quantile(config['quantile']),
        value=Value(config['value'], 1)
    )

def create_model(config, env, **kwargs):
    return PPO(config, env, **kwargs)
