import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.module import Module, Ensemble
from core.decorator import config
from algo.ppo.nn import Encoder, Actor, Value


class PPG(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)
    
    @tf.function
    def __call__(self, obs):
        x = self.encoder(obs)
        act_dist = self.actor(x)
        logits = act_dist.logits
        if hasattr(self, 'value_encoder'):
            x = self.value_encoder(obs)
        value = self.value(x)
        return logits, value

    @tf.function
    def action(self, obs, deterministic=False, epsilon=0):
        x = self.encoder(obs)
        if deterministic:
            act_dist = self.actor(x)
            action = tf.squeeze(act_dist.mode())
            return action
        else:
            act_dist = self.actor(x)
            action = act_dist.sample()
            logpi = act_dist.log_prob(action)
            if hasattr(self, 'value_encoder'):
                x = self.value_encoder(obs)
            value = self.value(x)
            terms = {'logpi': logpi, 'value': value}
            return action, terms    # keep the batch dimension for later use

    @tf.function
    def compute_value(self, x):
        if hasattr(self, 'value_encoder'):
            x =self.value_encoder(x)
        else:
            x = self.encoder(x)
        value = self.value(x)
        return value

    @tf.function
    def compute_aux_data(self, obs):
        x = self.encoder(obs)
        logits = self.actor(x).logits
        if hasattr(self, 'value_encoder'):
            x =self.value_encoder(obs)
        value = self.value(x)
        return logits, value

    def reset_states(self, **kwargs):
        return

    @property
    def state_keys(self):
        return None


def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    if config['architecture'] == 'dual':
        models = dict(
            encoder=Encoder(config['encoder']), 
            value_encoder=Encoder(config['encoder'], name='value_encoder'), 
            actor=Actor(config['actor'], action_dim, is_action_discrete),
            value=Value(config['value']),
            aux_value=Value(config['value'], name='aux_value'),
        )
    else:
        models = dict(
            encoder=Encoder(config['encoder']), 
            actor=Actor(config['actor'], action_dim, is_action_discrete),
            value=Value(config['value'])
        )
    return models

def create_model(config, env, **kwargs):
    return PPG(config, env, **kwargs)
