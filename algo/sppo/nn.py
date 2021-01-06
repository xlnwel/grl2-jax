import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.module import Ensemble
from nn.func import Encoder
from algo.ppo.nn import Actor, Value


class SPPO(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, evaluation=False, epsilon=0):
        x = self.encoder(x)
        act_dist = self.actor(x, evaluation=evaluation)
        action = self.actor.action(act_dist, evaluation)
        
        if evaluation:
            return action
        else:
            value = self.value(x)
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi, 'value': value, 'kl': -act_dist.entropy()}
            return action, terms    # keep the batch dimension for later use

    @tf.function
    def compute_value(self, x):
        x = self.encoder(x)
        value = self.value(x)
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
        value=Value(config['value'])
    )

def create_model(config, env, **kwargs):
    return SPPO(config, env, **kwargs)
