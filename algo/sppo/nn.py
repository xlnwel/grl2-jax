import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import Encoder, mlp


class Actor(Module):
    @config
    def __init__(self, action_dim, is_action_discrete, name='actor'):
        super().__init__(name=name)

        self.action_dim = action_dim
        self.is_action_discrete = is_action_discrete

        self.actor = mlp(self._units_list, 
                        out_size=action_dim, 
                        norm=self._norm,
                        activation=self._activation, 
                        kernel_initializer=self._kernel_initializer,
                        out_dtype='float32',
                        name=name,
                        out_gain=.01,
                        )

        if not self.is_action_discrete:
            self.logstd = tf.Variable(
                initial_value=np.log(self._init_std)*np.ones(action_dim), 
                dtype='float32', 
                trainable=True, 
                name=f'actor/logstd')

    def call(self, x):
        actor_out = self.actor(x)

        if self.is_action_discrete:
            act_dist = tfd.Categorical(actor_out)
        else:
            act_dist = tfd.MultivariateNormalDiag(actor_out, tf.exp(self.logstd))
        return act_dist

class Value(Module):
    def __init__(self, config, name='value'):
        super().__init__(name=name)
        self._layers = mlp(**config,
                          out_size=1,
                          out_dtype='float32',
                          name='value')

    def call(self, x):
        value = self._layers(x)
        value = tf.squeeze(value, -1)
        return value

class PPO(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, deterministic=False, epsilon=0):
        x = self.encoder(x)
        if deterministic:
            act_dist = self.actor(x)
            action = act_dist.mode()
            return action
        else:
            act_dist = self.actor(x)
            value = self.value(x)
            action = act_dist.sample()
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi, 'value': value}
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
    return PPO(config, env, **kwargs)
