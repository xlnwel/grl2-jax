import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from nn.func import Encoder, mlp
from algo.ppo.nn import Actor, Value


class Prior(Module):
    def __init__(self, config, action_dim, is_action_discrete, name='prior'):
        super().__init__(name=name)

        self.action_dim = action_dim
        self.is_action_discrete = is_action_discrete

        self._dropout = tf.keras.layers.Dropout(config.pop('dropout'), (None, 1))

        self._init_std = config.pop('init_std', 1)
        if not self.is_action_discrete:
            self.logstd = tf.Variable(
                initial_value=np.log(self._init_std)*np.ones(action_dim), 
                dtype='float32', 
                trainable=True, 
                name=f'actor/logstd')
        config.setdefault('out_gain', .01)
        self._layers = mlp(**config, 
                        out_size=action_dim, 
                        out_dtype='float32',
                        name=name)

    def call(self, x):
        x = self._dropout(x, True)
        actor_out = self._layers(x)

        if self.is_action_discrete:
            logits = actor_out
            act_dist = tfd.Categorical(logits)
        else:
            std = tf.exp(self.logstd)
            act_dist = tfd.MultivariateNormalDiag(actor_out, std)
        return act_dist

    def action(self, dist, evaluation):
        return dist.mode() if evaluation and self.eval_act_temp == 0 \
            else dist.sample()


class PPOKL(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, evaluation=False, return_eval_stats=False):
        x = self.encoder(x)
        act_dist = self.actor(x, evaluation=evaluation)
        action = self.actor.action(act_dist, evaluation)

        if evaluation:
            return action
        else:
            value = self.value(x)
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
        prior=Prior(config['prior'], action_dim, is_action_discrete),
        value=Value(config['value'])
    )

def create_model(config, env, **kwargs):
    return PPOKL(config, env, **kwargs)
