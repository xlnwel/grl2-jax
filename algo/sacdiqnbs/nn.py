import collections
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import mlp, cnn
from algo.sacdiqn.nn import Encoder, Actor, Q, Temperature


State = collections.namedtuple('State', ('deter', 'mean', 'std', 'stoch'))


class BeliefState(Module):
    def __init__(self, config, name='belief_state'):
        super().__init__(name=name)
        self._layers = mlp(**config, name='belief_state')

    def call(self, x, training=False):
        z = self._layers(x)
        mean, std = tf.split(z, 2, -1)
        std = tf.nn.softplus(std) + .1
        stoch = tfd.MultivariateNormalDiag(mean, std).sample()
        state = State(deter=x, mean=mean, std=std, stoch=stoch)

        return state

    def get_feat(self, state):
        return tf.concat([state.stoch, state.deter], -1)


class Transition(Module):
    def __init__(self, config, name='trans'):
        super().__init__(name=name)
        self._config = config
        
    def build(self, input_shape):
        config = self._config.copy()
        if 'dense' in config['layer_type']:
            config['out_size'] = input_shape[-1]
        self._layers = mlp(
            **config,
            out_dtype='float32',
            name=self.name,
        )

    def call(self, x, a, training=False):
        if 'conv2d' in self._config['layer_type']:
            a = a[..., None, None, :]
            m = np.ones_like(x.shape)
            m[-3:-1] = x.shape[-3:-1]
            a = tf.tile(a, m)
        assert a.shape[1:-1] == x.shape[1:-1], (x.shape, a.shape)
        x = tf.concat([x, a], axis=-1)
        x = self._layers(x, training=training)

        return x

class Reward(Module):
    def __init__(self, config, name='reward'):
        super().__init__(name=name)
        self._layers = mlp(
            **config,
            out_size=1,
            out_dtype='float32',
            name=name
        )
    
    def call(self, x, a, training=False):
        x = tf.concat([x, a], axis=-1)
        r = self._layers(x, training=training)

        return r


class SACIQNBS(Ensemble):
    def __init__(self, config, *, model_fn=None, env, **kwargs):
        model_fn = model_fn or create_components
        super().__init__(
            model_fn=model_fn, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, deterministic=False, epsilon=0, return_stats=False, return_eval_stats=False, **kwargs):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape

        x = self.encoder(x)
        state = self.state(x)
        feat = self.state.get_feat(state)
        action = self.actor(feat, deterministic=deterministic, epsilon=epsilon, return_stats=return_eval_stats)
        terms = {}
        if return_eval_stats:
            action, terms = action
            _, qtv, q = self.q(x, return_q=True)
            qtv = tf.transpose(qtv, [0, 2, 1])
            idx = tf.stack([tf.range(action.shape[0]), action], -1)
            qtv_max = tf.reduce_max(qtv, 1)
            q_max = tf.reduce_max(q, 1)
            action_best_q = tf.argmax(q, 1)
            qtv = tf.gather_nd(qtv, idx)
            q = tf.gather_nd(q, idx)
            action = tf.squeeze(action)
            action_best_q = tf.squeeze(action_best_q)
            qtv_max = tf.squeeze(qtv_max)
            qtv = tf.squeeze(qtv)
            q_max = tf.squeeze(q_max)
            q = tf.squeeze(q)
            terms = {
                'action': action,
                'action_best_q': action_best_q,
                'qtv_max': qtv_max,
                'qtv': qtv,
                'q_max': q_max,
                'q': q,
            }
        elif return_stats:
            _, qtv = self.q(x, action=action)
            qtv = tf.squeeze(qtv)
            terms['qtv'] = qtv
        action = tf.squeeze(action)

        return action, terms


def create_components(config, env, **kwargs):
    assert env.is_action_discrete
    action_dim = env.action_dim
    encoder_config = config['encoder']
    belief_state_config = config['belief_state']
    actor_config = config['actor']
    q_config = config['q']
    temperature_config = config['temperature']
    trans_config = config['transition']
    reward_config = config['reward']
    if temperature_config['temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config)
        
    models = dict(
        encoder=Encoder(encoder_config, name='encoder'),
        state=BeliefState(belief_state_config, name='belief_state'),
        target_encoder=Encoder(encoder_config, name='target_encoder'),
        actor=Actor(actor_config, action_dim, name='actor'),
        target_actor=Actor(actor_config, action_dim, name='target_actor'),
        q=Q(q_config, action_dim, name='q'),
        target_q=Q(q_config, action_dim, name='target_q'),
        temperature=temperature,
        trans=Transition(trans_config, name='trans'),
        reward=Reward(reward_config, name='reward'),
    )
    if config['twin_q']:
        models['q2'] = Q(q_config, action_dim, name='q2')
        models['target_q2'] = Q(q_config, action_dim, name='target_q2')

    return models

def create_model(config, env, **kwargs):
    return SACIQNBS(config=config, env=env, model_fn=create_components, **kwargs, name='saciqnbs')
