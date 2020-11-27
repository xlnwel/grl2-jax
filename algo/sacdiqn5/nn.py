import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import mlp
from algo.sacd.nn import Encoder, Actor, Temperature
from algo.iqn.nn import Quantile, Value


class SACIQN(Ensemble):
    def __init__(self, config, *, model_fn=None, env, **kwargs):
        model_fn = model_fn or create_components
        self._reward_entropy = config.pop('reward_entropy', False)
        super().__init__(
            model_fn=model_fn, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, obs, deterministic=False, epsilon=0, return_stats=False, return_eval_stats=False, **kwargs):
        if obs.shape.ndims % 2 != 0:
            obs = tf.expand_dims(obs, axis=0)
        assert obs.shape.ndims == 4, obs.shape

        x = self.actor_encoder(obs)
        action = self.actor(x, deterministic=deterministic, epsilon=epsilon)
        terms = {}
        if return_eval_stats:
            action, terms = action
            _, qt_embed = self.quantile(x)
            _, qtv, q = self.q(x, qt_embed, return_q=True)
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
            x = self.critic_encoder(obs)
            _, qt_embed = self.quantile(x)
            x_ext = tf.expand_dims(x, axis=1)
            _, v = self.v(x_ext, qt_embed, return_value=True)
            if self._reward_entropy:
                logp = tfd.Categorical(self.actor.logits).log_prob(action)
                if self.temperature.type == 'schedule':
                    _, temp = self.temperature(self._train_step)
                elif self.temperature.type == 'state-action':
                    raise NotImplementedError
                else:
                    _, temp = self.temperature()
                logp = temp * logp
                terms['logp'] = logp
            v = tf.squeeze(v)
            terms['v'] = v
        action = tf.squeeze(action)
        return action, terms


def create_components(config, env, **kwargs):
    assert env.is_action_discrete, env.name
    action_dim = env.action_dim
    encoder_config = config['encoder']
    actor_config = config['actor']
    v_config = config['v']
    q_config = config['q']
    temperature_config = config['temperature']
    quantile_config = config['quantile']

    models = dict(
        actor_encoder=Encoder(encoder_config, name='actor_encoder'),
        critic_encoder=Encoder(encoder_config, name='critic_encoder'),
        quantile=Quantile(quantile_config, name='phi'),
        actor=Actor(actor_config, action_dim),
        v=Value(v_config, 1, name='v'),
        q=Value(q_config, action_dim, name='q'),
        target_actor_encoder=Encoder(encoder_config, name='target_actor_encoder'),
        target_critic_encoder=Encoder(encoder_config, name='target_critic_encoder'),
        target_actor=Actor(actor_config, action_dim, name='target_actor'),
        target_v=Value(v_config, 1, name='target_v'),
        target_q=Value(q_config, action_dim, name='target_q'),
        temperature=Temperature(temperature_config),
    )

    return models

def create_model(config, env, **kwargs):
    return SACIQN(config=config, env=env, **kwargs)
