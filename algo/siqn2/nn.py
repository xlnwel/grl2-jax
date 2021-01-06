import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.module import Ensemble
from nn.func import Encoder
from algo.iqn.nn import Quantile, Value as IQNValue


class Value(IQNValue):
    def __init__(self, config, action_dim, name):
        config = config.copy()
        prior = np.ones(action_dim, dtype=np.float32)
        prior /= np.sum(prior)
        self.prior = tf.Variable(prior, trainable=False, name='prior')
        act_temp = config.pop('act_temp', 1.)
        if isinstance(act_temp, (list, tuple, np.ndarray)):
            act_temp = np.expand_dims(act_temp, axis=-1)
        self.act_inv_temp = 1. / act_temp
        self.eval_act_temp = config.pop('eval_act_temp', .5)
        self.temp = config.pop('temp')
        super().__init__(config=config, action_dim=action_dim, name=name)

    def v(self, qs):
        v = self.temp * tf.reduce_logsumexp(1 / self.temp * qs, axis=-1)
        return v

    def logits(self, qs, v=None):
        if v is None:
            v = self.v(qs)
        v = tf.expand_dims(v, axis=-1)
        logits = 1 / self.temp * (qs - v)
        return logits
    
    def prob_logp(self, logits):
        probs = tf.nn.softmax(logits)
        logps = tf.math.log(tf.maximum(probs, 1e-8))    # bound logps to avoid numerical instability
        return probs, logps
    
    def action(self, logits, evaluation=False, epsilon=0):
        if evaluation:
            if self.eval_act_temp == 0:
                dist = tfd.Categorical(logits)
                action = dist.mode()
            else:
                logits = logits / self.eval_act_temp
                dist = tfd.Categorical(logits)
                action = dist.sample()
        else:
            if isinstance(epsilon, tf.Tensor) or epsilon:
                # scaled_logits = logits * self.act_inv_temp
                # prior_logits = tf.math.log(tf.maximum(self.prior, 1e-8))
                # prior_logits = tf.broadcast_to(prior_logits, logits.shape)
                # cond = tf.random.uniform(tf.shape(epsilon), 0, 1) > epsilon
                # cond = tf.reshape(cond, (-1, 1))
                # logits = tf.where(cond, scaled_logits, prior_logits)
                scaled_logits = logits * self.act_inv_temp
                cond = tf.random.uniform(tf.shape(epsilon), 0, 1) > epsilon
                cond = tf.reshape(cond, (-1, 1))
                logits = tf.where(cond, logits, scaled_logits)

            dist = tfd.Categorical(logits)
            action = dist.sample()
        
        return action

    def update_prior(self, x, lr):
        self.prior.assign_add(lr * (x - self.prior))


class SIQN(Ensemble):
    def __init__(self, config, *, model_fn=None, env, **kwargs):
        model_fn = model_fn or create_components
        super().__init__(
            model_fn=model_fn, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, evaluation=False, epsilon=0, return_stats=False):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape

        x = self.encoder(x)
        _, qt_embed = self.quantile(x)
        _, qs = self.q(x, qt_embed, return_value=True)
        logits = self.q.logits(qs)
        action = self.q.action(logits, evaluation=evaluation, epsilon=epsilon)
        terms = {}
        if return_stats:
            act_one_hot = tf.one_hot(action, self.q.action_dim, dtype=qs.dtype)
            q = tf.reduce_sum(qs * act_one_hot, axis=-1)
            q = tf.squeeze(q)
            terms = {'q': q}
        action = tf.squeeze(action)

        return action, terms


def create_components(config, env, **kwargs):
    assert env.is_action_discrete
    action_dim = env.action_dim
    encoder_config = config['encoder']
    quantile_config = config['quantile']
    q_config = config['q']
    q_config.update(config['actor'])
    
    return dict(
        encoder=Encoder(encoder_config, name='encoder'),
        quantile=Quantile(quantile_config, name='phi'),
        q=Value(q_config, action_dim, name='q'),
        target_encoder=Encoder(encoder_config, name='target_encoder'),
        target_quantile=Quantile(quantile_config, name='target_phi'),
        target_q=Value(q_config, action_dim, name='target_q'),
    )

def create_model(config, env, **kwargs):
    return SIQN(config, env=env, **kwargs)
