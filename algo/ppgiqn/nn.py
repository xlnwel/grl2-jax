import tensorflow as tf

from core.module import Ensemble
from algo.ppo.nn import Encoder, Actor, Value
from algo.iqn.nn import Quantile, Value as QuantileValue


class PPG(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, obs, tau_hat, evaluation=False, return_eval_stats=0):
        x = self.encoder(obs)
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
        x =self.value_encoder(x)
        _, qt_embed = self.quantile(x, tau_hat=tau_hat)
        value = self.value(x, qt_embed)
        return value

    @tf.function
    def compute_aux_data(self, obs, tau_hat):
        x = self.encoder(obs)
        logits = self.actor(x).logits
        x =self.value_encoder(obs)
        _, qt_embed = self.quantile(x, tau_hat=tau_hat)
        value = self.value(x, qt_embed)
        return logits, value

    def reset_states(self, **kwargs):
        return

    @property
    def state_keys(self):
        return None


def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    models = dict(
        encoder=Encoder(config['encoder']), 
        value_encoder=Encoder(config['encoder'], name='value_encoder'), 
        actor=Actor(config['actor'], action_dim, is_action_discrete),
        quantile=Quantile(config['quantile']),
        value=QuantileValue(config['value'], 1),
        aux_value=Value(config['value'], name='aux_value'),
    )
    return models

def create_model(config, env, **kwargs):
    return PPG(config, env, **kwargs)
