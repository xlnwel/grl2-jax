import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from nn.func import Encoder, mlp
from algo.ppo.nn import Value


class Actor(Module):
    def __init__(self, config, action_dim, is_action_discrete, name='value'):
        super().__init__(name=name)

        self._action_dim = action_dim
        self._duel = config.pop('duel', False)
        self._tau = config.pop('tau', 1)
        self.eval_act_temp = config.pop('eval_act_temp', .5)

        self._add_layer(config)
    
    @property
    def tau(self):
        return self._tau

    def _add_layer(self, config):
        """ Network definition """
        if self._duel:
            self._v_layers = mlp(
                **config,
                out_size=1, 
                name=self.name+'/v',
                out_dtype='float32')
        self._layers = mlp(
            **config, 
            out_size=self.action_dim, 
            name=self.name,
            out_dtype='float32')

    @property
    def action_dim(self):
        return self._action_dim
    
    def call(self, x, evaluation=False):
        qs = self.value(x)
        act_dist = self.act_dist(qs, evaluation)
        return act_dist
    
    def value(self, x):
        if self._duel:
            v = self._v_layers(x)
            a = self._layers(x)
            q = v + a - tf.reduce_mean(a, axis=-1, keepdims=True)
        else:
            q = self._layers(x)
        return q

    def action(self, dist, evaluation):
        return dist.mode() if evaluation and self.eval_act_temp == 0 \
            else dist.sample()

    def act_dist(self, qs, evaluation=False):
        logits = (qs - tf.reduce_max(qs, axis=-1, keepdims=True)) / self._tau
        logits = logits / self.eval_act_temp \
            if evaluation and self.eval_act_temp else logits
        act_dist = tfd.Categorical(logits)

        return act_dist

class PPG(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, obs, evaluation=False, return_eval_stats=0):
        x = self.encoder(obs)
        act_dist = self.actor(x, evaluation=evaluation)
        action = self.actor.action(act_dist, evaluation)

        if evaluation:
            return action
        else:
            logpi = act_dist.log_prob(action)
            if hasattr(self, 'value_encoder'):
                x = self.value_encoder(obs)
            value = self.value(x)
            terms = {'logpi': logpi, 'value': value, 
                'kl': self.kl * logpi}
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

    if config['architecture'] == 'shared':
        models = dict(
            encoder=Encoder(config['encoder']), 
            actor=Actor(config['actor'], action_dim, is_action_discrete),
            value=Value(config['value'])
        )
    elif config['architecture'] == 'dual':
        models = dict(
            encoder=Encoder(config['encoder']), 
            value_encoder=Encoder(config['encoder'], name='value_encoder'), 
            actor=Actor(config['actor'], action_dim, is_action_discrete),
            value=Value(config['value']),
            aux_value=Value(config['value'], name='aux_value'),
        )
    else:
        raise ValueError(f'Unknown architecture: {config["architecture"]}')
    return models

def create_model(config, env, **kwargs):
    return PPG(config, env, **kwargs)
