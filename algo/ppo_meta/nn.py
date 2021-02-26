import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from nn.func import Encoder, mlp


class Actor(Module):
    def __init__(self, config, action_dim, is_action_discrete, name='actor'):
        super().__init__(name=name)

        self.action_dim = action_dim
        self.is_action_discrete = is_action_discrete
        self.eval_act_temp = config.pop('eval_act_temp', .5)
        assert self.eval_act_temp >= 0, self.eval_act_temp

        self._init_std = config.pop('init_std', 1)
        if not self.is_action_discrete:
            self.logstd = tf.Variable(
                initial_value=np.log(self._init_std)*np.ones(action_dim), 
                dtype='float32', 
                trainable=True, 
                name=f'actor/logstd')
        self._layers = mlp(**config, 
                        out_size=action_dim, 
                        out_dtype='float32',
                        out_gain=.01,
                        name=name)

    def call(self, x, evaluation=False):
        actor_out = self._layers(x)

        if self.is_action_discrete:
            logits = actor_out / self.eval_act_temp \
                if evaluation and self.eval_act_temp else actor_out
            act_dist = tfd.Categorical(logits)
        else:
            std = tf.exp(self.logstd)
            if evaluation and self.eval_act_temp:
                std = std * self.eval_act_temp
            act_dist = tfd.MultivariateNormalDiag(actor_out, std)
        return act_dist

    def action(self, dist, evaluation):
        return dist.mode() if evaluation and self.eval_act_temp == 0 \
            else dist.sample()


class Value(Module):
    def __init__(self, config, name='value'):
        super().__init__(name=name)
        self._layers = mlp(**config,
                          out_size=1,
                          out_dtype='float32',
                          out_gain=.01,
                          name=name)

    def call(self, x):
        value = self._layers(x)
        value = tf.squeeze(value, -1)
        return value


class MetaParams(Module):
    def __init__(self, config, name='meta_params'):
        super().__init__(name=name)

        for k, v in config.items():
            setattr(self, k, tf.Variable(
                v['init'], dtype='float32', trainable=True, name=f'meta/{k}'))
            setattr(self, f'{k}_outer', v['outer'])
            setattr(self, f'{k}_act', tf.keras.activations.get(v['act']))

        self.params = list(config)
    
    def __call__(self, name):
        val = getattr(self, name)
        outer = getattr(self, f'{name}_outer')
        act = getattr(self, f'{name}_act')
        return outer * act(val)


class MetaEmbeddings(Module):
    def __init__(self, config, name='meta_embed'):
        super().__init__(name=name)

        for k, v in config.items():
            setattr(self, f'{k}_layer', tf.keras.layers.Dense(
                v['units'], use_bias=False, name=f'meta/{k}_dense'))

    def __call__(self, x, name):
        if x.shape.rank == 0:
            x = tf.reshape(x, [-1, 1])
        y = getattr(self, f'{name}_layer')(x)
        return y


class PPO(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, evaluation=False, return_eval_stats=False):
        x = self.encode(x)
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
        x = self.encode(x)
        value = self.value(x)
        return value
    
    def reset_states(self, **kwargs):
        return

    @property
    def state_keys(self):
        return None

    def encode(self, obs):
        x = self.encoder(obs)
        meta_embeds = tf.concat([self.get_meta_embedding(name) for name in self.meta.params], -1)
        meta_embeds = tf.tile(meta_embeds, [tf.shape(obs)[0], 1])
        x = tf.concat([x, meta_embeds], axis=-1)
        return x

    def get_meta(self, name, stop_gradient=False):
        x = self.meta(name)
        if stop_gradient:
            x = tf.stop_gradient(x)
        return x

    def get_meta_embedding(self, name):
        x = self.get_meta(name, stop_gradient=True)
        return self.meta_embed(x, name)

def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    return dict(
        encoder=Encoder(config['encoder']), 
        actor=Actor(config['actor'], action_dim, is_action_discrete),
        value=Value(config['value']),
        meta=MetaParams(config['meta']),
        meta_embed=MetaEmbeddings(config['meta'])
    )

def create_model(config, env, **kwargs):
    return PPO(config, env, **kwargs)
