import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers

from core.module import Module, Ensemble
from core.decorator import config
from utility.rl_utils import logpi_correction
from utility.tf_distributions import Categorical
from utility.schedule import TFPiecewiseSchedule
from nn.func import mlp
from nn.utils import get_initializer


class Actor(Module):
    def __init__(self, config, action_dim, is_action_discrete, name='actor'):
        super().__init__(name=name)
        self._is_action_discrete = is_action_discrete
        self.LOG_STD_MIN = config.pop('LOG_STD_MIN', -20)
        self.LOG_STD_MAX = config.pop('LOG_STD_MAX', 2)

        out_size = action_dim if is_action_discrete else 2*action_dim
        self._layers = mlp(**config, out_size=out_size, name=name)

    def call(self, x, deterministic=False, epsilon=0):
        x = self._layers(x)

        if self._is_action_discrete:
            dist = tfd.Categorical(logits=x)
            action = dist.mode() if deterministic else dist.sample()
            if epsilon:
                rand_act = tfd.Categorical(tf.zeros_like(dist.logits)).sample()
                action = tf.where(
                    tf.random.uniform(action.shape[:-1], 0, 1) < epsilon,
                    rand_act, action)
        else:
            mu, logstd = tf.split(x, 2, -1)
            logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)
            std = tf.exp(logstd)
            dist = tfd.MultivariateNormalDiag(mu, std)
            raw_action = dist.mode() if deterministic else dist.sample()
            action = tf.tanh(raw_action)
            if epsilon:
                action = tf.clip_by_value(
                    tfd.Normal(action, epsilon).sample(), -1, 1)

        return action

    def train_step(self, x):
        x = self._layers(x)

        if self._is_action_discrete:
            dist = Categorical(logits=x)
            action = dist.sample(one_hot=True)
            logpi = dist.log_prob(action)
            terms = {}
        else:
            mu, logstd = tf.split(x, 2, -1)
            logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)
            std = tf.exp(logstd)
            dist = tfd.MultivariateNormalDiag(mu, std)
            raw_action = dist.sample()
            raw_logpi = dist.log_prob(raw_action)
            action = tf.tanh(raw_action)
            logpi = logpi_correction(raw_action, raw_logpi, is_action_squashed=False)
            terms = dict(raw_act_std=std)

        terms['entropy']= dist.entropy()

        return action, logpi, terms

class Q(Module):
    def __init__(self, config, name='q'):
        super().__init__(name=name)

        self._layers = mlp(**config, out_size=1, name=name)

    def call(self, x, a):
        x = tf.concat([x, a], axis=-1)
        x = self._layers(x)
        x = tf.squeeze(x, -1)
            
        return x


class Temperature(Module):
    @config
    def __init__(self, name='temperature'):
        super().__init__(name=name)

        if self._temp_type == 'state-action':
            kernel_initializer = get_initializer('orthogonal', gain=.01)
            self._layer = layers.Dense(1, name=name)
        elif self._temp_type == 'variable':
            self._log_temp = tf.Variable(np.log(self._value), dtype=tf.float32, name=name)
        elif self._temp_type == 'constant':
            self._temp = tf.Variable(self._value, trainable=False)
        elif self._temp_type == 'schedule':
            self._temp = TFPiecewiseSchedule(self._value)
        else:
            raise NotImplementedError(f'Error temp type: {self._temp_type}')
    
    @property
    def type(self):
        return self._temp_type

    @property
    def trainable(self):
        return self.type in ('state-action', 'variable')

    def call(self, x=None, a=None):
        if self._temp_type == 'state-action':
            x = tf.concat([x, a], axis=-1)
            x = self._layer(x)
            log_temp = -tf.nn.softplus(x)
            log_temp = tf.squeeze(log_temp)
            temp = tf.exp(log_temp)
        elif self._temp_type == 'variable':
            log_temp = self._log_temp
            temp = tf.exp(log_temp)
        elif self._temp_type == 'constant':
            temp = self._temp
            log_temp = tf.math.log(temp)
        elif self._temp_type == 'schedule':
            assert isinstance(x, int) or (
                isinstance(x, tf.Tensor) and x.shape == ())
            temp = self._temp(x)
            log_temp = tf.math.log(temp)
    
        return log_temp, temp


class SAC(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, deterministic=False, epsilon=0, **kwargs):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 2, x.shape
        
        action = self.actor(x, deterministic=deterministic, epsilon=epsilon)
        action = tf.squeeze(action)

        return action

    @tf.function
    def value(self, x):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 2, x.shape
        
        value = self.q(x)
        
        return value


def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete
    actor_config = config['actor']
    q_config = config['q']
    temperature_config = config['temperature']
    actor = Actor(actor_config, action_dim, is_action_discrete)
    q = Q(q_config, 'q')
    q2 = Q(q_config, 'q2')
    target_q = Q(q_config, 'target_q')
    target_q2 = Q(q_config, 'target_q2')
    if temperature_config['_temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config)
        
    return dict(
        actor=actor,
        q=q,
        q2=q2,
        target_q=target_q,
        target_q2=target_q2,
        temperature=temperature,
    )

def create_model(config, env, **kwargs):
    return SAC(config, env, **kwargs)