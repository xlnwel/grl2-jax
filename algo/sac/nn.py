import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.module import Module
from core.decorator import config
from utility.rl_utils import logpi_correction
from utility.tf_distributions import Categorical
from nn.func import mlp


class Actor(Module):
    @config
    def __init__(self, action_dim, is_action_discrete, name='actor'):
        super().__init__(name=name)
        self._dtype = global_policy().compute_dtype

        self._is_action_discrete = is_action_discrete
        
        out_dim = action_dim if is_action_discrete else 2*action_dim
        self._layers = mlp(self._units_list, 
                            out_dim=out_dim,
                            norm=self._norm, 
                            activation=self._activation)
    
    def __call__(self, x, deterministic=False, epsilon=0):
        x = tf.convert_to_tensor(x, self._dtype)
        x = tf.reshape(x, [-1, tf.shape(x)[-1]])
        deterministic = tf.convert_to_tensor(deterministic, tf.bool)

        action = self.action(x, deterministic, epsilon)
        action = np.squeeze(action.numpy())

        return action

    @tf.function(experimental_relax_shapes=True)
    def action(self, x, deterministic=False, epsilon=0):
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
            dist = Categorical(x)
            action = dist.sample()
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

class SoftQ(Module):
    @config
    def __init__(self, name='q'):
        super().__init__(name=name)

        self._layers = mlp(self._units_list, 
                            out_dim=1,
                            norm=self._norm, 
                            activation=self._activation)

    def __call__(self, x, a):
        x = tf.concat([x, a], axis=-1)
        x = self._layers(x)
        x = tf.squeeze(x)
            
        return x


class Temperature(Module):
    def __init__(self, config, name='temperature'):
        super().__init__(name=name)

        self.temp_type = config['temp_type']

        if self.temp_type == 'state-action':
            self.intra_layer = layers.Dense(1)
        elif self.temp_type == 'variable':
            self.log_temp = tf.Variable(0., dtype=global_policy().compute_dtype)
        else:
            raise NotImplementedError(f'Error temp type: {self.temp_type}')
    
    def __call__(self, x, a):
        if self.temp_type == 'state-action':
            x = tf.concat([x, a], axis=-1)
            x = self.intra_layer(x)
            log_temp = -tf.nn.softplus(x)
            log_temp = tf.squeeze(log_temp)
        else:
            log_temp = self.log_temp
        temp = tf.exp(log_temp)
    
        return log_temp, temp


def create_model(config, action_dim, is_action_discrete):
    actor_config = config['actor']
    q_config = config['q']
    temperature_config = config['temperature']
    actor = Actor(actor_config, action_dim, is_action_discrete)
    q1 = SoftQ(q_config, 'q1')
    q2 = SoftQ(q_config, 'q2')
    target_q1 = SoftQ(q_config, 'target_q1')
    target_q2 = SoftQ(q_config, 'target_q2')
    if temperature_config['temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config)
        
    return dict(
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        temperature=temperature,
    )
