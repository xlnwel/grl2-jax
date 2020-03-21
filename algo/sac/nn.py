import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers

from core.tf_config import build
from core.module import Module
from core.decorator import config
from utility.display import pwc
from utility.rl_utils import logpi_correction
from utility.tf_distributions import Categorical
from nn.func import mlp


class SoftPolicy(Module):
    @config
    def __init__(self, obs_shape, action_dim, is_action_discrete, name='actor'):
        super().__init__(name=name)

        # network parameters
        self._is_action_discrete = is_action_discrete
        self._obs_shape = obs_shape
        
        """ Network definition """
        out_dim = action_dim if is_action_discrete else 2*action_dim
        self._layers = mlp(self._units_list, 
                            out_dim=out_dim,
                            norm=self._norm, 
                            activation=self._activation, 
                            kernel_initializer=self._kernel_initializer)

        # build for avoiding unintended retrace
        TensorSpecs = [(obs_shape, tf.float32, 'obs'), (None, tf.bool, 'deterministic')]
        self._action = build(self._action_impl, TensorSpecs)
    
    def __call__(self, x, deterministic=False, epsilon=0):
        x = tf.convert_to_tensor(x, tf.float32)
        x = tf.reshape(x, [-1, *self._obs_shape])
        deterministic = tf.convert_to_tensor(deterministic, tf.bool)

        action, terms = self._action(x, deterministic)
        action = np.squeeze(action.numpy())
        terms = dict((k, np.squeeze(v.numpy())) for k, v in terms.items())

        if epsilon:
            action += np.random.normal(scale=epsilon, size=action.shape)

        return action, terms

    @tf.function(experimental_relax_shapes=True)
    def _action_impl(self, x, deterministic=False):
        print(f'Policy action retrace: {x.shape}, {deterministic}')
        x = self._layers(x)

        if self._is_action_discrete:
            dist = tfd.Categorical(x)
            action = dist.mode() if deterministic else dist.sample()
            terms = {}
        else:
            mu, logstd = tf.split(x, 2, -1)
            logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)
            std = tf.exp(logstd)
            dist = tfd.MultivariateNormalDiag(mu, std)
            raw_action = dist.sample()
            action = tf.tanh(raw_action)
            terms = dict(action_std=std)

        return action, terms

    def train_step(self, x):
        x = self._layers(x)

        if self._is_action_discrete:
            dist = Categorical(x)
            action = dist.sample()
            logpi = dist.log_prob(action)
        else:
            mu, logstd = tf.split(x, 2, -1)
            logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)
            std = tf.exp(logstd)
            dist = tfd.MultivariateNormalDiag(mu, std)
            raw_action = dist.sample()
            raw_logpi = dist.log_prob(raw_action)
            action = tf.tanh(raw_action)
            logpi = logpi_correction(raw_action, raw_logpi, is_action_squashed=False)

        terms = dict(entropy=dist.entropy())

        return action, logpi, terms

class SoftQ(Module):
    @config
    def __init__(self, name='q'):
        super().__init__(name=name)

        """ Network definition """
        self._layers = mlp(self._units_list, 
                            out_dim=1,
                            norm=self._norm, 
                            activation=self._activation, 
                            kernel_initializer=self._kernel_initializer)

    def __call__(self, x, a):
        return self.train_step(x, a)

    @tf.function
    def train_step(self, x, a):
        print(f'SoftQ train_step retrace: {x}, {a}')
        x = tf.concat([x, a], axis=-1)
        x = self._layers(x)
        x = tf.squeeze(x)
            
        return x


class Temperature(Module):
    def __init__(self, config, name='temperature'):
        super().__init__(name=name)

        self.temp_type = config['temp_type']
        """ Network definition """
        if self.temp_type == 'state-action':
            self.intra_layer = layers.Dense(1)
        elif self.temp_type == 'variable':
            self.log_temp = tf.Variable(0.)
        else:
            raise NotImplementedError(f'Error temp type: {self.temp_type}')
    
    def train_step(self, x, a):
        if self.temp_type == 'state-action':
            x = tf.concat([x, a], axis=-1)
            x = self.intra_layer(x)
            log_temp = -tf.nn.relu(x)
            log_temp = tf.squeeze(log_temp)
        else:
            log_temp = self.log_temp
        temp = tf.exp(log_temp)
    
        return log_temp, temp


def create_model(model_config, obs_shape, action_dim, is_action_discrete):
    actor_config = model_config['actor']
    q_config = model_config['q']
    temperature_config = model_config['temperature']
    actor = SoftPolicy(actor_config, obs_shape, action_dim, is_action_discrete)
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
