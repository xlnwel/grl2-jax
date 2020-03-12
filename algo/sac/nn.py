import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers

from core.tf_config import build
from core.module import Module
from utility.display import pwc
from utility.rl_utils import logpi_correction
from utility.tf_distributions import Categorical
from nn.func import mlp


class SoftPolicy(Module):
    def __init__(self, config, state_shape, action_dim, is_action_discrete, name='actor'):
        super().__init__(name=name)

        # network parameters
        self.is_action_discrete = is_action_discrete
        self.state_shape = state_shape

        norm = config.get('norm')
        activation = config.get('activation', 'relu')
        kernel_initializer = config.get('kernel_initializer', 'glorot_uniform')
        
        self.LOG_STD_MIN = config.get('LOG_STD_MIN', -20)
        self.LOG_STD_MAX = config.get('LOG_STD_MAX', 2)
        
        """ Network definition """
        self._layers = mlp(config['units_list'], 
                                norm=norm, 
                                activation=activation, 
                                kernel_initializer=kernel_initializer)

        out_dim = action_dim if is_action_discrete else 2*action_dim
        self._out = mlp([],
                        out_dim=out_dim, 
                        norm=norm, 
                        activation=activation, 
                        kernel_initializer=kernel_initializer, 
                        name='logits')

        # build for variable initialization and avoiding unintended retrace
        TensorSpecs = [(state_shape, tf.float32, 'state'), (None, tf.bool, 'deterministic')]
        self._action = build(self._action_impl, TensorSpecs)
    
    def action(self, x, deterministic=False, epsilon=0):
        x = tf.convert_to_tensor(x, tf.float32)
        x = tf.reshape(x, [-1, *self.state_shape])
        deterministic = tf.convert_to_tensor(deterministic, tf.bool)

        action, terms = self._action(x, deterministic)
        action = np.squeeze(action.numpy())
        terms = dict((k, np.squeeze(v.numpy())) for k, v in terms.items())

        if epsilon:
            action += np.random.normal(scale=epsilon, size=action.shape)

        return action, terms

    @tf.function(experimental_relax_shapes=True)
    def _action_impl(self, x, deterministic=False):
        print(f'action retrace: {x.shape}, {deterministic}')
        x = self._layers(x)
        x = self._out(x)

        if self.is_action_discrete:
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
        x = self._out(x)

        if self.is_action_discrete:
            dist = Categorical(x)
            action = dist.sample(reparameterize=True)
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
    def __init__(self, config, state_shape, action_dim, name='q'):
        super().__init__(name=name)

        # parameters
        units_list = config['units_list']

        norm = config.get('norm')
        activation = config.get('activation', 'relu')
        kernel_initializer = config.get('kernel_initializer', 'glorot_uniform')

        """ Network definition """
        self._layers = mlp(units_list, 
                                out_dim=1,
                                norm=norm, 
                                activation=activation, 
                                kernel_initializer=kernel_initializer)

        # build for variable initialization
        TensorSpecs = [
            (state_shape, tf.float32, 'state'),
            ([action_dim], tf.float32, 'action'),
        ]
        self.step = build(self._step, TensorSpecs)

    @tf.function(experimental_relax_shapes=True)
    def _step(self, x, a):
        return self.train_step(x, a)

    def train_step(self, x, a):
        x = tf.concat([x, a], axis=-1)
        x = self._layers(x)
        x = tf.squeeze(x)
            
        return x


class Temperature(Module):
    def __init__(self, config, state_shape, action_dim, name='temperature'):
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


def create_model(model_config, state_shape, action_dim, is_action_discrete):
    actor_config = model_config['actor']
    q_config = model_config['q']
    temperature_config = model_config['temperature']
    actor = SoftPolicy(actor_config, state_shape, action_dim, is_action_discrete)
    q1 = SoftQ(q_config, state_shape, action_dim, 'q1')
    q2 = SoftQ(q_config, state_shape, action_dim, 'q2')
    target_q1 = SoftQ(q_config, state_shape, action_dim, 'target_q1')
    target_q2 = SoftQ(q_config, state_shape, action_dim, 'target_q2')
    if temperature_config['temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config, state_shape, action_dim)
        
    return dict(
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        temperature=temperature,
    )
