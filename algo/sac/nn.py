import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utility.display import pwc
from core.tf_config import build
from utility.rl_utils import logpi_correction
from utility.tf_distributions import DiagGaussian, Categorical
from nn.layers.func import mlp_layers
from nn.initializers import get_initializer
from nn.cnn import get_cnn


class SoftPolicy(tf.Module):
    def __init__(self, config, state_shape, action_dim, is_action_discrete, name='actor'):
        super().__init__(name=name)

        # network parameters
        self.is_action_discrete = is_action_discrete
        units_list = config['units_list']

        norm = config.get('norm')
        activation = config.get('activation', 'relu')
        initializer_name = config.get('kernel_initializer', 'he_uniform')
        kernel_initializer = get_initializer(initializer_name)
        
        self.LOG_STD_MIN = config.get('LOG_STD_MIN', -20)
        self.LOG_STD_MAX = config.get('LOG_STD_MAX', 2)
        
        """ Network definition """
        self.intra_layers = mlp_layers(units_list, 
                                        norm=norm, 
                                        activation=activation, 
                                        kernel_initializer=kernel_initializer())

        if is_action_discrete:
            self.logits = layers.Dense(action_dim, name='logits')
            self.tau = 1    #tf.Variable(1., dtype=tf.float32, name='softmax_tau')
        else:
            self.mu = layers.Dense(action_dim, name='mu')
            self.logstd = layers.Dense(action_dim, name='logstd')

        # action distribution type    
        self.ActionDistributionType = Categorical if is_action_discrete else DiagGaussian

        # build for variable initialization
        TensorSpecs = [(state_shape, tf.float32, 'state')]
        self.action = build(self._action, TensorSpecs)
        self.det_action = build(self._det_action, TensorSpecs)

    @tf.function(experimental_relax_shapes=True)
    def _action(self, x):
        with tf.name_scope('action'):
            action_distribution, _ = self._action_distribution(x)

            if self.is_action_discrete:
                action = action_distribution.sample(one_hot=False)
            else:
                raw_action = action_distribution.sample()
                action = tf.tanh(raw_action)
            return action

    @tf.function(experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def _det_action(self, x):
        with tf.name_scope('det_action'):
            for l in self.intra_layers:
                x = l(x)

            if self.is_action_discrete:
                logits = self.logits(x)
                return tf.argmax(logits, axis=1)
            else:
                mu = self.mu(x)
                return tf.tanh(mu)

    @tf.Module.with_name_scope
    def train_action(self, x):
        with tf.name_scope('train_action'):
            action_distribution, _ = self._action_distribution(x)

            if self.is_action_discrete:
                action = action_distribution.sample(reparameterize=True, hard=True)
            else:
                raw_action = action_distribution.sample()
                action = tf.tanh(raw_action)

        return action

    @tf.Module.with_name_scope
    def train_step(self, x):
        with tf.name_scope('train_step'):
            action_distribution, logstd = self._action_distribution(x)

            if self.is_action_discrete:
                action = action_distribution.sample(reparameterize=True, hard=True)
                logpi = action_distribution.logp(action)
            else:                
                raw_action = action_distribution.sample()
                raw_logpi = action_distribution.logp(raw_action)
                action = tf.tanh(raw_action)
                logpi = logpi_correction(raw_action, raw_logpi, is_action_squashed=False)

            return action, logpi, action_distribution.entropy(), logstd

    def _action_distribution(self, x):
        for l in self.intra_layers:
            x = l(x)
        
        if self.is_action_discrete:
            logits = self.logits(x)
            action_distribution = self.ActionDistributionType(logits, self.tau)
            logstd = None
        else:
            mu = self.mu(x)
            logstd = self.logstd(x)
            logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)

            action_distribution = self.ActionDistributionType(mu, logstd)

        return action_distribution, logstd

    def get_weights(self):
        return [v.numpy() for v in self.variables]

    def set_weights(self, weights):
        [v.assign(w) for v, w in zip(self.variables, weights)]


class SoftQ(tf.Module):
    def __init__(self, config, state_shape, action_dim, name='q'):
        super().__init__(name=name)

        # parameters
        units_list = config['units_list']

        norm = config.get('norm')
        activation = config.get('activation', 'relu')
        initializer_name = config.get('kernel_initializer', 'he_uniform')
        kernel_initializer = get_initializer(initializer_name)

        """ Network definition """
        self.intra_layers = mlp_layers(units_list, 
                                        out_dim=1,
                                        norm=norm, 
                                        activation=activation, 
                                        kernel_initializer=kernel_initializer())

        # build for variable initialization
        TensorSpecs = [
            (state_shape, tf.float32, 'state'),
            ([action_dim], tf.float32, 'action'),
        ]
        self.step = build(self._step, TensorSpecs)

    @tf.function(experimental_relax_shapes=True)
    def _step(self, x, a):
        return self.train_value(x, a)

    @tf.Module.with_name_scope
    def train_value(self, x, a):
        with tf.name_scope('step'):
            x = tf.concat([x, a], axis=1)
            for l in self.intra_layers:
                x = l(x)
            
        return x

    def get_weights(self):
        return [v.numpy() for v in self.variables]

    def set_weights(self, weights):
        [v.assign(w) for v, w in zip(self.variables, weights)]


class Temperature(tf.Module):
    def __init__(self, config, state_shape, action_dim, name='temperature'):
        super().__init__(name=name)

        """ Network definition """
        self.intra_layer = layers.Dense(1)

        # build for variable initialization
        TensorSpecs = [
            (state_shape, tf.float32, 'state'),
            ([action_dim], tf.float32, 'action'),
        ]
        self.step = build(self._step, TensorSpecs)

    @tf.function(experimental_relax_shapes=True)
    def _step(self, x, a):
        return self.train_step(x, a)
    
    @tf.Module.with_name_scope
    def train_step(self, x, a):
        with tf.name_scope('step'):
            x = tf.concat([x, a], axis=1)
            log_temp = self.intra_layer(x)
            temp = tf.exp(log_temp)
        
        return log_temp, temp

    def get_weights(self):
        return [v.numpy() for v in self.variables]

    def set_weights(self, weights):
        [v.assign(w) for v, w in zip(self.variables, weights)]


def create_model(model_config, state_shape, action_dim, is_action_discrete):
    actor_config = model_config['actor']
    q_config = model_config['q']
    temperature_config = model_config['temperature']
    actor = SoftPolicy(actor_config, state_shape, action_dim, is_action_discrete)
    q1 = SoftQ(q_config, state_shape, action_dim, 'q1')
    q2 = SoftQ(q_config, state_shape, action_dim, 'q2')
    target_q1 = SoftQ(q_config, state_shape, action_dim, 'target_q1')
    target_q2 = SoftQ(q_config, state_shape, action_dim, 'target_q2')
    if temperature_config['temp_type'] == 'state-action':
        temperature = Temperature(temperature_config, state_shape, action_dim)
    elif temperature_config['temp_type'] == 'variable':
        temperature = tf.Variable(1)
    elif temperature_config['temp_type'] == 'constant':
        temperature = temperature_config.get('value', .2)
    return dict(
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        temperature=temperature,
    )
