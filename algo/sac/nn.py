import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utility.display import pwc
from utility.tf_utils import build
from utility.rl_utils import logpi_correction
from utility.tf_distributions import DiagGaussian, Categorical
from nn.layers.func import mlp_layers
from nn.initializers import get_initializer


class SoftPolicy(tf.Module):
    def __init__(self, config, state_shape, action_dim, is_action_discrete, name, **kwargs):
        super().__init__(name=name)

        # network parameters
        units_list = config['units_list']

        norm = config.get('norm')
        activation = config.get('activation', 'relu')
        initializer_name = config.get('kernel_initializer', 'he_uniform')
        kernel_initializer = get_initializer(initializer_name)
        
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        
        """ Network definition """
        self.intra_layers = mlp_layers(units_list, 
                                        norm=norm, 
                                        activation=activation, 
                                        kernel_initializer=kernel_initializer(),
                                        **kwargs)
        self.mu = layers.Dense(action_dim, name='mu')
        self.logstd = layers.Dense(action_dim, name='logstd')

        # action distribution type
        self.ActionDistributionType = Categorical if is_action_discrete else DiagGaussian

        # build for variable initialization
        TensorSpecs = [(state_shape, tf.float32, 'state')]
        self.step = build(self._step, TensorSpecs)

    @tf.function
    @tf.Module.with_name_scope
    def _step(self, x):
        pwc(f'{self.name} "step" is retracing: x={x}', color='cyan')
        for l in self.intra_layers:
            x = l(x)

        mu = self.mu(x)
        logstd = self.logstd(x)
        logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)

        action_distribution = self.ActionDistributionType((mu, logstd))
        
        raw_action = action_distribution.sample()

        action = tf.tanh(raw_action)

        return action

    @tf.function
    @tf.Module.with_name_scope
    def det_action(self, x):
        pwc(f'{self.name} "det_action" is retracing: x={x}', color='cyan')
        with tf.name_scope('det_action'):
            for l in self.intra_layers:
                x = l(x)

            mu = self.mu(x)

            return tf.tanh(mu)

    @tf.Module.with_name_scope
    def train_step(self, x):
        pwc(f'{self.name} "train_step" is retracing: x={x}', color='cyan')
        for l in self.intra_layers:
            x = l(x)

        mu = self.mu(x)
        logstd = self.logstd(x)
        logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)

        action_distribution = self.ActionDistributionType((mu, logstd))
        
        raw_action = action_distribution.sample()
        logpi = action_distribution.logp(raw_action)

        action = tf.tanh(raw_action)
        logpi = logpi_correction(raw_action, logpi, is_action_squashed=False)

        return action, logpi


class SoftQ(tf.Module):
    def __init__(self, config, state_shape, action_dim, name, **kwargs):
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

    @tf.function
    def _step(self, x, a):
        return self.train_step(x, a)

    @tf.Module.with_name_scope
    def train_step(self, x, a):
        # pwc(f'{self.name} "step" is retracing: x={x}, a={a}', color='cyan')
        with tf.name_scope('step'):
            x = tf.concat([x, a], axis=1)
            for l in self.intra_layers:
                x = l(x)
            
        return x

    
class Temperature(tf.Module):
    def __init__(self, config, state_shape, action_dim, name, **kwargs):
        super().__init__(name=name)

        """ Network definition """
        self.intra_layer = layers.Dense(1)

        # build for variable initialization
        TensorSpecs = [
            (state_shape, tf.float32, 'state'),
            ([action_dim], tf.float32, 'action'),
        ]
        self.step = build(self._step, TensorSpecs)

    @tf.function
    def _step(self, x, a):
        return self.train_step(x, a)
    
    @tf.Module.with_name_scope
    def train_step(self, x, a):
        # pwc(f'{self.name} "step" is retracing: x={x}, a={a}', color='cyan')
        with tf.name_scope('step'):
            x = tf.concat([x, a], axis=1)
            log_temp = self.intra_layer(x)
            temp = tf.exp(log_temp)
        
        return log_temp, temp

def create_model(model_config, state_shape, action_dim, is_action_discrete):
    actor_config = model_config['actor']
    softq_config = model_config['softq']
    temperature_config = model_config['temperature']
    actor = SoftPolicy(actor_config, state_shape, action_dim, is_action_discrete, 'actor')
    softq1 = SoftQ(softq_config, state_shape, action_dim, 'softq1')
    softq2 = SoftQ(softq_config, state_shape, action_dim, 'softq2')
    target_softq1 = SoftQ(softq_config, state_shape, action_dim, 'target_softq1')
    target_softq2 = SoftQ(softq_config, state_shape, action_dim, 'target_softq2')
    temperature = Temperature(temperature_config, state_shape, action_dim, 'temperature')
    
    return dict(
        actor=actor,
        softq1=softq1,
        softq2=softq2,
        target_softq1=target_softq1,
        target_softq2=target_softq2,
        temperature=temperature,
    )