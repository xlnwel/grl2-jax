import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utility.display import pwc
from core.tf_config import build
from utility.rl_utils import logpi_correction
from utility.tf_distributions import DiagGaussian, Categorical
from nn.func import mlp


class SoftPolicy(tf.Module):
    def __init__(self, config, state_shape, action_dim, is_action_discrete, name='actor'):
        super().__init__(name=name)

        self.max_ar = config['max_ar']
        self.gamma = config['gamma']
        self.action_dim = action_dim

        # network parameters
        self.is_action_discrete = is_action_discrete

        norm = config.get('norm')
        activation = config.get('activation', 'relu')
        kernel_initializer = config.get('kernel_initializer', 'he_uniform')
        
        self.LOG_STD_MIN = config.get('LOG_STD_MIN', -20)
        self.LOG_STD_MAX = config.get('LOG_STD_MAX', 2)
        
        """ Network definition """
        self.intra_layers = mlp(config['units_list'], 
                                norm=norm, 
                                activation=activation, 
                                kernel_initializer=kernel_initializer)

        if is_action_discrete:
            self.logits = mlp(config.get('logits_units', []),
                            out_dim=action_dim, 
                            norm=norm, 
                            activation=activation, 
                            kernel_initializer=kernel_initializer, 
                            name='logits')
            self.tau = 1.   # tf.Variable(1., dtype=tf.float32, name='softmax_tau')
        else:
            self.mu = mlp(config.get('mu_units', []),
                        out_dim=action_dim, 
                        norm=norm, 
                        activation=activation, 
                        kernel_initializer=kernel_initializer, 
                        name='mu')
            self.logstd = mlp(config.get('logstd_units', []),
                        out_dim=action_dim, 
                        norm=norm, 
                        activation=activation, 
                        kernel_initializer=kernel_initializer, 
                        name='logstd')
        
        self.action_repetition = mlp(config.get('ar_units', []), out_dim=self.max_ar, name='ar')
        self.ar_tau = 1 # tf.Variable(1., dtype=tf.float32, name='ar_softmax_tau')

        # action distribution type
        self.ActionDistributionType = Categorical if is_action_discrete else DiagGaussian

        # build for variable initialization
        # TensorSpecs = [(state_shape, tf.float32, 'state')]
        # self.action = build(self._action, TensorSpecs)

    @tf.function(experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def action(self, state):
        x = state
        with tf.name_scope('action'):
            x = self.intra_layers(x)

            if self.is_action_discrete:
                logits = self.logits(x)
                
                action_distribution = self.ActionDistributionType(logits, self.tau)
                action = action_distribution.sample(reparameterize=False, one_hot=False)
                action_repr = tf.one_hot(action, self.action_dim)
            else:
                mu = self.mu(x)
                logstd = self.logstd(x)
                logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)

                action_distribution = self.ActionDistributionType(mu, logstd)
                raw_action = action_distribution.sample()
                action_repr = action = tf.tanh(raw_action)

            ar_logits = self.action_repetition(tf.concat([state, action_repr], axis=-1))
            ar_distribution = Categorical(ar_logits, self.ar_tau)
            n = ar_distribution.sample(one_hot=False)

            return action, tf.squeeze(n)

    @tf.function(experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def det_action(self, state):
        x = state
        with tf.name_scope('det_action'):
            x = self.intra_layers(x)

            if self.is_action_discrete:
                logits = self.logits(x)
                action = tf.argmax(logits, axis=-1)
                action_repr = tf.one_hot(action, self.action_dim)
            else:
                mu = self.mu(x)
                action_repr = action = tf.tanh(mu)

            ar_logits = self.action_repetition(tf.concat([state, action_repr], axis=-1))
            n = tf.argmax(ar_logits, axis=-1)

        return action, tf.squeeze(n)

    @tf.Module.with_name_scope
    def train_action(self, state):
        x = state
        with tf.name_scope('train_action'):
            x = self.intra_layers(x)

            if self.is_action_discrete:
                logits = self.logits(x)
                action = tf.argmax(logits, axis=-1)
                action_repr = tf.one_hot(action, self.action_dim)
            else:
                mu = self.mu(x)
                action_repr = action = tf.tanh(mu)

            ar_logits = self.action_repetition(tf.concat([state, action_repr], axis=-1))
            n = tf.argmax(ar_logits, axis=-1)
            n = tf.one_hot(n, self.max_ar)

        return action, n

    @tf.Module.with_name_scope
    def train_det_step(self, state):
        """ For computing target value """
        x = state
        with tf.name_scope('train_det_step'):
            x = self.intra_layers(x)

            if self.is_action_discrete:
                logits = self.logits(x)
                action = tf.argmax(x, axis=-1)
                action = tf.one_hot(action, self.action_dim)
                
                action_distribution = self.ActionDistributionType(logits, self.tau)
                logpi = action_distribution.logp(action)
            else:
                mu = self.mu(x)
                logstd = self.logstd(x)
                logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)

                action_distribution = self.ActionDistributionType(mu, logstd)
                raw_action = mu
                raw_logpi = action_distribution.logp(raw_action)
                action = tf.tanh(raw_action)
                logpi = logpi_correction(raw_action, raw_logpi, is_action_squashed=False)

            ar_logits = self.action_repetition(tf.concat([state, tf.stop_gradient(action)], axis=-1))
            n = tf.argmax(ar_logits, axis=-1)
            n = tf.one_hot(n, self.max_ar)
            ar_distribution = Categorical(ar_logits, self.ar_tau)
            ar_logpi = ar_distribution.logp(n)

        return action, n, logpi, ar_logpi

    @tf.Module.with_name_scope
    def train_step(self, state):
        x = state
        with tf.name_scope('train_step'):
            x = self.intra_layers(x)
            terms = {}

            if self.is_action_discrete:
                logits = self.logits(x)
                
                action_distribution = self.ActionDistributionType(logits, self.tau)
                action = action_distribution.sample(reparameterize=True, hard=True)
                logpi = action_distribution.logp(action)
            else:
                mu = self.mu(x)
                logstd = self.logstd(x)
                logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)

                action_distribution = self.ActionDistributionType(mu, logstd)
                raw_action = action_distribution.sample(reparameterize=True)
                raw_logpi = action_distribution.logp(raw_action)
                action = tf.tanh(raw_action)
                logpi = logpi_correction(raw_action, raw_logpi, is_action_squashed=False)
                terms['std'] = action_distribution.std

            ar_logits = self.action_repetition(tf.concat([state, tf.stop_gradient(action)], axis=-1))
            ar_distribution = Categorical(ar_logits, self.ar_tau)
            n = ar_distribution.sample(reparameterize=True, hard=True)
            ar_logpi = ar_distribution.logp(n)

            terms['action_entropy'] = action_distribution.entropy()
            terms['ar_entropy'] = ar_distribution.entropy()

        return action, n, logpi, ar_logpi, terms

    def get_weights(self):
        return [v.numpy() for v in self.variables]

    def set_weights(self, weights):
        [v.assign(w) for v, w in zip(self.variables, weights)]


class SoftQ(tf.Module):
    def __init__(self, config, state_shape, action_dim, max_ar, name='q'):
        super().__init__(name=name)

        # parameters
        units_list = config['units_list']

        norm = config.get('norm')
        activation = config.get('activation', 'relu')
        kernel_initializer = config.get('kernel_initializer', 'he_uniform')

        """ Network definition """
        self.intra_layers = mlp(units_list, 
                                        out_dim=1,
                                        norm=norm, 
                                        activation=activation, 
                                        kernel_initializer=kernel_initializer)

        # build for variable initialization
        # TensorSpecs = [
        #     (state_shape, tf.float32, 'state'),
        #     ([action_dim], tf.float32, 'action'),
        #     ([max_ar], tf.float32, 'max_ar')
        # ]
        # self.step = build(self._step, TensorSpecs)

    @tf.function(experimental_relax_shapes=True)
    def step(self, x, a, n):
        return self.train_value(x, a, n)

    @tf.Module.with_name_scope
    def train_value(self, x, a, n):
        with tf.name_scope('step'):
            x = tf.concat([x, a, n], axis=-1)
            x = self.intra_layers(x)
            
        return x

    def get_weights(self):
        return [v.numpy() for v in self.variables]

    def set_weights(self, weights):
        [v.assign(w) for v, w in zip(self.variables, weights)]


class Temperature(tf.Module):
    def __init__(self, config, state_shape, action_dim, max_ar, name='temperature'):
        super().__init__(name=name)

        self.temp_type = config['temp_type']
        """ Network definition """
        if self.temp_type == 'state-action':
            self.intra_layer = layers.Dense(1)
        elif self.temp_type == 'variable':
            self.log_temp = tf.Variable(1.)
        else:
            raise NotImplementedError(f'Error temp type: {self.temp_type}')

        # build for variable initialization
        # TensorSpecs = [
        #     (state_shape, tf.float32, 'state'),
        #     ([action_dim], tf.float32, 'action'),
        #     ([max_ar], tf.float32, 'max_ar')
        # ]
        # self.step = build(self._step, TensorSpecs)

    @tf.function(experimental_relax_shapes=True)
    def step(self, x, a, n):
        return self.train_step(x, a, n)
    
    @tf.Module.with_name_scope
    def train_step(self, x, a, n):
        with tf.name_scope('step'):
            if self.temp_type == 'state-action':
                x = tf.concat([x, a, n], axis=-1)
                log_temp = self.intra_layer(x)
                temp = tf.exp(log_temp)
            else:
                log_temp = self.log_temp
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
    max_ar = actor_config['max_ar']
    actor = SoftPolicy(actor_config, state_shape, action_dim, is_action_discrete)
    q1 = SoftQ(q_config, state_shape, action_dim, max_ar, 'q1')
    q2 = SoftQ(q_config, state_shape, action_dim, max_ar, 'q2')
    target_q1 = SoftQ(q_config, state_shape, action_dim, max_ar, 'target_q1')
    target_q2 = SoftQ(q_config, state_shape, action_dim, max_ar, 'target_q2')
    if temperature_config['temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config, state_shape, action_dim, max_ar)
        
    return dict(
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        temperature=temperature,
    )
