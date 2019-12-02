import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utility.display import pwc
from core.tf_config import build
from utility.rl_utils import logpi_correction
from utility.tf_distributions import DiagGaussian, Categorical
from nn.layers.func import mlp_layers
from nn.initializers import get_initializer


class SAC(tf.Module):
    """ This class groups all models used by SAC together
    so that one can easily manipulate variables """
    def __init__(self, config, state_shape, action_dim, is_action_discrete, name='sac'):
        super().__init__(name=name)

        self.models = create_model(config, state_shape, action_dim, is_action_discrete)

    def get_weights(self):
        return [v.numpy() for v in self.variables]

    def set_weights(self, weights):
        [v.assign(w) for v, w in zip(self.variables, weights)]

    """ Auxiliary functions that make SAC like a dict """
    def __setitem__(self, key, item):
        self.models[key] = item

    def __getitem__(self, key):
        return self.models[key]

    def __len__(self):
        return len(self.models)
    
    def __iter__(self):
        return self.models.__iter__()

    def keys(self):
        return self.models.keys()

    def values(self):
        return self.models.values()
    
    def items(self):
        return self.models.items()


class SoftPolicy(tf.Module):
    def __init__(self, config, state_shape, action_dim, is_action_discrete, name='actor'):
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
                                        kernel_initializer=kernel_initializer())
        self.is_action_discrete = is_action_discrete

        if is_action_discrete:
            self.logits = layers.Dense(action_dim, name='logits')
            self.tau = tf.Variable(1., dtype=tf.float32, name='softmax_tau')
        else:
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
        # pwc(f'{self.name} "step" is retracing: x={x}', color='cyan')
        for l in self.intra_layers:
            x = l(x)

        if self.is_action_discrete:
            logits = self.logits(x)
            action_distribution = self.ActionDistributionType(logits, self.tau)
            action = action_distribution.sample(reparameterize=True)
        else:
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
        # pwc(f'{self.name} "det_action" is retracing: x={x}', color='cyan')
        with tf.name_scope('det_action'):
            for l in self.intra_layers:
                x = l(x)

            if self.is_action_discrete:
                logits = self.logits(x)
                action_distribution = self.ActionDistributionType(logits, self.tau)
                return action_distribution.sample(reparameterize=True, hard=True)
            else:
                mu = self.mu(x)
                return tf.tanh(mu)
           
    @tf.Module.with_name_scope
    def train_step(self, x):
        # pwc(f'{self.name} "train_step" is retracing: x={x}', color='cyan')
        for l in self.intra_layers:
            x = l(x)

        if self.is_action_discrete:
            logits = self.logits(x)
            action_distribution = self.ActionDistributionType(logits, self.tau)
            action = action_distribution.sample(reparameterize=True)
            logpi = action_distribution.logp(action)
        else:
            mu = self.mu(x)
            logstd = self.logstd(x)
            logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)

            action_distribution = self.ActionDistributionType((mu, logstd))
            
            raw_action = action_distribution.sample()
            logpi = action_distribution.logp(raw_action)
            action = tf.tanh(raw_action)
            logpi = logpi_correction(raw_action, logpi, is_action_squashed=False)

        return action, logpi

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
    temperature = Temperature(temperature_config, state_shape, action_dim)
    
    return dict(
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        temperature=temperature,
    )
    
if __name__ == '__main__':
    from utility.yaml_op import load_config
    config = load_config('algo/sac/config.yaml')
    config = config['model']
    state_shape = (2,)
    action_dim = 4
    is_action_discrete = False

    sac = SAC(config, state_shape, action_dim, is_action_discrete)

    assert len(sac.get_weights()) == len(sac.trainable_variables)
    target_vars = np.array([v.numpy() for v in sac.variables])

    for var, tvar in zip(sac.get_weights(), target_vars):
        np.testing.assert_allclose(var, tvar)

    for k, v in sac.items():
        print('model name:', k, v.name)

