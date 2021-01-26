import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers

from core.module import Module, Ensemble
from core.decorator import config
from utility.rl_utils import logpi_correction
from utility.tf_distributions import Categorical
from nn.utils import flatten, convert_obs, get_initializer
from nn.func import mlp, cnn


def center_crop_image(image, output_size):
    h, w = image.shape[1:3]
    new_h, new_w = output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    assert image.shape[1:3] == output_size, image.shape
    return image


class Encoder(Module):
    @config
    def __init__(self, name='encoder'):
        super().__init__(name=name)
        ki = get_initializer('glorot_uniform', gain=np.sqrt(2))
        self._convs = tf.keras.Sequential([
            layers.Conv2D(32, 3, 2, activation='relu', kernel_initializer=ki),
            layers.Conv2D(32, 3, 1, activation='relu', kernel_initializer=ki),
            layers.Conv2D(32, 3, 1, activation='relu', kernel_initializer=ki),
            layers.Conv2D(32, 3, 1, activation='relu', kernel_initializer=ki),
        ])

        self._dense = mlp([self._z_size], norm=self._norm, activation=self._dense_act)
        
    @tf.Module.with_name_scope
    def cnn(self, x):
        x = center_crop_image(x)
        x = convert_obs(x, [0, 1])
        x = self._convs(x)
        x = flatten(x)
        return x
    
    @tf.Module.with_name_scope
    def mlp(self, x):
        x = self._dense(x)
        return x


class Actor(Module):
    @config
    def __init__(self, action_dim, is_action_discrete, name='actor'):
        super().__init__(name=name)
        self._is_action_discrete = is_action_discrete
        
        out_size = action_dim if is_action_discrete else 2*action_dim
        self._z = mlp([self._z_size], norm='layer')
        self._layers = mlp(self._units_list, 
                            out_size=out_size,
                            norm=self._norm, 
                            activation=self._activation)

    @tf.function(experimental_relax_shapes=True)
    def action(self, x, evaluation=False, epsilon=0):
        x = self._z(x)
        x = self._layers(x)

        if self._is_action_discrete:
            dist = tfd.Categorical(logits=x)
            action = dist.mode() if evaluation else dist.sample()
            if epsilon:
                rand_act = tfd.Categorical(tf.zeros_like(dist.logits)).sample()
                action = tf.where(
                    tf.random.uniform(action.shape[:-1], 0, 1) < epsilon,
                    rand_act, action)
        else:
            mu, logstd = tf.split(x, 2, -1)
            logstd = tf.tanh(logstd)
            logstd = .5 * (logstd + 1.) / (self.LOG_STD_MAX - self.LOG_STD_MIN) + self.LOG_STD_MIN
            std = tf.exp(logstd)
            dist = tfd.MultivariateNormalDiag(mu, std)
            raw_action = dist.mode() if evaluation else dist.sample()
            action = tf.tanh(raw_action)
            if epsilon:
                action = tf.clip_by_value(
                    tfd.Normal(action, epsilon).sample(), -1, 1)

        return action

    @tf.Module.with_name_scope
    def train_step(self, x):
        x = self._z(x)
        x = self._layers(x)

        if self._is_action_discrete:
            dist = Categorical(logits=x)
            action = dist.sample(one_hot=True)
            logpi = dist.log_prob(action)
            terms = {}
        else:
            mu, logstd = tf.split(x, 2, -1)
            logstd = tf.tanh(logstd)
            logstd = .5 * (logstd + 1.) / (self.LOG_STD_MAX - self.LOG_STD_MIN) + self.LOG_STD_MIN
            std = tf.exp(logstd)
            dist = tfd.MultivariateNormalDiag(mu, std)
            raw_action = dist.sample()
            raw_logpi = dist.log_prob(raw_action)
            action = tf.tanh(raw_action)
            tf.summary.histogram('std', std)
            tf.summary.histogram('raw_logpi', raw_logpi)
            logpi = logpi_correction(raw_action, raw_logpi, is_action_squashed=False)
            tf.summary.histogram('logpi', logpi)
            terms = dict(raw_act_std=std)
            
        terms['entropy']=dist.entropy()

        return action, logpi, terms


class Q(Module):
    @config
    def __init__(self, name='q'):
        super().__init__(name=name)

        self._layers = mlp(self._units_list, 
                            out_size=1,
                            norm=self._norm, 
                            activation=self._activation)

    def call(self, x, a):
        x = tf.concat([x, a], axis=-1)
        x = self._layers(x)
        x = tf.squeeze(x)
            
        return x


class Temperature(Module):
    @config
    def __init__(self, name='temperature'):
        super().__init__(name=name)

        if self._temp_type == 'variable':
            self._log_temp = tf.Variable(np.log(self._value), dtype=tf.float32)
        else:
            raise NotImplementedError(f'Error temp type: {self._temp_type}')
    
    def call(self):
        log_temp = self._log_temp
        temp = tf.exp(log_temp)
    
        return temp

class ContrastiveRepresentationLearning(Module):
    @config
    def __init__(self, name='crl'):
        super().__init__(name=name)
        # self._layers = mlp(self._units_list, activation=self._activation)
        self._W = tf.Variable(tf.random.uniform((self._z_size, self._z_size)))
    
    def call(self, x_anchor, x_pos):
        # x_anchor = self._layers(x_anchor)
        # x_pos = self._layers(x_pos)
        x_pos = tf.stop_gradient(x_pos)
        Wx = tf.matmul(self._W, tf.transpose(x_pos))
        logits = tf.matmul(x_anchor, Wx)
        logits = logits - tf.reduce_max(logits, -1, keepdims=True)
        return logits


class CURL(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, evaluation=False, epsilon=0):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape

        x = self.encoder(x)
        action = self.actor(x, evaluation=evaluation, epsilon=epsilon)
        action = tf.squeeze(action)

        return action

    @tf.function
    def value(self, x):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape
        
        x = self.encoder(x)
        value = self.q(x)
        value = tf.squeeze(value)
        
        return value


def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete
    encoder_config = config['encoder']
    actor_config = config['actor']
    q_config = config['q']
    temperature_config = config['temperature']
    curl_config = config['curl']
    if temperature_config['temp_type'] == 'constant':
        temperature = temperature_config['value']
    else:
        temperature = Temperature(temperature_config)
        
    return dict(
        encoder=Encoder(encoder_config, 'encoder'),
        target_encoder=Encoder(encoder_config, 'target_encoder'),
        actor=Actor(actor_config, action_dim, is_action_discrete),
        q=Q(q_config, 'q'),
        target_q=Q(q_config, 'target_q'),
        temperature=temperature,
        crl=ContrastiveRepresentationLearning(curl_config, 'curl')
    )

def create_model(config, env, **kwargs):
    return CURL(config, env, **kwargs)
