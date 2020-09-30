import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.module import Module
from core.decorator import config
from utility.tf_distributions import DiagGaussian, Categorical, TanhBijector
from nn.func import cnn, mlp
from nn.utils import get_initializer
from nn.block.cnn import convert_obs


class AC(Module):
    @config
    def __init__(self, action_dim, is_action_discrete, name='ac'):
        super().__init__(name=name)

        self._is_action_discrete = is_action_discrete
        
        """ Network definition """
        self._cnn = cnn(
                self._cnn_name, time_distributed=False, out_size=256)
        
        self._mlps = []
        for i in range(3):
            self._mlps.append(mlp([448], 
                            activation=self._activation, 
                            kernel_initializer=self._kernel_initializer,
                            gain=np.sqrt(2) if i == 0 else .1,
                            name=f'mlp{i}'))
        self._actor = mlp(out_size=action_dim, 
                        activation=self._activation, 
                        kernel_initializer=self._kernel_initializer,
                        out_dtype='float32',
                        name='actor')
        self._value_int = mlp(out_size=1,
                            activation=self._activation, 
                            kernel_initializer=self._kernel_initializer,
                            out_dtype='float32',
                            name='value_int')
        self._value_ext = mlp(out_size=1,
                            activation=self._activation, 
                            kernel_initializer=self._kernel_initializer,
                            out_dtype='float32',
                            name='value_ext')

    def call(self, x, return_value=False):
        print(f'{self.name} is retracing: x={x.shape}')
        x = self._cnn(x)
        x = self._mlps[0](x)
        ax, vx = x, x
        ax = self._mlps[1](ax) + ax
        vx = self._mlps[2](vx) + vx
        actor_out = self._actor(ax)

        if self._is_action_discrete:
            act_dist = tfd.Categorical(actor_out)
        else:
            act_dist = tfd.MultivariateNormalDiag(actor_out, tf.exp(self.logstd))

        if return_value:
            value_int = tf.squeeze(self._value_int(vx), -1)
            value_ext = tf.squeeze(self._value_ext(vx), -1)
            return act_dist, value_int, value_ext
        else:
            return act_dist

    def reset_states(self, **kwargs):
        return


class Target(Module):
    def __init__(self, name='target'):
        super().__init__(name=name)
        
        self._cnn = cnn('nature', kernel_initializer='orthogonal', 
            time_distributed=True, out_size=None, activation='leaky_relu')
        ki = get_initializer('orthogonal', gain=np.sqrt(2))
        self._out = tf.keras.layers.Dense(512, kernel_initializer=ki, dtype='float32')

    def call(self, x):
        assert x.shape[-3:] == (84, 84, 1), x.shape
        x = self._cnn(x)
        shape = tf.concat([tf.shape(x)[:-3], [tf.reduce_prod(x.shape[-3:])]], 0)
        x = tf.reshape(x, shape)
        x = self._out(x)

        return x


class Predictor(Module):
    def __init__(self, name='predictor'):
        super().__init__(name=name)

        self._cnn = cnn('nature', kernel_initializer='orthogonal', 
            time_distributed=True, out_size=None, activation='leaky_relu')
        self._mlp = mlp([512, 512], 
            kernel_initializer='orthogonal', activation='relu')
        ki = get_initializer('orthogonal', gain=np.sqrt(2))
        self._out = tf.keras.layers.Dense(512, kernel_initializer=ki, dtype='float32')

    def call(self, x):
        assert x.shape[-3:] == (84, 84, 1), x.shape
        x = self._cnn(x)
        shape = tf.concat([tf.shape(x)[:-3], [tf.reduce_prod(x.shape[-3:])]], 0)
        x = tf.reshape(x, shape)
        x = self._mlp(x)
        x = self._out(x)

        return x


def create_model(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    return dict(ac=AC(config, action_dim, is_action_discrete),
                target=Target(),
                predictor=Predictor())
