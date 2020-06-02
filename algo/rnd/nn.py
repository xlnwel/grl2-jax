import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.module import Module
from core.decorator import config
from utility.tf_distributions import DiagGaussian, Categorical, TanhBijector
from nn.func import cnn, mlp
from nn.block.cnn import convert_obs


class PPOAC(Module):
    @config
    def __init__(self, action_dim, is_action_discrete, name='ac'):
        super().__init__(name=name)

        self._is_action_discrete = is_action_discrete
        
        """ Network definition """
        if self._cnn_name:
            self._shared_layers = cnn(
                self._cnn_name, time_distributed=False, out_size=256)
        else:
            self._shared_layers = lambda x: x

        self.mlps = []
        for i in range(3):
            self.mlps.append(mlp([448], 
                            norm=self._norm, 
                            activation=self._activation, 
                            kernel_initializer=self._kernel_initializer,
                            name=f'mlp{i}'))

        self.actor = mlp(self._actor_units, 
                        out_size=action_dim, 
                        norm=self._norm,
                        activation=self._activation, 
                        kernel_initializer=self._kernel_initializer,
                        out_dtype='float32',
                        name='actor',
                        )
        if not self._is_action_discrete:
            self.logstd = tf.Variable(
                initial_value=np.log(self._init_std)*np.ones(action_dim), 
                dtype='float32', 
                trainable=True, 
                name=f'actor/logstd')
        self.value_int = mlp(self._critic_units, 
                            out_size=1,
                            norm=self._norm,
                            activation=self._activation, 
                            kernel_initializer=self._kernel_initializer,
                            out_dtype='float32',
                            name='value_int')
        self.value_ext = mlp(self._critic_units, 
                            out_size=1,
                            norm=self._norm,
                            activation=self._activation, 
                            kernel_initializer=self._kernel_initializer,
                            out_dtype='float32',
                            name='value_ext')

    def __call__(self, x, return_value=False):
        print(f'{self.name} is retracing: x={x.shape}')
        x = self._shared_layers(x)
        x = self.mlps[0](x)
        ax, vx = x, x
        ax = self.mlps[1](ax) + ax
        vx = self.mlps[2](vx) + vx
        actor_out = self.actor(ax)

        if self._is_action_discrete:
            act_dist = tfd.Categorical(actor_out)
        else:
            act_dist = tfd.MultivariateNormalDiag(actor_out, tf.exp(self.logstd))

        if return_value:
            value_int = tf.squeeze(self.value_int(vx), -1)
            value_ext = tf.squeeze(self.value_ext(vx), -1)
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
        self._dense = mlp(out_size=512, 
            kernel_initializer='orthogonal', activation='relu')

    def __call__(self, x):
        assert x.shape[-3:] == (84, 84, 1), x.shape
        x = self._cnn(x)
        shape = tf.concat([tf.shape(x)[:-3], [tf.reduce_prod(x.shape[-3:])]], 0)
        x = tf.reshape(x, shape)
        x = self._dense(x)

        return x


class Predictor(Module):
    def __init__(self, name='predictor'):
        super().__init__(name=name)

        self._cnn = cnn('nature', kernel_initializer='orthogonal', 
            time_distributed=True, out_size=None, activation='leaky_relu')
        self._dense = mlp([512, 512], out_size=512, 
            kernel_initializer='orthogonal', activation='relu')

    def __call__(self, x):
        assert x.shape[-3:] == (84, 84, 1), x.shape
        x = self._cnn(x)
        shape = tf.concat([tf.shape(x)[:-3], [tf.reduce_prod(x.shape[-3:])]], 0)
        x = tf.reshape(x, shape)
        x = self._dense(x)

        return x


def create_model(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    return dict(ac=PPOAC(config, action_dim, is_action_discrete),
                target=Target(),
                predictor=Predictor())

if __name__ == '__main__':
    env_config = dict(
        name='atari_MontezumaRevenge',
        n_workers=1,
        n_envs=8,
        seed=0,
        precision=32,
        frame_stack=4,
        np_obs=True
    )
    config = dict(
        cnn_name='nature',
        actor_units=[256],
        critic_units=[256],
        norm='none',
        kernel_initializer='orthogonal',
        activation='relu',
        init_std=1,)
    from env.gym_env import create_env
    env = create_env(env_config)
    models = create_model(config, env)

    obs = env.reset()[0]
    models['ac'](obs, True)
    obs = np.expand_dims(obs, 1)
    models['target'](obs[..., -1:], obs[0, ..., -1:], obs[0, ..., -1:])
    models['predictor'](obs[..., -1:], obs[0, ..., -1:], obs[0, ..., -1:])

    from core.decorator import display_model_var_info
    display_model_var_info(models)