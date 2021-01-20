import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import cnn, mlp
from nn.utils import get_initializer


class AC(Module):
    @config
    def __init__(self, action_dim, is_action_discrete, name='ac'):
        super().__init__(name=name)

        self._is_action_discrete = is_action_discrete
        
        """ Network definition """
        self._cnn = cnn(
            self._cnn_name, 
            time_distributed=False, 
            kernel_initializer=self._kernel_initializer,
            out_size=256)
        
        self._eval_act_temp = getattr(self, '_eval_act_temp', 0)
        
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
                        out_gain=.01,
                        name='actor')
        self._value_int = mlp(out_size=1,
                            activation=self._activation, 
                            kernel_initializer=self._kernel_initializer,
                            out_dtype='float32',
                            out_gain=.01,
                            name='value_int')
        self._value_ext = mlp(out_size=1,
                            activation=self._activation, 
                            kernel_initializer=self._kernel_initializer,
                            out_dtype='float32',
                            out_gain=.01,
                            name='value_ext')

    def call(self, x, return_value=False):
        print(f'{self.name} is retracing: x={x.shape}')
        x = self._cnn(x)
        x = self._mlps[0](x)
        ax, vx = x, x
        ax = self._mlps[1](ax) + ax
        actor_out = self._actor(ax)

        if self._is_action_discrete:
            act_dist = tfd.Categorical(actor_out)
        else:
            act_dist = tfd.MultivariateNormalDiag(actor_out, tf.exp(self.logstd))

        if return_value:
            vx = self._mlps[2](vx) + vx
            value_int = tf.squeeze(self._value_int(vx), -1)
            value_ext = tf.squeeze(self._value_ext(vx), -1)
            return act_dist, value_int, value_ext
        else:
            return act_dist

    def compute_value(self, x):
        x = self._cnn(x)
        x = self._mlps[0](x)
        x = self._mlps[2](x) + x
        value_int = tf.squeeze(self._value_int(x), -1)
        value_ext = tf.squeeze(self._value_ext(x), -1)
        return value_int, value_ext
        
    def reset_states(self, **kwargs):
        return
    
    def action(self, dist, evaluation):
        return dist.mode() if evaluation and self._eval_act_temp == 0 \
            else dist.sample()


class Target(Module):
    def __init__(self, name='target'):
        super().__init__(name=name)
        
        self._cnn = cnn(
            'nature', 
            kernel_initializer='orthogonal', 
            time_distributed=True, 
            out_size=None, 
            activation='leaky_relu')
        ki = get_initializer('orthogonal', gain=np.sqrt(2))
        self._out = tf.keras.layers.Dense(512, kernel_initializer=ki, dtype='float32')

    def call(self, x):
        assert x.shape[-3:] == (84, 84, 1), x.shape
        x = self._cnn(x)
        x = self._out(x)

        return x


class Predictor(Module):
    def __init__(self, name='predictor'):
        super().__init__(name=name)

        self._cnn = cnn(
            'nature', 
            kernel_initializer='orthogonal', 
            time_distributed=True, 
            out_size=None, 
            activation='leaky_relu')
        ki = get_initializer('orthogonal', gain=np.sqrt(2))
        self._mlp = mlp(
            [512, 512], 
            out_size=512,
            kernel_initializer=ki,
            gain=np.sqrt(2), 
            activation='relu',
            out_dtype='float32',
            out_gain=np.sqrt(2))

    def call(self, x):
        assert x.shape[-3:] == (84, 84, 1), x.shape
        x = self._cnn(x)
        x = self._mlp(x)

        return x


class RND(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, evaluation=False, return_eval_stats=False):
        if evaluation:
            act_dist = self.ac(x, return_value=False)
            return act_dist.mode()
        else:
            act_dist, value_int, value_ext = self.ac(x, return_value=True)
            action = act_dist.sample()
            logpi = act_dist.log_prob(action)
            return action, dict(logpi=logpi, value_int=value_int, value_ext=value_ext)

    @tf.function
    def compute_value(self, x):
        return self.ac.compute_value(x)

    def reset_states(self, **kwargs):
        return

    @property
    def state_keys(self):
        return None

def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    return dict(ac=AC(config['ac'], action_dim, is_action_discrete),
                target=Target(),
                predictor=Predictor())

def create_model(config, env, **kwargs):
    return RND(config, env, **kwargs)