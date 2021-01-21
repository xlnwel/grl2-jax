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


class RandomNet(Module):
    def __init__(self, name):
        super().__init__(name)

        conv_cls = tf.keras.layers.Conv2D
        kwargs = {
            'kernel_initializer': get_initializer('orthogonal', gain=np.sqrt(2)),
            'activation': tf.keras.layers.LeakyReLU()
        }
        self._layers = [
            conv_cls(32, 8, 4, **kwargs),
            conv_cls(64, 4, 2, **kwargs),
            conv_cls(64, 3, 1, **kwargs),
        ]
    
    def call(self, x):
        assert x.shape[-3:] == (84, 84, 1), x.shape
        x = super().call(x)
        x = tf.reshape(x, (*x.shape[:2], -1))
        return x

class Target(Module):
    def __init__(self, name='target'):
        super().__init__(name=name)

        kwargs = {
            'kernel_initializer': get_initializer('orthogonal', gain=np.sqrt(2)),
        }
        self._layers = [
            RandomNet(name),
            tf.keras.layers.Dense(512, **kwargs, dtype='float32')
        ]


class Predictor(Module):
    def __init__(self, name='predictor'):
        super().__init__(name=name)

        kwargs = {
            'kernel_initializer': get_initializer('orthogonal', gain=np.sqrt(2)),
            'activation': 'relu'
        }
        self._layers = [
            RandomNet(name),
            mlp([512, 512], out_size=512, **kwargs, 
                out_dtype='float32', out_gain=np.sqrt(2))
        ]

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