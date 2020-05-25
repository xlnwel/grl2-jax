import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.module import Module
from core.decorator import config
from utility.tf_distributions import DiagGaussian, Categorical, TanhBijector
from nn.func import cnn, mlp


class PPOAC(Module):
    @config
    def __init__(self, action_dim, is_action_discrete, name):
        super().__init__(name=name)

        self._is_action_discrete = is_action_discrete
        
        """ Network definition """
        if self._cnn_name:
            self._shared_layers = cnn(self._cnn_name, time_distributed=False)
        else:
            self._shared_layers = lambda x: x
        # actor/critic head
        self.actor = mlp(self._actor_units, 
                        out_dim=action_dim, 
                        norm=self._norm, 
                        name='actor', 
                        activation=self._activation, 
                        kernel_initializer=self._kernel_initializer)
        if not self._is_action_discrete:
            self.logstd = tf.Variable(
                initial_value=np.log(self._init_std)*np.ones(action_dim),
                dtype=global_policy().compute_dtype, 
                trainable=True, 
                name=f'actor/logstd')
        self.critic = mlp(self._critic_units, 
                            out_dim=1,
                            norm=self._norm, 
                            name='critic', 
                            activation=self._activation, 
                            kernel_initializer=self._kernel_initializer)

    def __call__(self, x, return_value=False):
        print(f'{self.name} is retracing: x={x.shape}')
        x = self._shared_layers(x)
        actor_out = self.actor(x)

        if self._is_action_discrete:
            act_dist = tfd.Categorical(actor_out)
        else:
            act_dist = tfd.MultivariateNormalDiag(actor_out, tf.exp(self.logstd))

        if return_value:
            value = tf.squeeze(self.critic(x), -1)
            return act_dist, value
        else:
            return act_dist

    def reset_states(self, **kwargs):
        return


def create_model(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete
    ac = PPOAC(config, action_dim, is_action_discrete, 'ac')

    return dict(ac=ac)
