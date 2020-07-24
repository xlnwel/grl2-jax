import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.module import Module, Ensemble
from core.decorator import config
from nn.func import cnn, mlp


class AC(Module):
    @config
    def __init__(self, action_dim, is_action_discrete, name):
        super().__init__(name=name)

        self._is_action_discrete = is_action_discrete
        
        """ Network definition """
        if self._cnn_name:
            self._shared_layers = cnn(self._cnn_name, time_distributed=False)
        else:
            self._shared_layers = lambda x: x

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
        self.critic = mlp(self._critic_units, 
                            out_size=1,
                            norm=self._norm,
                            activation=self._activation, 
                            kernel_initializer=self._kernel_initializer,
                            out_dtype='float32',
                            name='critic')

    def __call__(self, x, return_terms=False):
        print(f'{self.name} is retracing: x={x.shape}')
        x = self._shared_layers(x)
        actor_out = self.actor(x)

        if self._is_action_discrete:
            act_dist = tfd.Categorical(actor_out)
        else:
            act_dist = tfd.MultivariateNormalDiag(actor_out, tf.exp(self.logstd))

        if return_terms:
            value = tf.squeeze(self.critic(x), -1)
            return act_dist, value
        else:
            return act_dist

    def reset_states(self, **kwargs):
        return

    @property
    def state_keys(self):
        return None

class PPO(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, deterministic=False, epsilon=0):
        if deterministic:
            act_dist = self.ac(x, return_terms=False)
            action = tf.squeeze(act_dist.mode())
            return action
        else:
            act_dist, value = self.ac(x, return_terms=True)
            action = act_dist.sample()
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi, 'value': value}
            return action, terms    # keep the batch dimension for later use

def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete
    ac = AC(config, action_dim, is_action_discrete, name='ac')

    return dict(ac=ac)

def create_model(config, env, **kwargs):
    return PPO(config, env, **kwargs)
