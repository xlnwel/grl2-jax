import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.tf_utils import assert_shape_compatibility, softmax
from utility.rl_utils import epsilon_greedy
from core.module import Module, Ensemble
from nn.func import Encoder, mlp


class Q(Module):
    def __init__(self, config, action_dim, name='q'):
        super().__init__(name=name)

        self._action_dim = action_dim
        self._duel = config.pop('duel', False)
        self._layer_type = config.get('layer_type', 'dense')
        self._stoch_action = config.pop('stoch_action', False)

        """ Network definition """
        if self._duel:
            self._v_layers = mlp(
                **config,
                out_size=1, 
                name=name+'/v',
                out_dtype='float32')
        self._layers = mlp(
            **config, 
            out_size=action_dim, 
            name=name,
            out_dtype='float32')

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def stoch_action(self):
        return self._stoch_action

    def action(self, x, noisy=True, reset=True, temp=1, return_stats=False):
        qs = self.call(x, noisy=noisy, reset=reset)

        if self._stoch_action:
            probs = softmax(qs, temp)
            self.dist = tfd.Categorical(probs=probs)
            self._action = action = self.dist.sample()
            one_hot = tf.one_hot(action, qs.shape[-1])
        else:
            self._action = action = tf.argmax(qs, axis=-1, output_type=tf.int32)
        if return_stats:
            if self._stoch_action:
                one_hot = tf.one_hot(action, qs.shape[-1])
                q = tf.reduce_sum(qs * one_hot, axis=-1)
            else:
                q = tf.reduce_max(qs, axis=-1)
            return action, {'q': tf.squeeze(q)}
        else:
            return action
    
    def call(self, x, action=None, noisy=True, reset=True):
        kwargs = dict(noisy=noisy, reset=reset) if self._layer_type == 'noisy' else {}

        if self._duel:
            v = self._v_layers(x, **kwargs)
            a = self._layers(x, **kwargs)
            q = v + a - tf.reduce_mean(a, axis=-1, keepdims=True)
        else:
            q = self._layers(x, **kwargs)

        if action is not None:
            if action.dtype.is_integer:
                action = tf.one_hot(action, self.action_dim, dtype=q.dtype)
            assert_shape_compatibility([action, q])
            q = tf.reduce_sum(q * action, -1)
        return q

    def reset_noisy(self):
        if self._layer_type == 'noisy':
            if self._duel:
                self._v_layers.reset()
            self._layers.reset()
    
    def compute_prob(self):
        return self.dist.prob(self._action) if self._stoch_action else 1


class DQN(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, 
            evaluation=False, 
            epsilon=0,
            temp=1.,
            return_stats=False,
            return_eval_stats=False):
        assert x.shape.ndims in (2, 4), x.shape

        x = self.encoder(x)
        noisy = not evaluation
        action = self.q.action(x, noisy=noisy, reset=noisy, temp=temp, return_stats=return_stats)
        terms = {}
        if return_stats:
            action, terms = action
        if isinstance(epsilon, tf.Tensor) or epsilon:
            action = epsilon_greedy(action, epsilon,
                is_action_discrete=True, 
                action_dim=self.q.action_dim)
        action = tf.squeeze(action)

        return action, terms


def create_components(config, env, **kwargs):
    action_dim = env.action_dim
    return dict(
        encoder=Encoder(config['encoder'], name='encoder'),
        q=Q(config['q'], action_dim, name='q'),
        target_encoder=Encoder(config['encoder'], name='target_encoder'),
        target_q=Q(config['q'], action_dim, name='target_q'),
    )

def create_model(config, env, **kwargs):
    return DQN(config, env, **kwargs)
