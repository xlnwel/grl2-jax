import tensorflow as tf

from utility.tf_utils import assert_shape_compatibility
from utility.rl_utils import epsilon_greedy
from core.module import Module, Ensemble
from core.decorator import config
from nn.func import Encoder, mlp


class Q(Module):
    @config
    def __init__(self, action_dim, name='q'):
        super().__init__(name=name)

        self.action_dim = action_dim

        """ Network definition """
        kwargs = {}
        kwargs = dict(
            kernel_initializer=getattr(self, '_kernel_initializer', 'glorot_uniform'),
            activation=getattr(self, '_activation', 'relu'),
            layer_type=getattr(self, '_layer_type', 'dense'),
            norm=getattr(self, '_norm', None),
            out_dtype='float32',
        )
        
        if self._duel:
            self._v_layers = mlp(
                self._units_list, 
                out_size=1, 
                name=name+'/v',
                **kwargs)
        self._layers = mlp(
            self._units_list, 
            out_size=action_dim, 
            name=name,
            **kwargs)

    def action(self, x, noisy=True, reset=True):
        q = self.call(x, noisy=noisy, reset=reset)
        return tf.argmax(q, axis=-1, output_type=tf.int32)
    
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
            return_stats=False,
            return_eval_stats=False):
        assert x.shape.ndims in (2, 4), x.shape

        x = self.encoder(x)
        noisy = not evaluation
        q = self.q(x, noisy=noisy, reset=False)
        action = tf.argmax(q, axis=-1, output_type=tf.int32)
        terms = {}
        if return_stats:
            terms = {'q': q}
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
