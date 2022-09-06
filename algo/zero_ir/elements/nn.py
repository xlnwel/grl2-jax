from algo.zero.elements.nn import *


@nn_registry.register('reward')
class Reward(IndexedModule):
    def __init__(self, name='reward', **config):
        super().__init__(name=name)
        config = config.copy()
        
        config.setdefault('out_gain', 1)
        self._out_scale = config.pop('out_scale', 1)
        self._out_act = config.pop('out_act', None)
        if self._out_act:
            self._out_act = get_activation(self._out_act)
        self._out_size = config.pop('out_size', 1)
        self._combine_sa = config.pop('combine_sa', False)

        self._build_nets(config, out_size=self._out_size)

    def call(self, x, action, hx=None):
        if len(action.shape) < len(x.shape):
            action = tf.one_hot(action, self._out_size)

        if self._combine_sa:
            x = tf.concat([x, action], -1)
        x = super().call(x, hx)
        out = x

        if not self._combine_sa:
            x = tf.reduce_sum(x * action, -1)
        if x.shape[-1] == 1:
            x = tf.squeeze(x, -1)
        reward = x * self._out_scale
        if self._out_act is not None:
            reward = self._out_act(reward)
        reward = reward * self._out_scale

        return out, reward
