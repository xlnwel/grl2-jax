import tensorflow as tf

from nn.layers import Layer 
from nn.utils import get_initializer


class MLP(tf.Module):
    def __init__(self, units_list, out_size=None, layer_type=tf.keras.layers.Dense, 
                norm=None, activation=None, kernel_initializer='glorot_uniform', 
                name=None, out_dtype=None, **kwargs):
        super().__init__(name=name)

        # Follows OpenAI's baselines, which uses a small-scale initializer
        # for policy head when using othogonal initialization
        out_gain = kwargs.pop('out_gain', .01)

        self._layers = [
            Layer(u, layer_type=layer_type, norm=norm, 
                activation=activation, kernel_initializer=kernel_initializer, 
                name=f'layer{i}', **kwargs)
            for i, u in enumerate(units_list)]
        if out_size:
            kernel_initializer = get_initializer(kernel_initializer, gain=out_gain)
            self._layers.append(layer_type(
                out_size, kernel_initializer=kernel_initializer, 
                dtype=out_dtype, name='out'))
            
    def __call__(self, x, **kwargs):
        if self.name:
            with tf.name_scope(self.name):
                return self.call(x, **kwargs)
        else:
            return self.call(x, **kwargs)

    def call(self, x, **kwargs):
        for l in self._layers:
            x = l(x)
        
        return x

    def reset(self):
        for l in self._layers:
            l.reset()
