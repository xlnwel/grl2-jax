import tensorflow as tf

from nn.layers import Layer 


class MLP(tf.Module):
    def __init__(self, units_list, out_dim=None, layer_type=tf.keras.layers.Dense, 
                norm=None, activation=None, kernel_initializer='glorot_uniform', 
                name=None, out_dtype=None, **kwargs):
        super().__init__(name=name)

        self._layers = [
            Layer(u, layer_type=layer_type, norm=norm, 
                activation=activation, kernel_initializer=kernel_initializer, 
                name=f'layer_{i}', **kwargs)
            for i, u in enumerate(units_list)]
        if out_dim:
            self._layers.append(layer_type(out_dim, dtype=out_dtype, name='out'))
            
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
