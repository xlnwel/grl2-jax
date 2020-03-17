from tensorflow.keras import layers

from nn.layers import Layer 


class MLP(layers.Layer):
    def __init__(self, units_list, out_dim=None, layer_type=layers.Dense, 
                norm=None, activation=None, kernel_initializer='glorot_uniform', 
                name=None, out_dtype='float32', **kwargs):
        super().__init__(name=name)

        self._layers = [
            Layer(u, layer_type=layer_type, norm=norm, 
                activation=activation, kernel_initializer=kernel_initializer, 
                **kwargs)
            for u in units_list]
        if out_dim:
            self._layers.append(layer_type(out_dim, dtype=out_dtype))
            
    def call(self, x, **kwargs):
        for l in self._layers:
            x = l(x, **kwargs)
        
        return x

    def reset(self):
        for l in self._layers:
            l.reset()
