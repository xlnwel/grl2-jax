from tensorflow.keras import layers

from nn.layers import Layer 


class MLP(layers.Layer):
    def __init__(self, units_list, out_dim=None, layer_type=layers.Dense, 
                norm=None, activation=None, kernel_initializer='glorot_uniform', 
                name=None, **kwargs):
        super().__init__(name=name)

        self.intra_layers = [
            Layer(u, layer_type=layer_type, norm=norm, 
                    activation=activation, kernel_initializer=kernel_initializer, 
                    **kwargs)
            for u in units_list]
        if out_dim:
            self.intra_layers.append(layer_type(out_dim))
            
    def call(self, x, **kwargs):
        for l in self.intra_layers:
            x = l(x, **kwargs)
        
        return x

    def reset(self):
        for l in self.intra_layers:
            l.reset()
