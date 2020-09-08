from nn.utils import *
from nn.block.cnns.utils import *


@register_cnn('nature')
class NatureCNN(layers.Layer):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 obs_range=[0, 1], 
                 name='nature', 
                 kernel_initializer='glorot_uniform',
                 activation='relu',
                 out_size=512,
                 padding='valid',
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        kwargs['time_distributed'] = time_distributed
        gain = kwargs.pop('gain', calculate_gain(activation))
        kwargs['kernel_initializer'] = get_initializer(kernel_initializer, gain=gain)
        activation = get_activation(activation)
        kwargs['activation'] = activation
        kwargs['padding'] = padding

        self._conv_layers = [
            conv2d(32, 8, 4, **kwargs),
            conv2d(64, 4, 2, **kwargs),
            conv2d(64, 3, 1, **kwargs),
        ]
        self.out_size = out_size
        if out_size:
            self._dense = layers.Dense(self.out_size, activation=activations.relu)

    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        for l in self._conv_layers:
            x = l(x)
        x = flatten(x)
        if self.out_size:
            x = self._dense(x)
        
        return x