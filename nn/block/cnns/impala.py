from nn.utils import *
from nn.block.cnns.utils import *
from nn.block.cnns.res import ResidualV1, ResidualV2


_block_map = {
    'resv1': ResidualV1,
    'resv2': ResidualV2,
}

@register_cnn('impala')
class IMPALACNN(layers.Layer):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 obs_range=[0, 1], 
                 channels=[16, 32, 32],
                 kernel_initializer='glorot_uniform',
                 block='resv2',
                 block_filters=[],
                 block_kernel_sizes=[3, 3],
                 activation='relu',
                 out_size=None,
                 name='impala',
                 norm=None,
                 norm_kwargs={},
                 se_kwargs={},
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range

        kwargs['time_distributed'] = time_distributed
        gain = kwargs.pop('gain', calculate_gain(activation))
        kwargs['kernel_initializer'] = get_initializer(kernel_initializer, gain=gain)
        block_kwargs = kwargs.copy()
        block_kwargs['filters'] = block_filters
        block_kwargs['kernel_sizes']= block_kernel_sizes
        block_kwargs['activation'] = activation
        block_kwargs['norm'] = norm
        block_kwargs['norm_kwargs'] = norm_kwargs
        block_kwargs['se_kwargs'] = se_kwargs
        block = _block_map[block]
        self._activation = get_activation(activation)
        self._layers = []
        for i, c in enumerate(channels):
            self._layers += [
                conv2d(c, 3, strides=1, padding='same', **kwargs),
                maxpooling2d(3, strides=2, padding='same', time_distributed=time_distributed),
                block(name=f'block{i}_{c}_1', **block_kwargs),
                block(name=f'block{i}_{c}_2', **block_kwargs),
            ]

        self.out_size = out_size
        if self.out_size:
            self._dense = layers.Dense(self.out_size, activation=self._activation)
    
    def call(self, x, training=True, return_cnn_out=False):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        for l in self._layers:
            x = l(x)
        x = self._activation(x)
        z = flatten(x)
        if self.out_size:
            z = self._dense(z)
        if return_cnn_out:
            return z, x
        else:
            return z
