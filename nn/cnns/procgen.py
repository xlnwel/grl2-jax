from core.module import Module
from nn.registry import cnn_registry, subsample_registry, block_registry
from nn.utils import *


@cnn_registry.register('procgen')
class ProcgenCNN(Module):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 obs_range=[0, 1], 
                 filters=[16, 32, 32],
                 n_blocks=[2, 2, 2],
                 kernel_initializer='glorot_uniform',
                 stem='conv_maxblurpool',
                 stem_kwargs={},
                 subsample='strided_resv1',
                 subsample_kwargs={},
                 block='resv1',
                 block_kwargs: dict(
                    filter_coefs=[],
                    kernel_sizes=[3, 3],
                    norm=None,
                    norm_kwargs={},
                    activation='relu',
                    am_type='se',
                    am_kwargs={},
                    dropout_rate=0.,
                    rezero=False,
                 ),
                 sa=None,
                 sa_pos=[],
                 sa_kwargs={},
                 out_activation='relu',
                 out_size=None,
                 name='procgen',
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range
        self._time_distributed = time_distributed

        # kwargs specifies general kwargs for conv2d
        kwargs['kernel_initializer'] = get_initializer(kernel_initializer)
        assert 'activation' not in kwargs, kwargs

        stem_cls = subsample_registry.get(stem)
        stem_kwargs.update(kwargs.copy())
        
        block_cls = block_registry.get(block)
        block_kwargs.update(kwargs.copy())

        subsample_cls = subsample_registry.get(subsample)
        subsample_kwargs.update(kwargs.copy())

        sa_cls = block_registry.get(sa)
        sa_kwargs.update(kwargs.copy())

        self._layers = []
        prefix = f'{self.scope_name}/'
        with self.name_scope:
            for i, (f, n) in enumerate(zip(filters, n_blocks)):
                subsample_kwargs['filters'] = [f for _ in range(2)]
                self._layers += [
                    stem_cls(filters[0], name=prefix+stem, **stem_kwargs) if i == 0 else
                    subsample_cls(name=f'{prefix}{subsample}_{i}_f{f}', **subsample_kwargs),
                ]+ [block_cls(name=f'{prefix}{block}_{i}', **block_kwargs) for n in range(n_blocks[0]-1)]
                if i in sa_pos:
                    self._layers += [
                        sa_cls(name=f'{prefix}{sa}_{i}', **sa_kwargs)
                    ]
            out_act_cls = get_activation(out_activation, return_cls=True)
            self._layers.append(out_act_cls(name=prefix+out_activation))
            self._flat = layers.Flatten(name=prefix+'flatten')

            self.out_size = out_size
            if self.out_size:
                self._dense = layers.Dense(self.out_size, activation=self._out_act, name=prefix+'out')
        
        self._training_cls += [block_cls, subsample_cls, sa_cls]
    
    def call(self, x, training=False, return_cnn_out=False):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        if self._time_distributed:
            t = x.shape[1]
            x = tf.reshape(x, [-1, *x.shape[2:]])
        x = super().call(x, training=training)
        if self._time_distributed:
            x = tf.reshape(x, [-1, t, *x.shape[1:]])
        z = self._flat(x)
        if self.out_size:
            z = self._dense(z)
        if return_cnn_out:
            return z, x
        else:
            return z