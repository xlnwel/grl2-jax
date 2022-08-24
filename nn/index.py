import string
import tensorflow as tf

from core.module import Module
from nn.hyper import HyperNet
from nn.func import mlp, nn_registry
from nn.utils import get_initializer
from utility.utils import dict2AttrDict


class IndexedLayer(Module):
    def __init__(self, name='indexed_layer', **config):
        super().__init__(name=name)
        self.scope_name = name

        self.w_in = config.pop('w_in')
        self.w_out = config.pop('w_out')
        self.n = config.pop('n', 1)
        initializer = config.pop('kernel_initializer', 'orthogonal')
        self.initializer = get_initializer(initializer, gain=config.pop('gain', 1.))
        self.use_bias = config.pop('use_bias')

    def build(self, x):
        w_in = 1 if self.w_in is None else self.w_in
        init = self.initializer((x[-1] * w_in, self.w_out))
        self.w = tf.Variable(
            init, 
            name=f'{self.scope_name}_w', 
            dtype=tf.float32, 
            trainable=True
        )
        if self.use_bias:
            self.b = tf.Variable(
                self.initializer((w_in, self.w_out)),
                name=f'{self.scope_name}_b', 
                dtype=tf.float32, 
                trainable=True
            )

    def call(self, x):
        tf.debugging.assert_equal(tf.reduce_sum(x, -1), tf.cast(self.n, tf.float32))
        w_in = 1 if self.w_in is None else self.w_in
        matrix = tf.reshape(self.w, (-1, w_in * self.w_out))
        x = tf.matmul(x, matrix)
        if self.w_in is None:
            x = tf.reshape(x, (-1, *x.shape[1:-1], self.w_out))
        else:
            x = tf.reshape(x, (-1, *x.shape[1:-1], self.w_in, self.w_out))
        if self.use_bias:
            x = x + self.b

        return x

@nn_registry.register('index')
class IndexedNet(Module):
    def __init__(self, name='indexed_net', **config):
        super().__init__(name=name)
        self._raw_name = name
        
        self.out_size = config.pop('out_size')
        self.use_bias = config.pop('use_bias', True)
        self.use_shared_bias = config.pop('use_shared_bias', False)
        self.config = dict2AttrDict(config)
    
    def build(self, x, hx):
        config = self.config.copy()
        config['use_bias'] = self.use_shared_bias  # no bias to avoid any potential parameter sharing
        w_config = config.copy()
        w_config['w_in'] = x[-1]
        w_config['w_out'] = self.out_size
        self._wlayer = IndexedLayer(**w_config, name=f'{self._raw_name}_w')

        if self.use_bias:
            b_config = config.copy()
            b_config['w_in'] = None
            b_config['w_out'] = self.out_size
            self._blayer = IndexedLayer(**b_config, name=f'{self._raw_name}_b')

    def call(self, x, hx):
        w = self._wlayer(hx)
        pre = string.ascii_lowercase[:len(x.shape)-1]
        eqt = f'{pre}h,{pre}ho->{pre}o'
        out = tf.einsum(eqt, x, w)
        if self.use_bias:
            out = out + self._blayer(hx)

        return out


class IndexedModule(Module):
    def _build_nets(self, config, out_size):
        self.indexed = config.pop('indexed', None)
        indexed_config = config.pop('indexed_config', {})
        indexed_config['kernel_initializer'] = config.get(
            'kernel_initializer', 'orthogonal')
        if self.indexed == 'all':
            assert indexed_config, self.scope_name
            units_list = config.pop('units_list', [])
            self._layers = [IndexedNet(
                **indexed_config, out_size=u, name=f'{self.scope_name}_l{i}') 
                for i, u in enumerate(units_list)]
            indexed_config['gain'] = config.pop('out_gain', 1.)
            self._layers.append(IndexedNet(
                **indexed_config, 
                out_size=out_size, 
                name=f'{self.scope_name}_out'
            ))
        elif self.indexed == 'head':
            assert indexed_config, indexed_config
            self._layers = mlp(
                **config, 
                name=self.scope_name
            )
            indexed_config['gain'] = config.pop('out_gain', 1.)
            self._head = IndexedNet(
                **indexed_config, 
                out_size=out_size, 
                out_dtype='float32',
                name=f'{self.scope_name}_out'
            )
        else:
            self._layers = mlp(
                **config, 
                out_size=out_size, 
                out_dtype='float32',
                name=self.scope_name
            )
