import string
import tensorflow as tf

from core.module import Module
from nn.hyper import HyperNet
from nn.func import mlp, nn_registry
from utility.utils import dict2AttrDict


@nn_registry.register('index')
class IndexedNet(Module):
    def __init__(self, name='indexed_head', **config):
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
        self._wlayer = HyperNet(**w_config, name=f'{self._raw_name}_w')

        if self.use_bias:
            b_config = config.copy()
            b_config['w_in'] = None
            b_config['w_out'] = self.out_size
            self._blayer = HyperNet(**b_config, name=f'{self._raw_name}_b')

    def call(self, x, hx):
        w = self._wlayer(hx)
        pre = string.ascii_lowercase[:len(x.shape)-1]
        eqt = f'{pre}h,{pre}ho->{pre}o'
        out = tf.einsum(eqt, x, w)
        if self.use_bias:
            out = out + self._blayer(hx)

        return out
