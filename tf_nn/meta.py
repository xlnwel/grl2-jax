import logging
import tensorflow as tf

from core.log import do_logging
from core.ensemble import Module
from nn.registry import nn_registry
from nn.utils import get_activation
from core.typing import AttrDict
from tools.utils import dict2AttrDict

logger = logging.getLogger(__name__)


@nn_registry.register('meta')
class MetaParams(Module):
    def __init__(self, name='meta_params', **config):
        super().__init__(name=name)

        self.config = dict2AttrDict(config)
        self.n_meta_params = 0
        for k, v in self.config.items():
            do_logging(f'{k}: {v}', prefix=name, logger=logger)
            if isinstance(v, (int, float)):
                self.config[k] = AttrDict(
                    outer=v, 
                    default=v, 
                    init=None, 
                )
            elif v.init is not None:
                init = v.init * tf.ones(v.shape) if v.get('shape') else v.init
                setattr(self, f'{k}_var', tf.Variable(
                    init, dtype='float32', name=f'{name}/{k}'))
                setattr(self, f'{k}_act', get_activation(v.act))
                self.n_meta_params += 1
        
        self.params = list(config)

    def __call__(self, name, inner):
        if inner:
            if self.config[name].init is None:
                return float(self.config[name].default)
            scale = float(self.config[name].scale)
            bias = float(self.config[name].bias)
            var = getattr(self, f'{name}_var')
            act = getattr(self, f'{name}_act')
            
            return scale * act(var) + bias
        else:
            val = float(self.config[name].outer)
            return val

    def get_var(self, name):
        var = getattr(self, f'{name}_var', None)
        return var
