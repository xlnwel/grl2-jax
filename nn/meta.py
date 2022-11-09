import logging
from netrc import netrc
import haiku as hk

from core.log import do_logging
from nn.func import mlp
from nn.registry import nn_registry
from nn.utils import get_activation
from core.typing import AttrDict, dict2AttrDict

logger = logging.getLogger(__name__)


@nn_registry.register('meta')
class MetaParams(hk.Module):
    def __init__(self, name='meta_params', **config):
        super().__init__(name=name)

        self.config = dict2AttrDict(config, to_copy=True)
        for k, v in self.config.items():
            if isinstance(v, (int, float)):
                self.config[k] = AttrDict(
                    outer=v, 
                    default=v, 
                    init=None, 
                )
            else:
                self.config[k].act = get_activation(v.act)

    def __call__(self, inner, stats=AttrDict()):
        if inner:
            res = AttrDict()
            for k, v in self.config.items():
                if k in stats and v.net is not None:
                    var = self.build_net(v.net)(stats[k]).squeeze()
                    res[f'{k}_var'] = var
                    var = v.act(var)
                    if v.scale:
                        var = v.scale * var
                    if v.bias:
                        var = var + v.bias
                elif self.config[k].init is None:
                    var = float(v.default)
                else:
                    var = self.get_var(k)
                    res[f'{k}_var'] = var
                    var = v.act(var)
                    if v.scale:
                        var = v.scale * var
                    if v.bias:
                        var = var + v.bias
                res[k] = var
        else:
            res = dict2AttrDict({k: v.outer for k, v in self.config.items()})

        return res

    def get_var(self, name):
        var = hk.get_parameter(
            name, 
            self.config[name].get('shape', ()), 
            init=hk.initializers.Constant(self.config[name].init)
        )
        return var

    @hk.transparent
    def build_net(self, config):
        net = mlp(
            **config, 
            out_size=1
        )
        return net
