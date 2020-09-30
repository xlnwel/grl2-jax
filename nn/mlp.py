import tensorflow as tf

from core.module import Module
from nn.registry import layer_registry
from nn.utils import get_initializer


class MLP(Module):
    def __init__(self, units_list, out_size=None, layer_type='dense', 
                norm=None, activation=None, kernel_initializer='glorot_uniform', 
                name=None, out_dtype=None, out_gain=1, **kwargs):
        super().__init__(name=name)
        layer_cls = layer_registry.get(layer_type)
        Layer = layer_registry.get('layer')
        # Follows OpenAI's baselines, which uses a small-scale initializer
        # for policy head when using orthogonal initialization
        # out_gain = kwargs.pop('out_gain', .01)

        self._layers = [
            Layer(u, layer_type=layer_cls, norm=norm, 
                activation=activation, kernel_initializer=kernel_initializer, 
                name=f'{name}/{layer_type}_{i}', **kwargs)
            for i, u in enumerate(units_list)]
        if out_size:
            kernel_initializer = get_initializer(kernel_initializer, gain=out_gain)
            self._layers.append(layer_cls(
                out_size, kernel_initializer=kernel_initializer, 
                dtype=out_dtype, name=f'{name}/out'))

    def reset(self):
        for l in self._layers:
            l.reset()
