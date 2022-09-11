import logging
import jax
import haiku as hk

from core.log import do_logging
from nn.layers import Layer
from nn.registry import layer_registry, nn_registry
from nn.utils import get_initializer


logger = logging.getLogger(__name__)


@nn_registry.register('mlp')
class MLP(hk.Module):
    def __init__(
        self, 
        units_list=[], 
        out_size=None, 
        layer_type='linear', 
        norm=None, 
        activation=None, 
        w_init='glorot_uniform', 
        name=None, 
        out_scale=1, 
        norm_after_activation=False, 
        norm_kwargs={
            'axis': -1, 
            'create_scale': True, 
            'create_offset': True, 
        }, 
        **kwargs
    ):
        super().__init__(name=name)
        if activation is None and (len(units_list) > 1 or (units_list and out_size)):
            do_logging(f'MLP({name}) with units_list({units_list}) and out_size({out_size}) has no activation.', 
                logger=logger, level='pwc')

        self.units_list = units_list
        self.layer_kwargs = dict(
            layer_type=layer_type, 
            norm=norm, 
            activation=activation, 
            w_init=w_init, 
            norm_after_activation=norm_after_activation, 
            norm_kwargs=norm_kwargs, 
            **kwargs
        )

        self.out_size = out_size
        kwargs.pop('scale', None)
        do_logging(f'{self.name} out scale: {out_scale}', logger=logger, level='info')
        w_init = get_initializer(w_init, scale=out_scale)
        self.out_kwargs = dict(
            layer_type=layer_type, 
            w_init=w_init, 
            name='out', 
            **kwargs
        )

    def __call__(self, x, is_training=True):
        layers = self.build_net()

        for l in layers:
            x = l(x, is_training)

        return x
    
    @hk.transparent
    def build_net(self):
        layers = []

        for u in self.units_list:
            layers.append(Layer(u, **self.layer_kwargs))
        if self.out_size:
            layers.append(Layer(self.out_size, **self.out_kwargs))
        
        return layers


if __name__ == '__main__':
    config = dict(
        units_list=[2, 3], 
        w_init='orthogonal', 
        scale=1, 
        activation='relu', 
        norm='layer', 
        name='mlp', 
        out_scale=.01, 
        out_size=1
    )
    def mlp(x):
        layer = MLP(**config)
        return layer(x)
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (2, 3))
    net = hk.transform(mlp)
    params = net.init(rng, x)
    print(params)
    print(net.apply(params, None, x))
    print(hk.experimental.tabulate(net)(x))