import string
import jax.numpy as jnp
import haiku as hk
import chex

from nn.func import mlp, nn_registry
from nn.layers import Layer
from nn.utils import get_initializer
from core.typing import dict2AttrDict


class FixedInitializer(hk.initializers.Initializer):
  """Initializes by sampling from a normal distribution."""

  def __init__(self, init):
    """Constructs a :class:`RandomNormal` initializer.
    Args:
      stddev: The standard deviation of the normal distribution to sample from.
      mean: The mean of the normal distribution to sample from.
    """
    self.init = init

  def __call__(self, shape, dtype):
    chex.assert_shape(self.init, shape)
    chex.assert_type(self.init, dtype)
    return self.init


class IndexParams(hk.Module):
    def __init__(
        self, 
        *, 
        w_in, 
        w_out, 
        w_init='orthogonal', 
        use_bias, 
        scale=1.,
        name='index_params', 
    ):
        super().__init__(name=name)

        self.w_in = w_in
        self.w_out = w_out
        self.w_init = get_initializer(w_init, scale=scale)
        self.use_bias = use_bias

    def __call__(self, x):
        w, b = self.build_net(x)

        pre = string.ascii_lowercase[:len(x.shape)-1]
        eqt = f'{pre}h,hio->{pre}io'
        x = jnp.einsum(eqt, x, w)
        if self.w_in is None:
            x = jnp.reshape(x, (*x.shape[:-2], self.w_out))
        if self.use_bias:
            x = x + b

        return x

    @hk.transparent
    def build_net(self, x):
        w_in = 1 if self.w_in is None else self.w_in
        inits = [self.w_init((w_in, self.w_out), x.dtype) 
            for _ in range(x.shape[-1])]
        init = jnp.stack(inits)
        init_shape = init.shape
        init = FixedInitializer(init)
        w = hk.get_parameter('w', shape=init_shape, init=init)
        if self.use_bias:
            init_shape = (self.w_out,) if self.w_in is None else (w_in, self.w_out)
            init = get_initializer('zero')
            b = hk.get_parameter('b', shape=init_shape, init=init)
        else:
            b = None

        return w, b


@nn_registry.register('index')
class IndexLayer(hk.Module):
    def __init__(
        self, 
        out_size, 
        use_bias=True, 
        use_shared_bias=False, 
        name='index_layer', 
        **config
    ):
        super().__init__(name=name)
        
        self.config = dict2AttrDict(config, to_copy=True)
        self.out_size = out_size
        self.use_bias = use_bias
        self.use_shared_bias = use_shared_bias
 
    def __call__(self, x, hx):
        assert x is not None, x
        assert hx is not None, hx
        wlayer, blayer = self.build_net(x)
        
        w = wlayer(hx)
        pre = string.ascii_lowercase[:len(x.shape)-1]
        eqt = f'{pre}h,{pre}ho->{pre}o'
        out = jnp.einsum(eqt, x, w)
        if self.use_bias:
            out = out + blayer(hx)

        return out

    @hk.transparent
    def build_net(self, x):
        config = self.config.copy()
        config['use_bias'] = self.use_shared_bias  # no bias to avoid any potential parameter sharing
        w_config = config.copy()
        w_config['w_in'] = x.shape[-1]
        w_config['w_out'] = self.out_size
        wlayer = IndexParams(**w_config, name='w')

        if self.use_bias:
            b_config = config.copy()
            b_config['w_in'] = None
            b_config['w_out'] = self.out_size
            # b_config['scale'] = 1e-3
            # b_config['w_init'] = 'zeros'
            blayer = IndexParams(**b_config, name='b')
        else:
            blayer = None

        return wlayer, blayer


class IndexModule(hk.Module):
    @hk.transparent
    def __init__(self, config, out_size, name=None):
        super().__init__(name=name)

        self.config = dict2AttrDict(config, to_copy=True)
        self.out_size = out_size
        self.index = self.config.pop('index', None)
        self.index_config = self.config.pop('index_config', {})
        self.index_config['w_init'] = self.config.get('w_init', 'orthogonal')

    def __call__(self, x, hx):
        layers = self.build_net()

        for l in layers:
            x = l(x, hx)

        return x

    @hk.transparent
    def build_net(self):
        if self.index == 'all':
            units_list = self.config.pop('units_list', [])
            layers = []
            for i, u in enumerate(units_list):
                layers += [IndexLayer(
                    **self.index_config, out_size=u, name=f'index_layer{i}'), 
                    Layer(layer_type=None, **self.config)]
            self.index_config['scale'] = self.config.pop('out_scale', 1.)
            layers.append(IndexLayer(
                **self.index_config, 
                out_size=self.out_size, 
                name='out'
            ))
        elif self.index == 'head':
            self.index_config['scale'] = self.config.pop('out_scale', 1.)
            layers = [mlp(**self.config)]
            
            layers.append(IndexLayer(
                **self.index_config, 
                out_size=self.out_size, 
                name='out'
            ))
        else:
            layers = [mlp(
                **self.config, 
                out_size=self.out_size, 
            )]

        return layers


if __name__ == '__main__':
    import jax
    config = {
        'units_list': [2], 
        'w_init': 'orthogonal', 
        'activation': 'relu', 
        'norm': 'layer', 
        'eval_act_temp': 1, 
        'out_scale': .01, 
        'index': 'head', 
        'index_config': {
            'use_shared_bias': False, 
            'use_bias': True, 
            'w_init': 'orthogonal', 
        },
    }

    def layer(x, hx):
        layer = IndexModule(config, out_size=2)
        return layer(x, hx)
    rng = jax.random.PRNGKey(42)
    hx = jax.nn.one_hot([2, 3], 3)
    x = jax.random.normal(rng, (2, 3))
    net = hk.transform(layer)
    params = net.init(rng, x, hx)
    # print(params)
    # print(net.apply(params, rng, x, hx))
    print(hk.experimental.tabulate(net)(x, hx))
