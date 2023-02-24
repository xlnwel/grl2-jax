import jax.numpy as jnp
import haiku as hk

from core.typing import dict2AttrDict
from nn.func import mlp, nn_registry
from algo.dynamics_tx.elements.utils import combine_sa
from algo.dynamics.elements.nn import Reward, Discount, \
    get_discrete_dist, get_normal_dist, DISCRETE_MODEL, CONTINUOUS_MODEL
""" Source this file to register Networks """


@nn_registry.register('model')
class Model(hk.Module):
    def __init__(
        self, 
        out_size, 
        in_config, 
        tx_config, 
        out_config, 
        name='model', 
    ):
        super().__init__(name=name)
        self.out_size = out_size

        self.in_config = dict2AttrDict(in_config, to_copy=True)
        self.tx_config = dict2AttrDict(tx_config, to_copy=True)
        self.out_config = dict2AttrDict(out_config, to_copy=True)
        self.out_type = self.out_config.pop('type')
        assert self.out_type in (DISCRETE_MODEL, CONTINUOUS_MODEL)

    def __call__(self, x, action, training=False):
        in_net, tx_net, out_net = self.build_net()
        x = combine_sa(x, action)
        x = self.call_net(in_net, tx_net, out_net, x, training=training)
        dist = self.get_dist(x)
        return dist

    @hk.transparent
    def build_net(self, name='model'):
        if self.out_type == DISCRETE_MODEL:
            out_size = self.out_size * self.out_config.n_classes
        else:
            out_size = self.out_size * 2
        in_net = mlp(**self.in_config, name=f'{name}_in')
        tx_net = nn_registry.get('tx')(
            **self.tx_config, name=f'{name}_tx'
        )
        out_net = mlp(out_size=out_size, name=f'{name}_out')
        return in_net, tx_net, out_net

    def call_net(self, in_net, tx_net, out_net, x, training):
        x = in_net(x)
        x = tx_net(x, training=training)
        x = out_net(x)
        return x
    
    def get_dist(self, x):
        if self.out_type == DISCRETE_MODEL:
            dist = get_discrete_dist(x, self.out_size, self.out_config.n_classes)
        else:
            dist = get_normal_dist(x, **self.out_config)
        
        return dist


@nn_registry.register('emodels')
class EnsembleModels(Model):
    def __init__(
        self, 
        n_models, 
        out_size, 
        in_config, 
        tx_config, 
        out_config, 
        name='emodel', 
    ):
        self.n_models = n_models
        super().__init__(out_size, in_config, tx_config, out_config, name=name)

    @hk.transparent
    def build_net(self):
        if self.out_type == 'discrete':
            out_size = self.out_size * self.out_config.n_classes
        else:
            out_size = self.out_size * 2
        in_net = [mlp(**self.in_config, name=f'model{i}_in') 
            for i in range(self.n_models)]
        tx_net = [nn_registry.get('tx')(
            **self.tx_config, name=f'model{i}_tx'
        ) for i in range(self.n_models)]
        out_net = [mlp(out_size=out_size, name=f'model{i}_out') 
            for i in range(self.n_models)]
        return in_net, tx_net, out_net

    def call_net(self, in_net, tx_net, out_net, x, training):
        ys = [net(x) for net in in_net]
        ys = [net(y, training=training) for net, y in zip(tx_net, ys)]
        ys = [net(y) for net, y in zip(out_net, ys)]
        x = jnp.stack(ys, -2)
        return x


if __name__ == '__main__':
    import jax
    # config = dict( 
    #     w_init='orthogonal', 
    #     scale=1, 
    #     activation='relu', 
    #     norm='layer', 
    #     out_scale=.01, 
    #     out_size=2
    # )
    # def layer_fn(x, *args):
    #     layer = HyperParamEmbed(**config)
    #     return layer(x, *args)
    # import jax
    # rng = jax.random.PRNGKey(42)
    # x = jax.random.normal(rng, (2, 3))
    # net = hk.transform(layer_fn)
    # params = net.init(rng, x, 1, 2, 3.)
    # print(params)
    # print(net.apply(params, None, x, 1., 2, 3))
    # print(hk.experimental.tabulate(net)(x, 1, 2, 3.))
    import os, sys

    config = {
        'units_list': [3], 
        'w_init': 'orthogonal', 
        'activation': 'relu', 
        'norm': None, 
        'out_scale': .01,
    }
    def layer_fn(x, *args):
        layer = EnsembleModels(5, 3, **config)
        return layer(x, *args)
    import jax
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (2, 3, 3))
    a = jax.random.normal(rng, (2, 3, 2))
    net = hk.transform(layer_fn)
    params = net.init(rng, x, a)
    print(params)
    print(net.apply(params, rng, x, a))
    print(hk.experimental.tabulate(net)(x, a))

    # config = {
    #     'units_list': [64,64], 
    #     'w_init': 'orthogonal', 
    #     'activation': 'relu', 
    #     'norm': None, 
    #     'index': 'all', 
    #     'index_config': {
    #         'use_shared_bias': False, 
    #         'use_bias': True, 
    #         'w_init': 'orthogonal', 
    #     }
    # }
    # def net_fn(x, *args):
    #     net = Value(**config)
    #     return net(x, *args)

    # rng = jax.random.PRNGKey(42)
    # x = jax.random.normal(rng, (2, 3, 4))
    # hx = jnp.eye(3)
    # hx = jnp.tile(hx, [2, 1, 1])
    # net = hk.transform(net_fn)
    # params = net.init(rng, x, hx)
    # print(params)
    # print(net.apply(params, rng, x, hx))
    # print(hk.experimental.tabulate(net)(x, hx))

    # config = {
    #     'units_list': [2, 3], 
    #     'w_init': 'orthogonal', 
    #     'activation': 'relu', 
    #     'norm': None, 
    #     'out_scale': .01,
    #     'rescale': .1, 
    #     'out_act': 'atan', 
    #     'combine_xa': True, 
    #     'out_size': 3, 
    #     'index': 'all', 
    #     'index_config': {
    #         'use_shared_bias': False, 
    #         'use_bias': True, 
    #         'w_init': 'orthogonal', 
    #     }
    # }
    # def net_fn(x, *args):
    #     net = Reward(**config)
    #     return net(x, *args)

    # rng = jax.random.PRNGKey(42)
    # x = jax.random.normal(rng, (2, 3, 4))
    # action = jax.random.randint(rng, (2, 3), minval=0, maxval=3)
    # hx = jnp.eye(3)
    # hx = jnp.tile(hx, [2, 1, 1])
    # net = hk.transform(net_fn)
    # params = net.init(rng, x, action, hx)
    # print(params)
    # print(net.apply(params, rng, x, action, hx))
    # print(hk.experimental.tabulate(net)(x, action, hx))
