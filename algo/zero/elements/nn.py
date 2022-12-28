from jax import lax, nn
import jax.numpy as jnp
import haiku as hk

from core.typing import dict2AttrDict
from nn.func import nn_registry
from nn.mlp import MLP
from nn.utils import get_activation
from jax_tools import jax_assert
""" Source this file to register Networks """


@nn_registry.register('policy')
class Policy(hk.Module):
    def __init__(
        self, 
        is_action_discrete, 
        action_dim, 
        out_act=None, 
        init_std=1, 
        name='policy', 
        **config
    ):
        super().__init__(name=name)
        self.config = dict2AttrDict(config, to_copy=True)
        self.action_dim = action_dim
        self.is_action_discrete = is_action_discrete
        self.out_act = out_act
        self.init_std = init_std

    def __call__(self, x, reset=None, state=None, action_mask=None):
        net = self.build_net()
        x = net(x, reset, state)
        if isinstance(x, tuple):
            assert len(x) == 2, x
            x, state = x
        
        if self.is_action_discrete:
            if action_mask is not None:
                jax_assert.assert_shape_compatibility([x, action_mask])
                x = jnp.where(action_mask, x, -1e10)
            return x, state
        else:
            if self.out_act == 'tanh':
                x = jnp.tanh(x)
            logstd_init = hk.initializers.Constant(lax.log(self.init_std))
            logstd = hk.get_parameter(
                'logstd', 
                shape=(self.action_dim,), 
                init=logstd_init
            )
            scale = lax.exp(logstd)
            return (x, scale), state

    @hk.transparent
    def build_net(self):
        net = MLP(
            **self.config, 
            out_size=self.action_dim, 
        )

        return net


@nn_registry.register('value')
class Value(hk.Module):
    def __init__(
        self, 
        out_act=None, 
        out_size=1, 
        name='value', 
        **config
    ):
        super().__init__(name=name)
        self.config = dict2AttrDict(config, to_copy=True)
        self.out_act = get_activation(out_act)
        self.out_size = out_size

    def __call__(self, x, reset=None, state=None):
        net = self.build_net()
        x = net(x, reset, state)
        if isinstance(x, tuple):
            assert len(x) == 2, x
            x, state = x
        if x.shape[-1] == 1:
            x = jnp.squeeze(x, -1)
        value = self.out_act(x)

        return value, state

    @hk.transparent
    def build_net(self):
        net = MLP(
            **self.config, 
            out_size=self.out_size, 
        )

        return net


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
    os.environ["XLA_FLAGS"] = '--xla_dump_to=/tmp/foo'
    os.environ['XLA_FLAGS'] = "--xla_gpu_force_compilation_parallelism=1"

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
