import jax.numpy as jnp
import haiku as hk

from core.typing import dict2AttrDict
from nn.func import mlp, nn_registry
from nn.transformer import Transformer
from jax_tools import jax_dist
from algo.dynamics.elements.nn import get_model_kwargs, get_model_dist, \
    get_reward_dist, DISCRETE_MODEL, CONTINUOUS_MODEL, ENSEMBLE_AXIS
from algo.dynamics_tx.elements.utils import *
""" Source this file to register Networks """


@nn_registry.register('dynamics')
class Dynamics(hk.Module):
    def __init__(
        self, 
        model_out_size, 
        in_config, 
        tx_config, 
        model_out_config, 
        model_config={}, 
        reward_config={}, 
        discount_config={}, 
        name='dynamics', 
    ):
        super().__init__(name=name)
        self.model_out_size = model_out_size
        self.model_out_config = model_out_config
        self.in_config = dict2AttrDict(in_config, to_copy=True)
        self.tx_config = dict2AttrDict(tx_config, to_copy=True)
        self.model_config = dict2AttrDict(model_config, to_copy=True)
        self.reward_config = dict2AttrDict(reward_config, to_copy=True)
        self.discount_config = dict2AttrDict(discount_config, to_copy=True)
        self.model_out_type = self.model_out_config.model_type
        assert self.model_out_type in (DISCRETE_MODEL, CONTINUOUS_MODEL)

    def __call__(self, x, action, training=False):
        in_net, tx_net, ml, rl, dl = self.build_net()
        x = combine_sa(x, action)
        model_out, reward_out, disc_out = self.call_net(
            x, in_net, tx_net, ml, rl, dl, training=training)
        model_dist = get_model_dist(
            model_out, self.model_out_type, self.model_out_config)
        reward_dist = get_reward_dist(reward_out)
        disc_dist = jax_dist.Bernoulli(disc_out)

        return DynamicsOutput(model_dist, reward_dist, disc_dist)

    @hk.transparent
    def build_net(self, name='dynamics'):
        model_config = get_model_kwargs(
            self.model_config, 
            self.model_out_config, 
            self.model_out_size
        )
        in_net = mlp(**self.in_config, name=f'{name}_in')
        tx_net = Transformer(
            **self.tx_config, name=f'{name}_tx'
        )
        model_layer = mlp(
            **model_config, 
            name='model'
        )
        reward_layer = mlp(
            **self.reward_config, 
            out_size=1, 
            name='reward'
        )
        discount_layer = mlp(
            **self.discount_config, 
            out_size=1, 
            name='discount'
        )

        return in_net, tx_net, model_layer, reward_layer, discount_layer

    @hk.transparent
    def call_net(self, x, in_net, tx_net, ml, rl, dl, training=False):
        x = in_net(x)
        x = tx_net(x, training=training)
        model_out = ml(x)
        model_out = jnp.swapaxes(model_out, -3, -2)
        reward_out = rl(x)
        disc_out = jnp.squeeze(dl(x), -1)
        return model_out, reward_out, disc_out


@nn_registry.register('edynamics')
class EnsembleDynamics(Dynamics):
    def __init__(
        self, 
        n_models, 
        model_out_size, 
        in_config, 
        tx_config, 
        model_out_config, 
        model_config={}, 
        reward_config={}, 
        discount_config={}, 
        name='edynamics', 
    ):
        self.n_models = n_models
        super().__init__(
            model_out_size, 
            in_config=in_config, 
            tx_config=tx_config, 
            model_out_config=model_out_config, 
            model_config=model_config, 
            reward_config=reward_config, 
            discount_config=discount_config, 
            name=name
        )

    @hk.transparent
    def build_net(self, name='dynamics'):
        in_nets = [mlp(**self.in_config, name=f'{name}_in{i}') 
            for i in range(self.n_models)]
        tx_nets = [Transformer(
            **self.tx_config, name=f'{name}_tx{i}'
        ) for i in range(self.n_models)]
        model_config = get_model_kwargs(
            self.model_config, 
            self.model_out_config, 
            self.model_out_size
        )
        model_layers = [mlp(
            **model_config, 
            name=f'model{i}'
        ) for i in range(self.n_models)]
        reward_layers = [mlp(
            **self.reward_config, 
            out_size=1, 
            name=f'reward{i}'
        ) for i in range(self.n_models)]
        discount_layers = [mlp(
            **self.discount_config, 
            out_size=1, 
            name=f'discount{i}'
        ) for i in range(self.n_models)]

        return in_nets, tx_nets, model_layers, reward_layers, discount_layers

    def call_net(self, x, in_nets, tx_nets, mls, rls, dls, training=True):
        xs = [net(x) for net in in_nets]
        xs = [net(x, training=training) for net, x in zip(tx_nets, xs)]
        model_out = jnp.stack([ml(x) for ml, x in zip(mls, xs)], ENSEMBLE_AXIS)
        model_out = jnp.swapaxes(model_out, -3, -2)
        reward_out = jnp.stack([rl(x) for rl, x in zip(rls, xs)], ENSEMBLE_AXIS)
        disc_out = jnp.squeeze(
            jnp.stack([dl(x) for dl, x in zip(dls, xs)], ENSEMBLE_AXIS), -1)

        return model_out, reward_out, disc_out


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
        layer = EnsembleDynamics(5, 3, **config)
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
