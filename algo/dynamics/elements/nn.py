from jax import lax
import jax.numpy as jnp
import haiku as hk

from core.typing import dict2AttrDict
from nn.func import mlp, nn_registry
from jax_tools import jax_dist
from algo.dynamics.elements.utils import *
""" Source this file to register Networks """


ENSEMBLE_AXIS = 0

def get_normal_dist(loc, logvar, max_logvar, min_logvar):
    logvar = bound_logvar(logvar, max_logvar, min_logvar)
    scale = lax.exp(logvar / 2)
    dist = jax_dist.MultivariateNormalDiag(loc, scale)
    return dist


def get_discrete_dist(x):
    dist = jax_dist.Categorical(x)
    return dist


def get_model_dist(x, out_type, out_config):
    if out_type == DISCRETE_MODEL:
        dist = get_discrete_dist(x)
    else:
        dist = get_normal_dist(x.take(0, axis=-2), x.take(1, axis=-2), 
            out_config.max_logvar, out_config.min_logvar)
    
    return dist


def get_reward_dist(x):
    if x.shape[-1] == 1:
        x = jnp.squeeze(x, -1)
        dist = jax_dist.MultivariateNormalDiag(x, jnp.ones_like(x))
    else:
        dist = jax_dist.Categorical(x)
    return dist

CONTINUOUS_MODEL = 'continuous'
DISCRETE_MODEL = 'discrete'


def get_model_kwargs(model_config, out_config, out_size):
    model_config = model_config.copy()
    model_config.out_layer_type = 'elinear'
    kwargs = AttrDict()
    if out_config.model_type == DISCRETE_MODEL:
        kwargs.ensemble_size = out_size
        kwargs.expand_edim = True
    else:
        kwargs.ensemble_size = 2
        kwargs.expand_edim = True
    model_config.out_kwargs = kwargs
    model_config.out_size = out_config.n_classes

    return model_config


@nn_registry.register('dynamics')
class Dynamics(hk.Module):
    def __init__(
        self, 
        model_out_size, 
        repr_config, 
        model_out_config, 
        model_config={}, 
        reward_config={}, 
        discount_config={}, 
        name='dynamics', 
    ):
        super().__init__(name=name)
        self.model_out_size = model_out_size
        self.model_out_config = model_out_config
        self.repr_config = dict2AttrDict(repr_config, to_copy=True)
        self.model_config = dict2AttrDict(model_config, to_copy=True)
        self.reward_config = dict2AttrDict(reward_config, to_copy=True)
        self.discount_config = dict2AttrDict(discount_config, to_copy=True)
        self.model_out_type = self.model_out_config.model_type
        assert self.model_out_type in (DISCRETE_MODEL, CONTINUOUS_MODEL)

    def __call__(self, x, action, training=False):
        net, ml, rl, dl = self.build_net()
        x = combine_sa(x, action)
        model_out, reward_out, disc_out = self.call_net(x, net, ml, rl, dl)
        model_dist = get_model_dist(
            model_out, self.model_out_type, self.model_out_config)
        reward_dist = get_reward_dist(reward_out)
        disc_dist = jax_dist.Bernoulli(disc_out)

        return DynamicsOutput(model_dist, reward_dist, disc_dist)

    @hk.transparent
    def build_net(self):
        net = mlp(**self.repr_config, name='repr')
        model_config = get_model_kwargs(
            self.model_config, 
            self.model_out_config, 
            self.model_out_size
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

        return net, model_layer, reward_layer, discount_layer

    @hk.transparent
    def call_net(self, x, net, ml, rl, dl):
        x = net(x)
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
        repr_config, 
        model_out_config, 
        model_config={}, 
        reward_config={}, 
        discount_config={}, 
        name='edynamics', 
    ):
        self.n_models = n_models
        super().__init__(
            model_out_size, 
            repr_config=repr_config, 
            model_out_config=model_out_config, 
            model_config=model_config, 
            reward_config=reward_config, 
            discount_config=discount_config, 
            name=name, 
        )

    @hk.transparent
    def build_net(self):
        nets = [
            mlp(**self.repr_config, name=f'repr{i}') for i in range(self.n_models)
        ]
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

        return nets, model_layers, reward_layers, discount_layers

    @hk.transparent
    def call_net(self, x, nets, mls, rls, dls):
        xs = [net(x) for net in nets]
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
