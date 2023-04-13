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


def get_dist(x, out_type, out_config):
    if out_type == DISCRETE_MODEL:
        dist = get_discrete_dist(x)
    else:
        dist = get_normal_dist(x.take(0, axis=-2), x.take(1, axis=-2), 
            out_config.max_logvar, out_config.min_logvar)
    
    return dist


CONTINUOUS_MODEL = 'continuous'
DISCRETE_MODEL = 'discrete'


def get_out_kwargs(out_type, out_config, out_size):
    kwargs = AttrDict(out_layer_type='elayer')
    if out_type == DISCRETE_MODEL:
        kwargs.out_size = out_config.n_classes
        kwargs.out_kwargs = {'ensemble_size': out_size, 'expand_edim': True}
    else:
        kwargs.out_size = out_size
        kwargs.out_kwargs = {'ensemble_size': 2, 'expand_edim': True}
    return kwargs


@nn_registry.register('model')
class Model(hk.Module):
    def __init__(
        self, 
        out_size, 
        out_type, 
        out_config, 
        name='model', 
        **config, 
    ):
        super().__init__(name=name)
        self.config = dict2AttrDict(config, to_copy=True)
        self.out_size = out_size

        self.out_config = dict2AttrDict(out_config, to_copy=True)
        self.out_type = out_type
        assert self.out_type in (DISCRETE_MODEL, CONTINUOUS_MODEL)

    def __call__(self, x, action, training=False):
        net = self.build_net()
        x = combine_sa(x, action)
        x = self.call_net(net, x)
        dist = get_dist(x, self.out_type, self.out_config)
        return dist

    @hk.transparent
    def build_net(self):
        out_kwargs = get_out_kwargs(
            self.out_type, self.out_config, self.out_size)
        net = mlp(
            **self.config, 
            **out_kwargs, 
            name='model_mlp', 
        )
        return net

    def call_net(self, net, x):
        x = net(x)
        x = jnp.swapaxes(x, -3, -2)
        return x
    

@nn_registry.register('emodels')
class EnsembleModels(Model):
    def __init__(
        self, 
        n_models, 
        out_size, 
        out_type, 
        out_config, 
        name='emodels', 
        **config, 
    ):
        self.n_models = n_models
        super().__init__(out_size, out_type, out_config, name=name, **config)

    @hk.transparent
    def build_net(self):
        out_kwargs = get_out_kwargs(
            self.out_type, self.out_config, self.out_size)
        nets = [mlp(
            **self.config,
            **out_kwargs, 
            name=f'model{i}_mlp', 
        ) for i in range(self.n_models)]
        return nets

    def call_net(self, nets, x):
        x = jnp.stack([net(x) for net in nets], ENSEMBLE_AXIS)
        x = jnp.swapaxes(x, -3, -2)
        return x


@nn_registry.register('reward')
class Reward(hk.Module):
    def __init__(
        self, 
        use_next_obs=False, 
        balance_dimension=False, 
        obs_embed=None, 
        action_embed=None, 
        out_size=1, 
        name='reward', 
        **config
    ):
        super().__init__(name=name)

        self.use_next_obs = use_next_obs
        self.balance_dimension = balance_dimension
        self.obs_embed = obs_embed
        self.action_embed = action_embed
        self.out_size = out_size
        self.config = dict2AttrDict(config, to_copy=True)

    def __call__(self, x, action, next_x=None):
        if self.balance_dimension:
            obs_transform, action_transform, net = self.build_net()
            obs_embed = obs_transform(x)
            actions = joint_actions(action)
            action_embed = action_transform(actions)
            if self.use_next_obs:
                next_obs_embed = obs_transform(next_x)
                x = jnp.concatenate([obs_embed, action_embed, next_obs_embed], -1)
            else:
                x = jnp.concatenate([obs_embed, action_embed], -1)
        else:
            net = self.build_net()
            action = joint_actions(action)
            if self.use_next_obs:
                x = jnp.concatenate([x, action, next_x], -1)
            else:
                x = jnp.concatenate([x, action], -1)
            x = combine_sa(x, action)
        x = net(x)
        if self.out_size == 1:
            x = jnp.squeeze(x, -1)
            dist = jax_dist.MultivariateNormalDiag(x, jnp.ones_like(x))
        else:
            dist = jax_dist.Categorical(x)

        return dist

    @hk.transparent
    def build_net(self):
        if self.balance_dimension:
            obs_transform = mlp(out_size=self.obs_embed)
            action_transform = mlp(out_size=self.action_embed)
            net = mlp(**self.config, out_size=self.out_size)
            return obs_transform, action_transform, net
        else:
            net = mlp(**self.config, out_size=self.out_size)
            return net


@nn_registry.register('discount')
class Discount(hk.Module):
    def __init__(
        self, 
        out_size=1, 
        name='discount', 
        **config
    ):
        super().__init__(name=name)

        self.out_size = out_size
        self.config = dict2AttrDict(config, to_copy=True)

    def __call__(self, x, action):
        action = joint_actions(action)
        x = jnp.concatenate([x, action], -1)
        net = self.build_net()
        x = net(x)
        x = jnp.squeeze(x, -1)
        dist = jax_dist.Bernoulli(x)

        return dist

    @hk.transparent
    def build_net(self):
        net = mlp(**self.config, out_size=self.out_size)

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
