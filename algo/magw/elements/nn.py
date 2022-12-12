from jax import lax, nn
import jax.numpy as jnp
import haiku as hk

from core.typing import dict2AttrDict
from nn.func import mlp, nn_registry
from jax_tools import jax_dist
""" Source this file to register Networks """


def combine_sa(x, a, one_hot, action_dim):
    a2 = a[..., ::-1, :]
    a = jnp.stack([a, a2], -2)
    if one_hot:
        a = nn.one_hot(a, action_dim)
        a = jnp.reshape(a, (*a.shape[:-2], -1))
    x = jnp.concatenate([x, a], -1)

    return x


@nn_registry.register('model')
class Model(hk.Module):
    def __init__(
        self, 
        one_hot, 
        action_dim, 
        out_size, 
        name='model', 
        **config, 
    ):
        super().__init__(name=name)
        self.config = dict2AttrDict(config, to_copy=True)
        self.one_hot = one_hot
        self.n_grids = 5
        self.action_dim = action_dim
        self.out_size = out_size

    def __call__(self, x, action):
        net = self.build_net()

        x = combine_sa(x, action, self.one_hot, self.action_dim)

        x = net(x)
        logits = jnp.reshape(x, (*x.shape[:-1], self.out_size, self.n_grids))
        dist = jax_dist.Categorical(logits)

        return dist

    @hk.transparent
    def build_net(self):
        net = mlp(
            **self.config, 
            out_size=self.n_grids * self.out_size, 
        )
        return net


@nn_registry.register('emodels')
class EnsembleModels(hk.Module):
    def __init__(
        self, 
        n, 
        one_hot, 
        action_dim, 
        out_size, 
        name='emodels', 
        **config
    ):
        super().__init__(name=name)
        self.config = dict2AttrDict(config, to_copy=True)
        self.n = n
        self.one_hot = one_hot
        self.n_grids = 5
        self.action_dim = action_dim
        self.out_size = out_size

    def __call__(self, x, action):
        nets = self.build_net()

        x = combine_sa(x, action, self.one_hot, self.action_dim)

        x = jnp.stack([net(x) for net in nets], -2)
        logits = jnp.reshape(x, (*x.shape[:-1], self.out_size, self.n_grids))
        dist = jax_dist.Categorical(logits)

        return dist

    @hk.transparent
    def build_net(self):
        nets = [mlp(
            **self.config,
            out_size=self.n_grids * self.out_size, 
            name=f'model{i}'
        ) for i in range(self.n)]
        return nets


@nn_registry.register('reward')
class Reward(hk.Module):
    def __init__(
        self, 
        one_hot, 
        action_dim, 
        out_size=1, 
        name='reward', 
        **config
    ):
        super().__init__(name=name)

        self.one_hot = one_hot
        self.action_dim = action_dim
        self.out_size = out_size
        self.config = dict2AttrDict(config, to_copy=True)

    def __call__(self, x, action):
        net = self.build_net()

        x = combine_sa(x, action, self.one_hot, self.action_dim)
        
        x = net(x)
        if self.out_size == 1:
            dist = jax_dist.MultivariateNormalDiag(x, 0.)
        else:
            dist = jax_dist.Categorical(x)

        return dist

    @hk.transparent
    def build_net(self):
        net = mlp(**self.config, out_size=self.out_size)

        return net


def compute_mean_logvar(x, max_logvar, min_logvar):
    mean, logvar = jnp.split(x, -1)
    logvar = max_logvar - nn.softplus(max_logvar - logvar)
    logvar = min_logvar + nn.softplus(logvar - min_logvar)

    return mean, logvar


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
