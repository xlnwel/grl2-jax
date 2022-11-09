import numpy as np
from jax import lax, nn
import jax.numpy as jnp
import haiku as hk
from core.typing import AttrDict, dict2AttrDict

from nn.func import mlp, nn_registry
from nn.index import IndexModule
from nn.utils import get_activation
from jax_tools import jax_assert
""" Source this file to register Networks """


@nn_registry.register('hpembed')
class HyperParamEmbed(hk.Module):
    def __init__(self, name='hp_embed', **config):
        super().__init__(name=name)
        self.config = config.copy()

    def __call__(self, x, *args):
        layers = self.build_net()
        hp = jnp.expand_dims(jnp.array(args), 0)
        embed = layers(hp)
        embed = jnp.broadcast_to(embed, [*x.shape[:-1], embed.shape[-1]])
        x = jnp.concatenate([x, embed], -1)

        return x
    
    @hk.transparent
    def build_net(self):
        layers = mlp(
            **self.config, 
            use_bias=False, 
        )
        return layers


@nn_registry.register('policy')
class Policy(IndexModule):
    def __init__(
        self, 
        is_action_discrete, 
        action_dim, 
        out_act=None, 
        init_std=1, 
        out_scale=.01, 
        name='policy', 
        **config
    ):
        self.action_dim = action_dim
        self.is_action_discrete = is_action_discrete
        self.out_act = out_act
        self.init_std = init_std

        config['out_scale'] = out_scale
        assert config['index_config'].get('scale', 1) == 1, config
        super().__init__(config=config, out_size=self.action_dim , name=name)

    def __call__(self, x, hx=None, action_mask=None):
        x = super().__call__(x, hx)

        if self.is_action_discrete:
            if action_mask is not None:
                jax_assert.assert_shape_compatibility([x, action_mask])
                x = jnp.where(action_mask, x, -1e10)
            return x
        else:
            if self.out_act == 'tanh':
                x = jnp.tanh(x)
            logstd_init = hk.initializers.Constant(lax.log(self.init_std))
            logstd = hk.get_parameter(
                'logstd', 
                shape=(self.action_dim,), 
                init=logstd_init
            )
            return x, logstd


@nn_registry.register('value')
class Value(IndexModule):
    def __init__(
        self, 
        out_act=None, 
        out_size=1, 
        name='value', 
        **config
    ):
        self.out_act = get_activation(out_act)
        super().__init__(config=config, out_size=out_size, name=name)

    def __call__(self, x, hx=None):
        value = super().__call__(x, hx)

        if value.shape[-1] == 1:
            value = jnp.squeeze(value, -1)
        value = self.out_act(value)
        return value



@nn_registry.register('reward')
class Reward(IndexModule):
    def __init__(
        self, 
        action_dim, 
        combine_xa=False, 
        reward_type='scalar', 
        reward_rescale=5, 
        out_act='tanh', 
        reward_size=11, 
        reward_range=5, 
        sparse_reward=True, 
        name='reward', 
        **config
    ):
        config = config.copy()
        
        self.action_dim = action_dim
        self.out_act = get_activation(out_act)
        self.combine_xa = combine_xa
        self.reward_type = reward_type
        self.reward_rescale = reward_rescale
        self.sparse_reward = sparse_reward
        reward_size = 1 if reward_type == 'scalar' else reward_size
        self.reward_size = reward_size if combine_xa else reward_size * action_dim
        reward_range = reward_range
        self.reward_range = np.linspace(-reward_range, reward_range, reward_size)
        assert config['index_config'].get('scale', 1) == 1, config['index_config']
        super().__init__(config=config, out_size=self.reward_size + 1, name=name)

    def __call__(self, x, action, hx=None):
        if len(action.shape) < len(x.shape):
            action = nn.one_hot(action, self.action_dim)
        if self.combine_xa:
            x = jnp.concatenate([x, action], -1)
        
        x = super().__call__(x, hx)
        
        stats = AttrDict()
        ind, x = jnp.split(x, [1], axis=-1)
        ind = ind.squeeze(-1)
        
        if self.reward_type == 'scalar':
            if action is not None and not self.combine_xa:
                x = jnp.sum(x * action, -1)
            stats.meta_reward_out = x
            reward = self.reward_rescale * self.out_act(x / self.reward_rescale)
        elif self.reward_type == 'categorical':
            if action is not None and not self.combine_xa:
                x = jnp.reshape(x, (*action.shape, len(self.reward_range)))
                action = jnp.expand_dims(action, -1)
                x = jnp.sum(x * action, -2)
            stats.meta_reward_out = x
            x = nn.softmax(x)
            stats.meta_reward_dist = x
            reward = jnp.sum(x * self.reward_range, -1)
        else:
            raise ValueError(f'Unknown reward type {self.reward_type}')
        stats.raw_meta_reward = reward
        
        if self.sparse_reward:
            ind = lax.stop_gradient(jnp.where(ind > 0, 1., 0.) - ind) + ind
            jax_assert.assert_shape_compatibility([reward, ind])
            meta_reward = reward * ind
        else:
            meta_reward = reward
        stats.meta_indicator = ind
        stats.meta_reward = meta_reward

        return meta_reward, stats


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

    config = {
        'units_list': [3], 
        'w_init': 'orthogonal', 
        'activation': 'relu', 
        'norm': None, 
        'out_scale': .01,
        'is_action_discrete': True,  
        'action_dim': 3, 
        'index': 'all', 
        'index_config': {
            'use_shared_bias': False, 
            'use_bias': True, 
            'w_init': 'orthogonal', 
        }
    }
    def layer_fn(x, *args):
        layer = Policy(**config)
        return layer(x, *args)
    import jax
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (2, 3, 4))
    hx = jnp.eye(3)
    hx = jnp.tile(hx, [2, 1, 1])
    net = hk.transform(layer_fn)
    params = net.init(rng, x, hx)
    print(params)
    print(net.apply(params, rng, x, hx))
    print(hk.experimental.tabulate(net)(x, hx))

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
