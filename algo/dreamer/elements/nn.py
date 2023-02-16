from jax import lax, nn
import jax.numpy as jnp
import haiku as hk
import jax

from core.typing import dict2AttrDict
from nn.func import mlp, nn_registry
from nn.rssm import RSSM, EmbedLayer, TransLayer, RepreLayer, RSSMRNNLayer
from nn.mlp import MLP
from nn.utils import get_activation
from jax_tools import jax_dist, jax_assert 
from algo.dreamer.elements.utils import *
""" Source this file to register Networks """


class RSSMModel:
    def __init__(
        self,
        embed_layer,
        rssm_rnn_layer,
        trans_layer,
        repre_layer,
        stoch=32,
        deter=32,
    ):
        self.rssm = RSSM(
            embed_layer, rssm_rnn_layer, trans_layer, repre_layer, stoch, deter)

@nn_registry.register('stateencoder')
class StateEncoder(hk.Module):
    def __init__(
        self,
        out_size=16,
        name='stateencoder',
        **config
    ):
        super().__init__(name=name)

        self.out_size = out_size
        self.config = dict2AttrDict(config, to_copy=True)
    
    def __call__(self, x):
        net = self.build_net()
        x = net(x)
        
        return x
    
    @hk.transparent
    def build_net(self):
        # TODO: whether to deepen the network
        net = mlp(**self.config, out_size=self.out_size)
        
        return net

@nn_registry.register('obsencoder')
class ObsEncoder(hk.Module):
    def __init__(
        self,
        out_size=16,
        name='obsencoder',
        **config
    ):
        super().__init__(name=name)

        self.out_size = out_size
        self.config = dict2AttrDict(config, to_copy=True)
    
    def __call__(self, x):
        net = self.build_net()
        x = net(x)
        
        return x
    
    @hk.transparent
    def build_net(self):
        # TODO: whether to deepen the network
        net = mlp(**self.config, out_size=self.out_size)
        
        return net

@nn_registry.register('decoder')
class Decoder(hk.Module):
    def __init__(
        self,
        out_size=None,
        name='decoder',
        **config
    ):
        super().__init__(name=name)

        self.out_size = out_size
        self.config = dict2AttrDict(config, to_copy=True)
    
    def __call__(self, x):
        net = self.build_net()
        x = net(x)

        return x

    @hk.transparent
    def build_net(self):
        net = mlp(**self.config, out_size=self.out_size)

        return net

@nn_registry.register('reward')
class Reward(hk.Module):
    def __init__(
        self,
        out_size=1,
        name='reward',
        **config
    ):
        super().__init__(name=name)
        
        self.out_size = out_size
        self.config = dict2AttrDict(config, to_copy=True)

    def __call__(self, x, action):
        net = self.build_net()
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

    def __call__(self, x):
        x = x.astype(jnp.float32)
        net = self.build_net()
        x = net(x)
        x = jnp.squeeze(x, -1)
        dist = jax_dist.Bernoulli(x)

        return dist

    @hk.transparent
    def build_net(self):
        net = mlp(**self.config, out_size=self.out_size)

        return net

@nn_registry.register('policy')
class Policy(hk.Module):
    def __init__(
        self, 
        is_action_discrete, 
        action_dim, 
        out_act=None, 
        init_std=1., 
        sigmoid_scale=False, 
        std_x_coef=1., 
        std_y_coef=.5, 
        use_feature_norm=False, 
        name='policy', 
        **config
    ):
        super().__init__(name=name)
        self.config = dict2AttrDict(config, to_copy=True)
        self.action_dim = action_dim
        self.is_action_discrete = is_action_discrete

        self.out_act = out_act
        self.init_std = init_std
        self.sigmoid_scale = sigmoid_scale
        self.std_x_coef = std_x_coef
        self.std_y_coef = std_y_coef
        self.use_feature_norm = use_feature_norm

    def __call__(self, x, reset=None, state=None, action_mask=None):
        if self.use_feature_norm:
            ln = hk.LayerNorm(-1, True, True)
            x = ln(x)
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
            if self.sigmoid_scale:
                logstd_init = self.std_x_coef
            else:
                logstd_init = lax.log(self.init_std)
            logstd_init = hk.initializers.Constant(logstd_init)
            logstd = hk.get_parameter(
                'logstd', 
                shape=(self.action_dim,), 
                init=logstd_init
            )
            if self.sigmoid_scale:
                scale = nn.sigmoid(logstd / self.std_x_coef) * self.std_y_coef
            else:
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
        use_feature_norm=False, 
        **config
    ):
        super().__init__(name=name)
        self.config = dict2AttrDict(config, to_copy=True)
        self.out_act = get_activation(out_act)
        self.out_size = out_size
        self.use_feature_norm = use_feature_norm

    def __call__(self, x, reset=None, state=None):
        if self.use_feature_norm:
            ln = hk.LayerNorm(-1, True, True)
            x = ln(x)
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