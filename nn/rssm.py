import collections
import logging
import jax
from jax import random
import jax.numpy as jnp
import haiku as hk
from tensorflow_probability import distributions as tfd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath('__file__'))))

from core.log import do_logging
from jax_tools.jax_utils import static_scan
from jax_tools.jax_dist import MultivariateNormalDiag
from nn.layers import Layer
from nn.registry import layer_registry, nn_registry

logger = logging.getLogger(__name__)

RSSMState = collections.namedtuple('RSSMState', ('mean', 'std', 'stoch', 'deter'))

def _prepare_for_rnn(x):
    shape = x.shape
    # we assert x.shape oughts to be [B, U, *]
    x = jnp.expand_dims(x, 0)
    x = jnp.reshape(x, (x.shape[0], -1, *x.shape[3:]))
    return x, shape

def _recover_shape(x, shape):
    x = jnp.squeeze(x, 0)
    x = jnp.reshape(x, (*shape[:2], x.shape[-1]))
    return x

def _rnn_reshape(rnn_out, shape, rnn_type):
    if rnn_type == "gru":
        return rnn_out.reshape(shape)
    elif rnn_type == "lstm":
        rnn_out.hidden = rnn_out.hidden.reshape(shape)
        rnn_out.cell = rnn_out.cell.reshape(shape)
        return rnn_out
    else:
        assert 0, rnn_type

"""
RSSM这个class其实就是以神经网络参数的形式封装了RSSM方法中的三个函数（参数化）
问题是，之前的方法当中，这些东西都可以实例化（表达成模块）
jax当中模块的内容必须表达成函数
    - 
"""

@nn_registry.register('rssm_model')
class RSSMModel(hk.Module):
    def __init__(self):
        pass

@nn_registry.register('rssm')
class RSSM(hk.Module):
    def __init__(
        self,
        name='rssm',
        stoch=30,
        deter=200,
        hidden=200,
        discrete=False,
        act='elu',
        norm='none',
        std_act='softplus',
        min_std=0.1,
        embed_units_list=[],
        trans_units_list=[],
        repre_units_list=[],
        layer_type='linear',
        activation=None,
        w_init='glorot_uniform',
        b_init='zeros',
        norm_after_activation=False,
        norm_kwargs={
            'axis': -1,
            'create_scale': True,
            'create_offset': True,
        },
        rnn_type=None,
        rnn_units=None,
        rng=None,
        **kwargs
    ):
        super().__init__(name=name)
        
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        assert not self._discrete
        self.embed_units_list = embed_units_list
        self.trans_units_list = trans_units_list
        self.repre_units_list = repre_units_list
        assert self.trans_units_list[-1] == 2 * self._stoch
        assert self.repre_units_list[-1] == 2 * self._stoch
        self.layer_kwargs = dict(
            layer_type=layer_type,
            norm=norm,
            activation=activation,
            w_init=w_init,
            norm_after_activation=norm_after_activation,
            norm_kwargs=norm_kwargs,
            **kwargs
        )

        self.rnn_type = rnn_type
        self.rnn_units = rnn_units
        self.net_rng = rng

    def initial_rssm_state(self, batch_size, n_units):
        _, core, _ = self.build_recurrent_net()
        # TODO: 1 here denotes the dimensione of units
        mean = jnp.zeros([batch_size, n_units, self._stoch])
        std = jnp.zeros([batch_size, n_units, self._stoch])
        stoch = jnp.zeros([batch_size, n_units, self._stoch])
        deter = _rnn_reshape(core.initial_state(batch_size * n_units), (batch_size, n_units, -1), self.rnn_type)
        return RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)

    def observe(self, embed, action, state=None):
        """
            TODO: add introduction str
        """
        if state is None:
            state = self.initial_rssm_state(action.shape[0], action.shape[2])
        # assume action.shape and embed.shape is [B, T, U, *]
        embed = jnp.swapaxes(embed, 0, 1)
        action = jnp.swapaxes(action, 0, 1)
        post, prior = static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs),
            (action, embed), (state, state)
        )
        post = {_key: jnp.swapaxes(post[_key], 0, 1) for _key in post}
        prior = {_key: jnp.swapaxes(prior[_key], 0, 1) for _key in prior}
        return RSSMState(
            mean=post["mean"], std=post["std"], stoch=post["stoch"], deter=post["deter"]
        ), RSSMState(
            mean=prior["mean"], std=prior["std"], stoch=prior["stoch"], deter=prior["deter"]
        )

    def imagine(self, action, state=None):
        """
            TODO: introduce the function of this function
        """
        if state is None:
            state = self.initial_rssm_state(action.shape[0], action.shape[2])
        # assume that action.shape is [B, T, U, *]
        action = jnp.swapaxes(action, 0, 1)
        # the shape of each element is [T, B, U, *]
        prior = static_scan(self.img_step, action, state)
        for _key in prior:
            prior[_key] = jnp.swapaxes(prior[_key], 0, 1)
        return RSSMState(mean=prior["mean"], std=prior["std"], stoch=prior["stoch"], deter=prior["deter"])

    @hk.transparent
    def build_recurrent_net(self):
        embed_layers = []
        for u in self.embed_units_list:
            embed_layers.append(Layer(u, **self.layer_kwargs))
        
        if self.rnn_type == 'lstm':
            core = hk.LSTM(self.rnn_units)
        elif self.rnn_type == 'gru':
            core = hk.GRU(self.rnn_units)
        core = hk.ResetCore(core)

        trans_pred_layers = []
        for u in self.trans_units_list:
            trans_pred_layers.append(Layer(u, **self.layer_kwargs))
        return embed_layers, core, trans_pred_layers
    
    @hk.transparent
    def build_posteriori_net(self):
        post_layers = []
        for u in self.repre_units_list:
            post_layers.append(Layer(u, **self.layer_kwargs))
        return post_layers

    def obs_step(self, prev_state, prev_action, embed, is_training=True):
        prior = self.img_step(prev_state, prev_action, is_training=is_training)
        x = jnp.concatenate([prior.deter, embed], -1)
        post_layers = self.build_posteriori_net()
        for layer in post_layers:
            x = layer(x, is_training)
        post = self._compute_rssm_state(x, prior.deter)
        return post, prior

    def img_step(self, prev_state, prev_action, is_training=True):
        # process data
        prev_stoch = prev_state.stoch   # [B, U, *]
        x = jnp.concatenate([prev_stoch, prev_action], -1)
        
        embed_layers, rnn_cell, trans_pred_layers = self.build_recurrent_net()
        for layer in embed_layers:
            x = layer(x, is_training)

        if prev_state.deter is None:
            prev_deter_state = rnn_cell.initial_state(x.shape[0] * x.shape[1])
        else:
            prev_deter_state = prev_state.deter

        # Conduct rnn process
        # We assert x.shape is [B, U, *]
        # Transform the shape
        x, shape = _prepare_for_rnn(x)
        prev_deter_state = _rnn_reshape(prev_deter_state, (shape[0]*shape[1], -1), self.rnn_type)
        x, deter = hk.dynamic_unroll(rnn_cell, x, prev_deter_state) 
        # Recover the shape
        x = _recover_shape(x, shape)
        deter = _rnn_reshape(deter, (shape[0], shape[1], -1), self.rnn_type)

        for layer in trans_pred_layers:
            x = layer(x)
        return self._compute_rssm_state(x, deter)

    def get_feat(self, state):
        if self.rnn_type == "gru":
            return jnp.concatenate([state.stoch, state.deter], -1)
        elif self.rnn_type == "lstm":
            return jnp.concatenate([state.stoch, state.deter.hidden], -1)
        else:
            assert 0

    def get_dist(self, mean, std):
        return MultivariateNormalDiag(mean, std)

    def _compute_rssm_state(self, x, deter):
        mean, std = jnp.split(x, 2, -1)
        std = jax.nn.softplus(std) + .1
        # here stoch gradient stop ##
        self.net_rng, rng = random.split(self.net_rng, 2)
        stoch, _ = self.get_dist(mean, std).sample_and_log_prob(seed=rng)
        state = RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)
        return state


if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)
    config = dict(
        stoch=4,
        deter=4,
        hidden=200,
        discrete=False,
        act='elu',
        std_act='softplus',
        min_std=0.1,
        embed_units_list=[2, 2],
        trans_units_list=[2, 8],
        repre_units_list=[2, 8],
        w_init='orthogonal',
        scale=1,
        activation='relu',
        norm='layer',
        rnn_type='gru',
        rnn_units=2,
        rng=rng
    )
    
    def rssm_img(prev_state, prev_action):
        rssm = RSSM(**config)
        return rssm.img_step(prev_state, prev_action)

    def rssm_obs(prev_state, prev_action, embed):
        rssm = RSSM(**config)
        return rssm.obs_step(prev_state, prev_action, embed)

    def img(action):
        rssm = RSSM(**config)
        return rssm.imagine(action)
    
    def obs(embed, action):
        rssm = RSSM(**config)
        return rssm.observe(embed, action)
    
    def feat(state):
        rssm = RSSM(**config)
        return rssm.get_feat(state)

    import jax.numpy as jnp
    rng = jax.random.PRNGKey(42)
    b = 2
    s = 8
    d = 4
    ad = 6
    # x = jnp.ones((b, s, d))

    '''
        note: 要求hidden state h.dim == x.dim
    '''

    fake_state = RSSMState(
        mean=jnp.ones((b, 1, d)),
        std=jnp.ones((b, 1, d)),
        stoch=jnp.ones((b, 1, d)),
        deter=None,
    )
    # img_step test
    prev_action = jnp.ones((b, 1, ad))
    img_net = hk.transform(rssm_img)
    params = img_net.init(rng, fake_state, prev_action)
    out_state = img_net.apply(params, rng, fake_state, prev_action)
    
    # obs_step test
    embed = jnp.ones((b, 1, d))
    obs_net = hk.transform(rssm_obs)
    params = obs_net.init(rng, fake_state, prev_action, embed)
    post, prior = obs_net.apply(params, rng, fake_state, prev_action, embed)

    # imagine test
    action = jnp.ones((b, s, 1, ad))
    net = hk.transform(img)
    params = net.init(rng, action)
    prior = net.apply(params, rng, action)
    print(prior.mean.shape)
    print(prior.stoch.shape)

    # observe test
    action = jnp.ones((b, s, 1, ad))
    embed = jnp.ones((b, s, 1, d))
    net = hk.transform(obs)
    params = net.init(rng, embed, action)
    post, prior = net.apply(params, rng, embed, action)
    print(post.mean.shape)

    # get feat test
    feat = hk.transform(feat)
    params = feat.init(rng, post)
    print(feat.apply(params, rng, post).shape)

    print("Test Pass.")