import collections
import logging
import jax
from jax import random
import jax.numpy as jnp
import haiku as hk

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath('__file__'))))

from core.log import do_logging
from jax_tools.jax_utils import static_scan
from jax_tools.jax_dist import MultivariateNormalDiag
from nn.layers import Layer
from nn.registry import nn_registry
from jax_tools import jax_utils

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

def _rnn_reshape(rnn_out, shape):
  rnn_out = jax.tree_util.tree_map(lambda x: x.reshape(shape), rnn_out)
  return rnn_out

class RSSM:
  def __init__(
    self,
    embed_layer,
    rssm_rnn_layer,
    trans_layer,
    repre_layer,
    stoch=32,
    deter=32,
    rnn_type='gru',
    rng=None
  ):
    self._stoch = stoch
    self._deter = deter
    self.embed_layer = embed_layer
    self.rssm_rnn_layer = rssm_rnn_layer
    self.trans_layer = trans_layer
    self.repre_layer = repre_layer
    self.rnn_type = rnn_type
    self.net_rng = rng

  def initial_rssm_state(self, params, rng, batch_size, n_units):
    mean = jnp.zeros([batch_size, n_units, self._stoch])
    std = jnp.zeros([batch_size, n_units, self._stoch])
    stoch = jnp.zeros([batch_size, n_units, self._stoch])
    deter = _rnn_reshape(self.rssm_rnn_layer(params.rssm_rnn, rng, batch_size=batch_size * n_units, init=True), (batch_size, n_units, -1))
    return RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)

  def observe(self, params, rng, embed, action, reset, state=None):
    if state is None:
      state = self.initial_rssm_state(params, rng, action.shape[0], action.shape[2])
    
    # assume action.shape and embed.shape is [B, T, U, *]
    embed = jnp.swapaxes(embed, 0, 1)
    action = jnp.swapaxes(action, 0, 1) if action is not None else action
    reset = jnp.swapaxes(reset, 0, 1)
    post, prior = static_scan(
      lambda prev, inputs: self.obs_step(params, rng, prev[0], *inputs),
      (action, embed, reset), (state, state)
    )
    post = {_key: jnp.swapaxes(post[_key], 0, 1) for _key in post}
    prior = {_key: jnp.swapaxes(prior[_key], 0, 1) for _key in prior}
    return RSSMState(
      mean=post["mean"], std=post["std"], stoch=post["stoch"], deter=post["deter"]
    ), RSSMState(
      mean=prior["mean"], std=prior["std"], stoch=prior["stoch"], deter=prior["deter"]
    )

  def imagine(self, params, rng, action, reset, state=None):
    """
      TODO: introduce the function of this function
    """
    if state is None:
      state = self.initial_rssm_state(params, rng, action.shape[0], action.shape[2])
    # assume that action.shape is [B, T, U, *]
    action = jnp.swapaxes(action, 0, 1) if action is not None else action
    reset = jnp.swapaxes(reset, 0, 1)
    # the shape of each element is [T, B, U, *]
    prior = static_scan(
      lambda prev, inputs: self.img_step(params, rng, prev, *inputs),
      (action, reset), state
    )
    for _key in prior:
      prior[_key] = jnp.swapaxes(prior[_key], 0, 1)
    return RSSMState(mean=prior["mean"], std=prior["std"], stoch=prior["stoch"], deter=prior["deter"])

  def obs_step(self, params, rng, prev_state, prev_action, embed, reset, is_training=True):
    if len(embed.shape) == 4: # [B, T, U, *] -> [B, U, *]
      prev_action = prev_action[:, -1] if prev_action is not None else prev_action
      embed = embed[:, -1]
      reset = reset[:, -1]
    else:
      assert len(embed.shape) == 3
    prior = self.img_step(params, rng, prev_state, prev_action, reset=reset, is_training=is_training)
    x = jnp.concatenate([prior.deter, embed], -1)
    x = self.repre_layer(params.rssm_repre, rng, x, is_training)
    post = self._compute_rssm_state(rng, x, prior.deter)
    return post, prior

  def img_step(self, params, rng, prev_state, prev_action, reset, is_training=True):
    if len(reset.shape) == 3: # [B, T, U, *] -> [B, U, *]
      print('=-='*20)
      print(prev_action.shape)
      print(reset.shape)
      print('=-='*20)
      prev_action = prev_action[:, -1] if prev_action is not None else prev_action
      reset = reset[:, -1]

    # process data
    if prev_state is None:
      prev_state = self.initial_rssm_state(prev_action.shape[0], prev_action.shape[1])
    if prev_action is None:
      return prev_state

    prev_stoch = prev_state.stoch   # [B, U, *]
    prev_deter_state = prev_state.deter
    x = jnp.concatenate([prev_stoch, prev_action], -1)
    print("*-*"*30)
    print(prev_stoch.shape)
    print(x.shape)
    # Embed x
    x = self.embed_layer(params.rssm_embed, rng, x, is_training) 
    # Conduct rnn process
    x, deter = self.rssm_rnn_layer(params.rssm_rnn, rng, x, reset, prev_deter_state) 
    # Do trans
    x = self.trans_layer(params.rssm_trans, rng, x, is_training)
    
    return self._compute_rssm_state(rng, x, deter)

  def get_feat(self, state):
    if self.rnn_type == "gru":
      return jnp.concatenate([state.stoch, state.deter], -1)
    elif self.rnn_type == "lstm":
      return jnp.concatenate([state.stoch, state.deter.hidden], -1)
    else:
      assert 0

  def get_dist(selfd, mean, std):
    return MultivariateNormalDiag(mean, std)

  def _compute_rssm_state(self, rng, x, deter):
    mean, std = jnp.split(x, 2, -1)
    std = jax.nn.softplus(std) + .1
    # here stoch gradient stop ##
    # self.net_rng, rng = random.split(self.net_rng, 2)
    stoch, _ = self.get_dist(mean, std).sample_and_log_prob(seed=rng)
    state = RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)
    return state

@nn_registry.register('reprelayer')
class RepreLayer(hk.Module):
  def __init__(
    self,
    name='reprelayer',
    norm='none',
    repre_units_list=[],
    layer_type='linear',
    activation=None,
    w_init='glorot_uniform',
    norm_after_activation=False,
    norm_kwargs={
      'axis': -1,
      'create_scale': True,
      'create_offset': True,
    },
    rng=None,
    **kwargs,
  ):
    super().__init__(name=name)
    self.repre_units_list = repre_units_list
    self.net_rng = rng

    self.layer_kwargs = dict(
      layer_type=layer_type,
      norm=norm,
      activation=activation,
      w_init=w_init,
      norm_after_activation=norm_after_activation,
      norm_kwargs=norm_kwargs,
      **kwargs
    )
  
  def __call__(self, x, is_training=True):
    repre_layers = self.build_net()
    for layer in repre_layers:
      x = layer(x, is_training)
    return x
  
  @hk.transparent
  def build_net(self):
    repre_layers = []
    for u in self.repre_units_list:
      repre_layers.append(Layer(u, **self.layer_kwargs))
    
    return repre_layers


@nn_registry.register('embedlayer')
class EmbedLayer(hk.Module):
  def __init__(
    self,
    name='embedlayer',
    norm='none',
    embed_units_list=[],
    layer_type='linear',
    activation=None,
    w_init='glorot_uniform',
    norm_after_activation=False,
    norm_kwargs={
      'axis': -1,
      'create_scale': True,
      'create_offset': True,
    },
    rng=None,
    **kwargs,
  ):
    super().__init__(name=name)
    self.embed_units_list = embed_units_list
    self.net_rng = rng

    self.layer_kwargs = dict(
      layer_type=layer_type,
      norm=norm,
      activation=activation,
      w_init=w_init,
      norm_after_activation=norm_after_activation,
      norm_kwargs=norm_kwargs,
      **kwargs
    )

  def __call__(self, x, is_training=True):
    embed_layers = self.build_net()
    for layer in embed_layers:
      x = layer(x, is_training)
    return x

  @hk.transparent
  def build_net(self):
    embed_layers = []
    for u in self.embed_units_list:
      embed_layers.append(Layer(u, **self.layer_kwargs))

    return embed_layers

@nn_registry.register('rssmrnnlayer')
class RSSMRNNLayer(hk.Module):
  def __init__(
    self,
    name='rssmrnnlayer',
    rnn_type=None,
    rnn_units=None,
    rng=None,
    **kwargs,
  ):
    super().__init__(name=name)
    self.rnn_type = rnn_type
    self.rnn_units = rnn_units
    self.net_rng = rng

  def __call__(self, x=None, reset=None, state=None, batch_size=1, init=False):
    rnn_cell = self.build_net()

    if init:
      return rnn_cell.initial_state(batch_size)
    else:
      # We assert x.shape is [B, U, *]
      # Transform the shape
      x, shape = _prepare_for_rnn(x)
      reset, _ = _prepare_for_rnn(reset)
      x = (x, reset)
      state = _rnn_reshape(state, (shape[0]*shape[1], -1))
      # Perform RNN
      x, state = hk.dynamic_unroll(rnn_cell, x, state)
      # Recover the shape
      x = _recover_shape(x, shape)
      state = _rnn_reshape(state, (shape[0], shape[1], -1))
    
      return x, state

  @hk.transparent
  def build_net(self):
    if self.rnn_type == 'lstm':
      core = hk.LSTM(self.rnn_units)
    elif self.rnn_type == 'gru':
      core = hk.GRU(self.rnn_units)
    core = hk.ResetCore(core)
    return core  

@nn_registry.register('translayer')
class TransLayer(hk.Module):
  def __init__(
    self,
    name='translayer',
    norm='none',
    trans_units_list=[],
    layer_type='linear',
    activation=None,
    w_init='glorot_uniform',
    norm_after_activation=False,
    norm_kwargs={
      'axis': -1,
      'create_scale': True,
      'create_offset': True,
    },
    rng=None,
    **kwargs
  ):
    super().__init__(name=name)
    self.trans_units_list = trans_units_list
    self.net_rng = rng

    self.layer_kwargs = dict(
      layer_type=layer_type,
      norm=norm,
      activation=activation,
      w_init=w_init,
      norm_after_activation=norm_after_activation,
      norm_kwargs=norm_kwargs,
      **kwargs
    )

  def __call__(self, x, is_training=True):
    trans_layers = self.build_net()
    for layer in trans_layers:
      x = layer(x, is_training)
    return x

  @hk.transparent
  def build_net(self):
    trans_layers = []
    for u in self.trans_units_list:
      trans_layers.append(Layer(u, **self.layer_kwargs))

    return trans_layers 

# ========== below is the old version ==========

# @nn_registry.register('rssm')
# class RSSM(hk.Module):
#   def __init__(
#     self,
#     name='rssm',
#     stoch=32,
#     deter=32,
#     discrete=False,
#     norm='none',
#     embed_units_list=[],
#     trans_units_list=[],
#     repre_units_list=[],
#     layer_type='linear',
#     activation=None,
#     w_init='glorot_uniform',
#     norm_after_activation=False,
#     norm_kwargs={
#       'axis': -1,
#       'create_scale': True,
#       'create_offset': True,
#     },
#     rnn_type=None,
#     rng=None,
#     **kwargs
#   ):
#     super().__init__(name=name)
    
#     self._stoch = stoch
#     self._deter = deter
#     self._discrete = discrete
#     assert not self._discrete
#     self.embed_units_list = embed_units_list
#     self.trans_units_list = trans_units_list
#     self.repre_units_list = repre_units_list
#     assert self.trans_units_list[-1] == 2 * self._stoch
#     assert self.repre_units_list[-1] == 2 * self._stoch
#     self.layer_kwargs = dict(
#       layer_type=layer_type,
#       norm=norm,
#       activation=activation,
#       w_init=w_init,
#       norm_after_activation=norm_after_activation,
#       norm_kwargs=norm_kwargs,
#       **kwargs
#     )

#     self.rnn_type = rnn_type
#     self.rnn_units = self._deter
#     self.net_rng = rng

#   def initial_rssm_state(self, batch_size, n_units):
#     _, core, _ = self.build_recurrent_net()
#     # TODO: 1 here denotes the dimensione of units
#     mean = jnp.zeros([batch_size, n_units, self._stoch])
#     std = jnp.zeros([batch_size, n_units, self._stoch])
#     stoch = jnp.zeros([batch_size, n_units, self._stoch])
#     deter = _rnn_reshape(core.initial_state(batch_size * n_units), (batch_size, n_units, -1), self.rnn_type)
#     return RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)

#   def observe(self, embed, action, reset, state=None, just_step=False):
#     """
#       TODO: add introduction str
#     """
#     if state is None:
#       state = self.initial_rssm_state(action.shape[0], action.shape[2])
#     if just_step:
#       return self.obs_step(state, action, embed, reset)
#     # assume action.shape and embed.shape is [B, T, U, *]
#     embed = jnp.swapaxes(embed, 0, 1)
#     action = jnp.swapaxes(action, 0, 1) if action is not None else action
#     reset = jnp.swapaxes(reset, 0, 1)
#     post, prior = static_scan(
#       lambda prev, inputs: self.obs_step(prev[0], *inputs),
#       (action, embed, reset), (state, state)
#     )
#     post = {_key: jnp.swapaxes(post[_key], 0, 1) for _key in post}
#     prior = {_key: jnp.swapaxes(prior[_key], 0, 1) for _key in prior}
#     return RSSMState(
#       mean=post["mean"], std=post["std"], stoch=post["stoch"], deter=post["deter"]
#     ), RSSMState(
#       mean=prior["mean"], std=prior["std"], stoch=prior["stoch"], deter=prior["deter"]
#     )

#   def imagine(self, action, reset, state=None, just_step=False):
#     """
#       TODO: introduce the function of this function
#     """
#     if state is None:
#       state = self.initial_rssm_state(action.shape[0], action.shape[2])
#     if just_step:
#       return self.img_step(state, action, reset)
#     # assume that action.shape is [B, T, U, *]
#     action = jnp.swapaxes(action, 0, 1) if action is not None else action
#     # print(reset)
#     reset = jnp.swapaxes(reset, 0, 1)
#     # the shape of each element is [T, B, U, *]
#     prior = static_scan(
#       lambda prev, inputs: self.img_step(prev, *inputs),
#       (action, reset), state
#     )
#     for _key in prior:
#       prior[_key] = jnp.swapaxes(prior[_key], 0, 1)
#     return RSSMState(mean=prior["mean"], std=prior["std"], stoch=prior["stoch"], deter=prior["deter"])

#   @hk.transparent
#   def build_recurrent_net(self):
#     embed_layers = []
#     for u in self.embed_units_list:
#       embed_layers.append(Layer(u, **self.layer_kwargs))
    
#     if self.rnn_type == 'lstm':
#       core = hk.LSTM(self.rnn_units)
#     elif self.rnn_type == 'gru':
#       core = hk.GRU(self.rnn_units)
#     core = hk.ResetCore(core)

#     trans_pred_layers = []
#     for u in self.trans_units_list:
#       trans_pred_layers.append(Layer(u, **self.layer_kwargs))
#     return embed_layers, core, trans_pred_layers
  
#   @hk.transparent
#   def build_posteriori_net(self):
#     post_layers = []
#     for u in self.repre_units_list:
#       post_layers.append(Layer(u, **self.layer_kwargs))
#     return post_layers

#   def obs_step(self, prev_state, prev_action, embed, reset, is_training=True):
#     if len(embed.shape) == 4: # [B, T, U, *] -> [B, U, *]
#       prev_action = prev_action[:, -1] if prev_action is not None else prev_action
#       embed = embed[:, -1]
#       reset = reset[:, -1]
#     else:
#       assert len(embed.shape) == 3
#     prior = self.img_step(prev_state, prev_action, reset=reset, is_training=is_training)
#     x = jnp.concatenate([prior.deter, embed], -1)
#     post_layers = self.build_posteriori_net()
#     for layer in post_layers:
#       x = layer(x, is_training)
#     post = self._compute_rssm_state(x, prior.deter)
#     return post, prior

#   def img_step(self, prev_state, prev_action, reset, is_training=True):
#     if len(reset.shape) == 3: # [B, T, U, *] -> [B, U, *]
#       prev_action = prev_action[:, -1] if prev_action is not None else prev_action
#       reset = reset[:, -1]
    
#     # process data
#     if prev_state is None:
#       prev_state = self.initial_rssm_state(prev_action.shape[0], prev_action.shape[1])
#     if prev_action is None:
#       return prev_state

#     prev_stoch = prev_state.stoch   # [B, U, *]
#     prev_deter_state = prev_state.deter
#     x = jnp.concatenate([prev_stoch, prev_action], -1)
    
#     embed_layers, rnn_cell, trans_pred_layers = self.build_recurrent_net()
#     for layer in embed_layers:
#       x = layer(x, is_training)

#     # Conduct rnn process
#     # We assert x.shape is [B, U, *]
#     # Transform the shape
#     x, shape = _prepare_for_rnn(x)
#     reset, _ = _prepare_for_rnn(reset)
#     x = (x, reset)
#     prev_deter_state = _rnn_reshape(prev_deter_state, (shape[0]*shape[1], -1), self.rnn_type)
#     x, deter = hk.dynamic_unroll(rnn_cell, x, prev_deter_state)
#     # Recover the shape
#     x = _recover_shape(x, shape)
#     deter = _rnn_reshape(deter, (shape[0], shape[1], -1), self.rnn_type)

#     for layer in trans_pred_layers:
#       x = layer(x)
#     return self._compute_rssm_state(x, deter)

#   def get_feat(self, state):
#     if self.rnn_type == "gru":
#       return jnp.concatenate([state.stoch, state.deter], -1)
#     elif self.rnn_type == "lstm":
#       return jnp.concatenate([state.stoch, state.deter.hidden], -1)
#     else:
#       assert 0

#   def get_dist(self, mean, std):
#     return MultivariateNormalDiag(mean, std)

#   def _compute_rssm_state(self, x, deter):
#     mean, std = jnp.split(x, 2, -1)
#     std = jax.nn.softplus(std) + .1
#     # here stoch gradient stop ##
#     self.net_rng, rng = random.split(self.net_rng, 2)
#     stoch, _ = self.get_dist(mean, std).sample_and_log_prob(seed=rng)
#     state = RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)
#     return state


# if __name__ == "__main__":
#   rng = jax.random.PRNGKey(42)
#   config = dict(
#     stoch=4,
#     deter=4,
#     # hidden=200,
#     discrete=False,
#     # act='elu',
#     # std_act='softplus',
#     # min_std=0.1,
#     embed_units_list=[2, 3],
#     trans_units_list=[2, 8],
#     repre_units_list=[2, 8],
#     w_init='orthogonal',
#     scale=1,
#     activation='relu',
#     norm='layer',
#     rnn_type='gru',
#     rnn_units=3,
#     rng=rng
#   )
  
#   def rssm_img(prev_state, prev_action, reset):
#     rssm = RSSM(**config)
#     return rssm.img_step(prev_state, prev_action, reset)

#   def rssm_obs(prev_state, prev_action, embed, reset):
#     rssm = RSSM(**config)
#     return rssm.obs_step(prev_state, prev_action, embed, reset)

#   def img(action, reset):
#     rssm = RSSM(**config)
#     return rssm.imagine(action, reset)
  
#   def obs(embed, action, reset):
#     rssm = RSSM(**config)
#     return rssm.observe(embed, action, reset)
  
#   def feat(state):
#     rssm = RSSM(**config)
#     return rssm.get_feat(state)

#   import jax.numpy as jnp
#   rng = jax.random.PRNGKey(42)
#   b = 2
#   s = 8
#   d = 4
#   ad = 6
#   # x = jnp.ones((b, s, d))

#   '''
#     note: 要求hidden state h.dim == x.dim
#   '''

#   fake_state = RSSMState(
#     mean=jnp.ones((b, 1, d)),
#     std=jnp.ones((b, 1, d)),
#     stoch=jnp.ones((b, 1, d)),
#     deter=None,
#   )
#   # img_step test
#   prev_action = jnp.ones((b, 1, ad))
#   reset = jnp.ones((b, 1))
#   img_net = hk.transform(rssm_img)
#   params = img_net.init(rng, fake_state, prev_action, reset)
#   out_state = img_net.apply(params, rng, fake_state, prev_action, reset)

#   # obs_step test
#   embed = jnp.ones((b, 1, d))
#   obs_net = hk.transform(rssm_obs)
#   params = obs_net.init(rng, fake_state, prev_action, embed, reset)
#   post, prior = obs_net.apply(params, rng, fake_state, prev_action, embed, reset)

#   # imagine test
#   action = jnp.ones((b, s, 1, ad))
#   reset = jnp.ones((b, s, 1))
#   net = hk.transform(img)
#   params = net.init(rng, action, reset)
#   prior = net.apply(params, rng, action, reset)
#   print(prior.mean.shape)
#   print(prior.stoch.shape)

#   # observe test
#   action = jnp.ones((b, s, 1, ad))
#   embed = jnp.ones((b, s, 1, d))
#   reset = jnp.ones((b, s, 1))
#   net = hk.transform(obs)
#   params = net.init(rng, embed, action, reset)
#   post, prior = net.apply(params, rng, embed, action, reset)
#   print(post.mean.shape)

#   # get feat test
#   feat = hk.transform(feat)
#   params = feat.init(rng, post)
#   print(feat.apply(params, rng, post).shape)

#   print("Test Pass.")