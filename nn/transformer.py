import haiku as hk

from core.typing import AttrDict
from nn.attention import Attention
from nn.mlp import MLP
from nn.utils import dropout
from nn.registry import nn_registry
from jx.tools import jax_utils


@nn_registry.register('tx')
class Transformer(hk.Module):
  def __init__(
    self, 
    n_blocks, 
    final_ln=False, 
    name='tx', 
    inp_dropout=0, 
    mlp_config=AttrDict(
      mlp_ratio=4, 
      activation='gelu', 
    ),
    attn_config={}
  ):
    super().__init__(name=name)
    self.n_blocks = n_blocks
    self.final_ln = final_ln
    self.mlp_config = mlp_config
    self.inp_dropout = inp_dropout
    self.attn_config = attn_config

  def __call__(self, x, mask=None, training=False):
    blocks, ln = self.build_net(x)
    x = dropout(self.inp_dropout, training, x)

    for i, b in enumerate(blocks):
      x = b(x, mask=mask, training=training)
    
    x = ln(x)
    
    return x

  @hk.transparent
  def build_net(self, x):
    blocks = [
      TXEncoderBlock(
        mlp_config=self.mlp_config, 
        attn_config=self.attn_config, 
        name=f'encoder_block{i}'
      )
      for i in range(self.n_blocks)
    ]
    if self.final_ln:
      ln = hk.LayerNorm(-1, True, True)
    else:
      ln = lambda x: x
    return blocks, ln


class TXEncoderBlock(hk.Module):
  def __init__(
    self, 
    mlp_config=AttrDict(
      mlp_ratio=4, 
      activation='gelu', 
    ),
    attn_config=AttrDict(), 
    name='encoder_block', 
  ):
    super().__init__(name=name)
    self.mlp_config = mlp_config
    self.attn_config = attn_config

  def __call__(self, x, mask=None, 
               data_mask=None, query_mask=None, 
               kv_mask=None, training=False):
    attn, mlp = self.build_net(x)

    x = attn(
      x, mask=mask, 
      data_mask=data_mask, query_mask=query_mask, 
      kv_mask=kv_mask, training=training
    )
    x = mlp(x, training=training)

    return x

  @hk.transparent
  def build_net(self, x):
    attn = AttentionBlock(self.attn_config)
    mlp = MLPBlock(**self.mlp_config)
    return attn, mlp


class AttentionBlock(hk.Module):
  def __init__(
    self, 
    attn_config={}, 
    name='attn_block', 
  ):
    super().__init__(name=name)
    self.attn_config = attn_config

  def __call__(self, x, kv=None, mask=None, 
               data_mask=None, query_mask=None, 
               kv_mask=None, training=False):
    if kv is None:
      attn, ln = self.build_net(kv)
      x_ln = ln(x)
    else:
      attn, ln, ln2 = self.build_net(kv)
      x_ln = ln(x)
      kv = ln2(kv)
    x_attn = attn(
      x_ln, kv, mask=mask, 
      data_mask=data_mask, query_mask=query_mask, 
      kv_mask=kv_mask, training=training
    )
    x = x + x_attn

    return x

  @hk.transparent
  def build_net(self, kv):
    attn = Attention(**self.attn_config)
    ln = hk.LayerNorm(-1, True, True)
    if kv is not None:
      ln2 = hk.LayerNorm(-1, True, True)
      return attn, ln, ln2

    return attn, ln


class MLPBlock(hk.Module):
  def __init__(
    self, 
    name='mlp_block', 
    mlp_ratio=4, 
    activation='gelu', 
    dropout=0, 
  ):
    super().__init__(name=name)
    self.mlp_ratio = mlp_ratio
    self.activation = activation
    self.dropout = dropout

  def __call__(self, x, training=False):
    mlp, ln = self.build_net(x)
    
    x_mlp = mlp(ln(x))
    x = x + x_mlp
    x = dropout(self.dropout, training, x)
    
    return x

  @hk.transparent
  def build_net(self, x):
    F = x.shape[-1]
    mlp = MLP(
      [self.mlp_ratio*F], 
      out_size=F, 
      activation=self.activation
    )
    ln = hk.LayerNorm(-1, True, True)

    return mlp, ln
