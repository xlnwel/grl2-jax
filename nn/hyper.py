from jax import lax
import haiku as hk
from core.typing import dict2AttrDict

from nn.func import mlp
from nn.registry import nn_registry


@nn_registry.register('hyper')
class HyperNet(hk.Module):
  def __init__(
    self, 
    w_in, 
    w_out, 
    name='hyper_net', 
    **config
  ):
    super().__init__(name=name)

    self.w_in = w_in
    self.w_out = w_out

    self.config = dict2AttrDict(config)
    self.config.out_size = self.w_out if self.w_in is None \
      else self.w_in * self.w_out
    
  def __call__(self, x):
    layers = self.build_net()

    x = layers(x)
    if self.w_in is None:
      x = lax.reshape(x, (*x.shape[:-1], self.w_out))
    else:
      x = lax.reshape(x, (*x.shape[:-1], self.w_in, self.w_out))

    return x

  @hk.transparent
  def build_net(self):
    layers = mlp(**self.config)
    return layers


if __name__ == '__main__':
  import jax
  def layer(x):
    layer = HyperNet(2, 3)
    return layer(x)
  rng = jax.random.PRNGKey(42)
  x = jax.random.uniform(rng, (2, 10))
  net = hk.transform(layer)
  params = net.init(rng, x)
  print(params)
  print(net.apply(params, None, x))
  print(hk.experimental.tabulate(net)(x))
